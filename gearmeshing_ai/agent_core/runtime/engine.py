from __future__ import annotations

"""LangGraph runtime engine.

``AgentEngine`` executes a normalized plan produced by the planning subsystem.

Execution model
--------------

- The engine runs a LangGraph state machine over a mutable ``_GraphState``.
- Each iteration executes exactly one plan step at index ``idx``.

Thought vs Action
-----------------

- Thought steps (``kind="thought"``) are non-side-effecting. The engine records
  them as events/artifacts and advances the index.
- Action steps (``kind="action"``) are side-effecting. The engine:

  1. Evaluates policy (allow/deny + risk + approval requirement).
  2. Validates safety constraints (e.g. argument size).
  3. Optionally pauses for approval and stores a checkpoint.
  4. Executes the capability and persists tool invocation logs.

Pause/resume
------------

When an action requires approval, the engine writes an ``Approval`` and a
``Checkpoint`` and transitions the run to ``paused_for_approval``. Resuming a
run restores the checkpointed state, clears the awaiting approval id, and
continues execution.
"""

from typing import Any, cast

from langgraph.graph import END, StateGraph

from ..capabilities.base import CapabilityContext
from ..planning.steps import normalize_plan
from ..policy.global_policy import GlobalPolicy
from ..schemas.domain import (
    AgentEvent,
    AgentEventType,
    AgentRun,
    AgentRunStatus,
    Approval,
    ApprovalDecision,
    CapabilityName,
    Checkpoint,
    ToolInvocation,
)
from .models import EngineDeps, _GraphState


class AgentEngine:
    """Execute an agent run plan with policy enforcement and persistence.

    The engine is deliberately small and orchestration-oriented: it delegates
    policy decisions to ``GlobalPolicy`` and delegates actual work to
    capability implementations registered in ``EngineDeps.capabilities``.
    """

    def __init__(self, *, policy: GlobalPolicy, deps: EngineDeps) -> None:
        self._policy = policy
        self._deps = deps
        self._graph = self._build_graph()

    def _build_graph(self):
        """Build and compile the LangGraph state machine."""
        g: StateGraph = StateGraph(_GraphState)
        g.add_node("start", self._node_start)
        g.add_node("execute", self._node_execute_next)
        g.add_node("pause_for_approval", self._node_pause_for_approval)
        g.add_node("finish", self._node_finish)

        g.set_entry_point("start")
        g.add_edge("start", "execute")

        g.add_conditional_edges(
            "execute",
            self._route_after_execute,
            {
                "pause": "pause_for_approval",
                "finish": "finish",
                "continue": "execute",
            },
        )
        g.add_edge("pause_for_approval", END)
        g.add_edge("finish", END)
        return g.compile()

    async def start_run(
        self,
        *,
        run: AgentRun,
        plan: list[dict[str, Any]],
    ) -> str:
        """Persist a new run and execute its plan.

        Notes
        -----
        The provided plan is normalized via ``normalize_plan`` before execution
        to enforce the thought/action boundary and maintain backward
        compatibility.
        """
        await self._deps.runs.create(run)
        await self._deps.events.append(
            AgentEvent(run_id=run.id, type=AgentEventType.run_started, payload={"role": run.role})
        )

        normalized_plan = normalize_plan(plan)

        state: _GraphState = {
            "run_id": run.id,
            "plan": normalized_plan,
            "idx": 0,
            "awaiting_approval_id": None,
        }
        await self._graph.ainvoke(state)
        return run.id

    async def resume_run(self, *, run_id: str, approval_id: str) -> None:
        """Resume a paused run after approval.

        The engine validates that the referenced approval exists and has been
        approved, restores the latest checkpoint state, and continues execution
        from that state.
        """
        approval = await self._deps.approvals.get(approval_id)
        if approval is None:
            raise ValueError("approval not found")
        if approval.decision != ApprovalDecision.approved:
            raise ValueError("approval not approved")

        cp = await self._deps.checkpoints.latest(run_id)
        if cp is None:
            raise ValueError("no checkpoint")

        state = cast(_GraphState, dict(cp.state))
        state["plan"] = normalize_plan(list(state.get("plan") or []))
        state["awaiting_approval_id"] = None
        state["_resume_skip_approval"] = True
        await self._deps.events.append(
            AgentEvent(run_id=run_id, type=AgentEventType.state_transition, payload={"resume": True})
        )
        await self._graph.ainvoke(state)

    async def _node_start(self, state: _GraphState) -> _GraphState:
        """Graph entry node. Currently a no-op."""
        return state

    async def _node_execute_next(self, state: _GraphState) -> _GraphState:
        """Execute the next plan step.

        This node is responsible for:

        - detecting terminal conditions (idx >= len(plan)),
        - executing thought steps (event/artifact emission only),
        - executing action steps (policy/approval/capability).
        """
        run_id = str(state["run_id"])
        run = await self._deps.runs.get(run_id)
        if run is None:
            raise ValueError("run not found")

        plan = list(state.get("plan") or [])
        idx = int(state.get("idx") or 0)
        if idx >= len(plan):
            state["_finished"] = True
            state["_terminal_status"] = AgentRunStatus.succeeded.value
            return state

        step = dict(plan[idx])
        kind = str(step.get("kind") or "action")
        if kind == "thought":
            thought = str(step.get("thought") or "")
            await self._deps.events.append(
                AgentEvent(
                    run_id=run_id,
                    type=AgentEventType.thought_executed,
                    payload={"thought": thought, "idx": idx},
                )
            )
            await self._deps.events.append(
                AgentEvent(
                    run_id=run_id,
                    type=AgentEventType.artifact_created,
                    payload={"kind": "thought", "thought": thought, "idx": idx, "data": dict(step.get("args") or {})},
                )
            )
            state["idx"] = idx + 1
            return state

        if kind != "action":
            raise ValueError(f"unknown step kind: {kind}")

        cap = CapabilityName(step["capability"])
        args = dict(step.get("args") or {})
        logical_tool = cast(str | None, step.get("logical_tool"))

        decision = self._policy.decide(cap, args=args, logical_tool=logical_tool)
        if decision.block:
            await self._deps.events.append(
                AgentEvent(
                    run_id=run_id,
                    type=AgentEventType.run_failed,
                    payload={"reason": decision.block_reason, "capability": cap},
                )
            )
            await self._deps.runs.update_status(run_id, status=AgentRunStatus.failed.value)
            state["_finished"] = True
            state["_terminal_status"] = AgentRunStatus.failed.value
            return state

        err = self._policy.validate_tool_args(args)
        if err is not None:
            await self._deps.events.append(
                AgentEvent(run_id=run_id, type=AgentEventType.run_failed, payload={"reason": err})
            )
            await self._deps.runs.update_status(run_id, status=AgentRunStatus.failed.value)
            state["_finished"] = True
            state["_terminal_status"] = AgentRunStatus.failed.value
            return state

        skip_approval = bool(state.get("_resume_skip_approval"))
        state.pop("_resume_skip_approval", None)
        if decision.require_approval and not skip_approval:
            approval = Approval(
                run_id=run_id,
                risk=decision.risk,
                capability=cap,
                reason=f"Approval required for {cap} at risk={decision.risk}",
            )
            await self._deps.approvals.create(approval)
            await self._deps.runs.update_status(run_id, status=AgentRunStatus.paused_for_approval.value)
            await self._deps.events.append(
                AgentEvent(
                    run_id=run_id,
                    type=AgentEventType.approval_requested,
                    payload={"approval_id": approval.id, "capability": cap, "risk": decision.risk},
                )
            )

            cp = Checkpoint(
                run_id=run_id, node="pause_for_approval", state=dict(state, awaiting_approval_id=approval.id)
            )
            await self._deps.checkpoints.save(cp)
            await self._deps.events.append(
                AgentEvent(run_id=run_id, type=AgentEventType.checkpoint_saved, payload={"checkpoint_id": cp.id})
            )
            state["awaiting_approval_id"] = approval.id
            return state

        cap_impl = self._deps.capabilities.get(cap)
        ctx = CapabilityContext(run=run, policy=self._policy, deps=self._deps)
        await self._deps.events.append(
            AgentEvent(run_id=run_id, type=AgentEventType.capability_requested, payload={"capability": cap})
        )
        res = await cap_impl.execute(ctx, args=args)

        await self._deps.events.append(
            AgentEvent(
                run_id=run_id,
                type=AgentEventType.capability_executed,
                payload={"capability": cap, "ok": res.ok, "output": res.output},
            )
        )

        await self._deps.tool_invocations.append(
            ToolInvocation(
                run_id=run_id,
                server_id=str(step.get("server_id") or ""),
                tool_name=str(step.get("tool_name") or logical_tool or cap.value),
                args=args,
                ok=res.ok,
                result=res.output,
                risk=decision.risk,
            )
        )

        state["idx"] = idx + 1
        return state

    async def _node_pause_for_approval(self, state: _GraphState) -> _GraphState:
        """Pause node.

        The graph transitions to END after this node; pause/resume coordination
        is managed outside the graph via persisted checkpoints.
        """
        return state

    async def _node_finish(self, state: _GraphState) -> _GraphState:
        """Finish node.

        Updates the run status and emits the run completion event.
        """
        run_id = str(state["run_id"])
        status = str(state.get("_terminal_status") or AgentRunStatus.succeeded.value)
        await self._deps.runs.update_status(run_id, status=status)
        await self._deps.events.append(AgentEvent(run_id=run_id, type=AgentEventType.run_completed, payload={}))
        return state

    def _route_after_execute(self, state: _GraphState) -> str:
        """Route to pause/finish/continue after executing a step."""
        if state.get("awaiting_approval_id"):
            return "pause"
        if state.get("_finished"):
            return "finish"
        return "continue"
