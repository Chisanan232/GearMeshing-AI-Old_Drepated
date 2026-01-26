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

import logging
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.errors import NodeInterrupt
from langgraph.graph import END, StateGraph

from ..abstraction import AgentConfigSource, get_agent_provider
from ..capabilities.base import CapabilityContext
from ..model_provider import async_create_model_for_role
from ..monitoring_integration import trace_capability_execution
from ..planning.steps import normalize_plan
from ..policy.global_policy import GlobalPolicy
from ..roles import get_role_spec
from ..schemas.domain import (
    AgentEvent,
    AgentEventType,
    AgentRun,
    AgentRunStatus,
    Approval,
    ApprovalDecision,
    CapabilityName,
    ToolInvocation,
    UsageLedgerEntry,
)
from .models import EngineDeps, _GraphState

logger = logging.getLogger(__name__)


class AgentEngine:
    """Execute an agent run plan with policy enforcement and persistence.

    The engine is deliberately small and orchestration-oriented: it delegates
    policy decisions to ``GlobalPolicy`` and delegates actual work to
    capability implementations registered in ``EngineDeps.capabilities``.
    """

    def __init__(self, *, policy: GlobalPolicy, deps: EngineDeps) -> None:
        """
        Initialize the AgentEngine.

        Args:
            policy: The global policy instance for enforcing rules and safety.
            deps: The runtime dependencies (repositories, capabilities, etc.).
        """
        self._policy = policy
        self._deps = deps
        self._checkpointer = deps.checkpointer
        self._graph = self._build_graph()

    def _build_graph(self):
        """Build and compile the LangGraph state machine."""
        g: StateGraph = StateGraph(_GraphState)
        g.add_node("start", self._node_start)
        g.add_node("execute", self._node_execute_next)
        g.add_node("wait_for_approval", self._node_wait_for_approval)
        g.add_node("finish", self._node_finish)

        g.set_entry_point("start")
        g.add_edge("start", "execute")
        g.add_edge("wait_for_approval", "execute")

        g.add_conditional_edges(
            "execute",
            self._route_after_execute,
            {
                "finish": "finish",
                "continue": "execute",
                "wait": "wait_for_approval",
            },
        )
        g.add_edge("finish", END)
        return g.compile(checkpointer=self._checkpointer)

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
        if self._deps.prompt_provider is not None:
            try:
                run.prompt_provider_version = self._deps.prompt_provider.version()
            except Exception:
                run.prompt_provider_version = run.prompt_provider_version

        # Try to create the run, but handle if it already exists (e.g. created by API layer)
        # In the async-first pattern, the API layer creates the run with status PENDING/QUEUED
        # before handing it off to the background worker (which calls this method).
        try:
            await self._deps.runs.create(run)
        except Exception as e:
            # We assume any error here might be due to existence if the repo implementation throws
            # but ideally we should check or have an upsert.
            # For now, let's try to fetch it to confirm it exists.
            existing = await self._deps.runs.get(run.id)
            if existing:
                logger.debug(f"Run {run.id} already exists, proceeding with execution")
                # If we want to ensure we update the passed run object with what's in DB or vice versa?
                # The passed 'run' object is the authoritative source for this execution session.
            else:
                raise e

        await self._deps.events.append(
            AgentEvent(run_id=run.id, type=AgentEventType.run_started, payload={"role": run.role})
        )

        normalized_plan = normalize_plan(plan)

        await self._deps.events.append(
            AgentEvent(run_id=run.id, type=AgentEventType.plan_created, payload={"plan": normalized_plan})
        )

        # Initialize state for LangGraph
        state: _GraphState = {
            "run_id": run.id,
            "plan": normalized_plan,
            "idx": 0,
            "awaiting_approval_id": None,
        }

        # Invoke with thread_id for checkpointing
        config = {"configurable": {"thread_id": run.id}}
        await self._graph.ainvoke(state, config=config)
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

        await self._deps.events.append(
            AgentEvent(
                run_id=run_id,
                type=AgentEventType.approval_resolved,
                payload={
                    "approval_id": approval.id,
                    "decision": approval.decision,
                    "decided_by": approval.decided_by,
                },
            )
        )

        # Resume execution
        # 1. Prepare config
        config = {"configurable": {"thread_id": run_id}}

        # 2. Update state to clear approval
        # We need to tell the graph that the approval is done.
        # In our graph, the node 'wait_for_approval' raised NodeInterrupt.
        # LangGraph resumption works by updating the state or just invoking (if just interruption).
        # But we need to clear 'awaiting_approval_id' in the state.

        # We use update_state to patch the state.
        # Current state has 'awaiting_approval_id' set. We set it to None.
        # We also set a flag to skip the check in next iteration (if needed, but usually clearing ID is enough).

        # Fetch current state to ensure we have a valid checkpoint
        # state_snapshot = await self._graph.aget_state(config)
        # if not state_snapshot.values:
        #    raise ValueError(f"No checkpoint found for run {run_id}")

        await self._graph.aupdate_state(
            config,
            {"awaiting_approval_id": None, "_resume_skip_approval": True},
        )

        await self._deps.events.append(
            AgentEvent(run_id=run_id, type=AgentEventType.state_transition, payload={"resume": True})
        )

        # 3. Resume execution
        # Passing None as input resumes from the current state (which we just updated)
        await self._graph.ainvoke(None, config=config)

    async def _execute_capability_traced(
        self, cap: CapabilityName, cap_impl: Any, ctx: CapabilityContext, args: dict
    ) -> Any:
        """Execute a capability with LangSmith tracing.

        This method wraps capability execution with @trace_capability_execution
        to capture capability invocation details in LangSmith traces.

        Args:
            cap: The capability name
            cap_impl: The capability implementation
            ctx: The capability context
            args: The capability arguments

        Returns:
            The capability execution result
        """

        # Create a traced wrapper for the capability execution
        @trace_capability_execution(cap.value)
        async def _traced_execution():
            return await cap_impl.execute(ctx, args=args)

        return await _traced_execution()

    async def _node_start(self, state: _GraphState) -> _GraphState:
        """Graph entry node. Currently a no-op."""
        return state

    async def _node_wait_for_approval(self, state: _GraphState) -> _GraphState:
        """Wait for approval node.

        This node checks if 'awaiting_approval_id' is set. If so, it raises
        NodeInterrupt to pause execution. Upon resumption (where the ID is cleared),
        it proceeds (returning state to continue to execute node).
        """
        approval_id = state.get("awaiting_approval_id")
        if approval_id:
            raise NodeInterrupt(f"Approval required: {approval_id}")
        return state

    async def _node_execute_next(self, state: _GraphState, config: RunnableConfig) -> _GraphState:
        """Execute the next plan step.

        This node is responsible for:

        - detecting terminal conditions (idx >= len(plan)),
        - executing thought steps (event/artifact emission only),
        - executing action steps (policy/approval/capability).

        Args:
            state: Current graph state containing plan, idx, run_id, etc.
            config: LangGraph runtime configuration with thread_id, checkpoint_id, etc.
        """
        # Extract configuration for tracing and checkpoint management
        configurable = config.get("configurable", {}) if config else {}
        thread_id = configurable.get("thread_id", "unknown")
        checkpoint_id = configurable.get("checkpoint_id")

        run_id = str(state["run_id"])
        run = await self._deps.runs.get(run_id)
        if run is None:
            raise ValueError("run not found")

        plan = list(state.get("plan") or [])
        idx = int(state.get("idx") or 0)

        logger.debug(
            f"[thread={thread_id}] Executing step {idx}/{len(plan)} "
            f"for run {run_id}" + (f" (resuming from checkpoint {checkpoint_id})" if checkpoint_id else "")
        )

        if idx >= len(plan):
            logger.info(f"[thread={thread_id}] Plan execution completed for run {run_id}")
            state["_finished"] = True
            state["_terminal_status"] = AgentRunStatus.succeeded.value
            return state

        step = dict(plan[idx])
        kind = str(step.get("kind") or "action")
        if kind == "thought":
            thought = str(step.get("thought") or "")
            args = dict(step.get("args") or {})

            logger.debug(f"[thread={thread_id}] Executing thought step: {thought} (idx={idx}, role={run.role})")

            output: dict[str, Any] = {}
            prompt_text: str | None = None
            thought_model = self._deps.thought_model

            # If no thought model provided, try to create from configuration
            if thought_model is None and self._deps.role_provider is not None:
                try:
                    thought_model = await async_create_model_for_role(run.role, tenant_id=run.tenant_id)
                    logger.debug(f"Created thought model for role '{run.role}' from configuration")
                except Exception as e:
                    logger.debug(f"Could not create thought model from configuration: {e}")
                    thought_model = None

            if (
                self._deps.prompt_provider is not None
                and self._deps.role_provider is not None
                and thought_model is not None
            ):
                try:
                    role_def = self._deps.role_provider.get(run.role)
                    prompt_key = role_def.cognitive.system_prompt_key
                except Exception:
                    # Fallback to role spec if role provider fails
                    prompt_key = get_role_spec(run.role).system_prompt_key

                # Use abstraction layer for thought execution
                provider = get_agent_provider()

                # Create config source for thought agent using role's prompt key
                config_source = AgentConfigSource(
                    model_config_key="gpt4_default",  # Use default GPT-4 model for thoughts
                    prompt_key=prompt_key,  # Use role-specific system prompt
                    overrides={
                        "output_type": dict,  # Expect structured output
                    },
                    tenant_id=run.tenant_id,  # Pass tenant for multi-tenancy
                    prompt_tenant_id=run.tenant_id,  # Pass tenant for prompts
                )

                try:
                    agent = await provider.create_agent_from_config_source(config_source, use_cache=True)
                    res = await agent.invoke(
                        input_text=f"thought={thought}\nrole={run.role}\nobjective={run.objective}\nargs={args}"
                    )
                    if isinstance(res.content, dict):
                        output = dict(res.content)
                    else:
                        output = {"result": res.content}
                    await agent.cleanup()
                except Exception as e:
                    # If prompt provider or agent creation fails, skip thought execution
                    logger.debug(f"Thought execution failed due to prompt provider error: {e}")
                    output = None

            await self._deps.events.append(
                AgentEvent(
                    run_id=run_id,
                    type=AgentEventType.thought_executed,
                    payload={"thought": thought, "idx": idx},
                )
            )
            payload: dict[str, Any] = {"kind": "thought", "thought": thought, "idx": idx, "data": args}
            if output:
                payload["output"] = output
            if prompt_text is not None:
                payload["system_prompt_key"] = get_role_spec(run.role).system_prompt_key
            await self._deps.events.append(
                AgentEvent(
                    run_id=run_id,
                    type=AgentEventType.artifact_created,
                    payload=payload,
                )
            )
            state["idx"] = idx + 1
            return state

        if kind != "action":
            raise ValueError(f"unknown step kind: {kind}")

        cap = CapabilityName(step["capability"])
        args = dict(step.get("args") or {})
        logical_tool = cast(str | None, step.get("logical_tool"))

        logger.debug(
            f"[thread={thread_id}] Executing action step: {cap.value} (idx={idx}, logical_tool={logical_tool})"
        )

        if cap == CapabilityName.mcp_call:
            server_id = step.get("server_id") or args.get("server_id")
            tool_name = step.get("tool_name") or args.get("tool_name")
            if server_id is not None and "server_id" not in args:
                args["server_id"] = server_id
            if tool_name is not None and "tool_name" not in args:
                args["tool_name"] = tool_name
            if logical_tool is None and tool_name is not None:
                logical_tool = str(tool_name)

            # MCP tool discovery/metadata should come from the MCP info provider.
            # The provider layer is intentionally tools-only and may apply its own
            # policies (allowed servers/tools, read-only) when agent_id is supplied.
            info = self._deps.mcp_info_provider
            if info is not None and server_id and tool_name:
                try:
                    tools = await info.list_tools(str(server_id), agent_id=str(run.role))
                    meta = None
                    for t in tools or []:
                        if getattr(t, "name", None) == tool_name:
                            meta = t
                            break
                    mut = getattr(meta, "mutating", None) if meta is not None else None
                    if isinstance(mut, bool):
                        args["_mcp_tool_mutating"] = mut
                except Exception:
                    pass

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

            # Return state update to trigger transition to wait node
            state["awaiting_approval_id"] = approval.id
            return state

        cap_impl = self._deps.capabilities.get(cap)
        ctx = CapabilityContext(run=run, policy=self._policy, deps=self._deps)
        await self._deps.events.append(
            AgentEvent(run_id=run_id, type=AgentEventType.capability_requested, payload={"capability": cap})
        )

        # Execute capability with LangSmith tracing
        res = await self._execute_capability_traced(cap, cap_impl, ctx, args)

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
                server_id=str(step.get("server_id") or args.get("server_id") or ""),
                tool_name=str(step.get("tool_name") or args.get("tool_name") or logical_tool or cap.value),
                args=args,
                ok=res.ok,
                result=res.output,
                risk=decision.risk,
            )
        )

        if self._deps.usage is not None:
            usage_payload = dict(res.output.get("usage") or {}) if isinstance(res.output.get("usage"), dict) else {}
            prompt_tokens = usage_payload.get("prompt_tokens", res.output.get("prompt_tokens"))
            completion_tokens = usage_payload.get("completion_tokens", res.output.get("completion_tokens"))
            total_tokens = usage_payload.get("total_tokens", res.output.get("total_tokens"))
            cost_usd = usage_payload.get("cost_usd", res.output.get("cost_usd"))
            provider = usage_payload.get("provider", res.output.get("provider"))
            model = usage_payload.get("model", res.output.get("model"))

            if any(v is not None for v in (prompt_tokens, completion_tokens, total_tokens, cost_usd, provider, model)):
                entry = UsageLedgerEntry(
                    run_id=run_id,
                    provider=str(provider) if provider is not None else None,
                    model=str(model) if model is not None else None,
                    prompt_tokens=int(prompt_tokens or 0),
                    completion_tokens=int(completion_tokens or 0),
                    total_tokens=int(total_tokens or (int(prompt_tokens or 0) + int(completion_tokens or 0))),
                    cost_usd=float(cost_usd) if cost_usd is not None else None,
                )
                await self._deps.usage.append(entry)
                await self._deps.events.append(
                    AgentEvent(
                        run_id=run_id,
                        type=AgentEventType.usage_recorded,
                        payload={
                            "usage_id": entry.id,
                            "provider": entry.provider,
                            "model": entry.model,
                            "prompt_tokens": entry.prompt_tokens,
                            "completion_tokens": entry.completion_tokens,
                            "total_tokens": entry.total_tokens,
                            "cost_usd": entry.cost_usd,
                        },
                    )
                )

        state["idx"] = idx + 1
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
        """Route to finish or continue after executing a step."""
        if state.get("_finished"):
            return "finish"
        if state.get("awaiting_approval_id"):
            return "wait"
        return "continue"
