from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest
from langgraph.checkpoint.memory import MemorySaver

from gearmeshing_ai.agent_core.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityResult,
)
from gearmeshing_ai.agent_core.capabilities.registry import CapabilityRegistry
from gearmeshing_ai.agent_core.policy.global_policy import GlobalPolicy
from gearmeshing_ai.agent_core.policy.models import PolicyConfig
from gearmeshing_ai.agent_core.runtime.engine import AgentEngine
from gearmeshing_ai.agent_core.runtime.models import EngineDeps, _GraphState
from gearmeshing_ai.agent_core.schemas.domain import (
    AgentEvent,
    AgentEventType,
    AgentRun,
    AgentRunStatus,
    Approval,
    ApprovalDecision,
    CapabilityName,
    RiskLevel,
)


class _GraphSpy:
    def __init__(self) -> None:
        self.invocations: List[tuple[Optional[_GraphState], Optional[Dict[str, Any]]]] = []
        self.state_updates: List[tuple[Dict[str, Any], Dict[str, Any]]] = []

    async def ainvoke(self, state: Optional[_GraphState], config: Optional[Dict[str, Any]] = None) -> None:
        self.invocations.append((state, config))

    async def aupdate_state(self, config: Dict[str, Any], values: Dict[str, Any]) -> None:
        self.state_updates.append((config, values))


@pytest.mark.asyncio
async def test_start_run_prompt_provider_version_error_is_swallowed(repos, registry, policy: GlobalPolicy) -> None:
    reg, _cap = registry

    class _PromptProvider:
        def version(self) -> str:
            raise RuntimeError("boom")

    deps = EngineDeps(
        runs=repos["runs"],
        events=repos["events"],
        approvals=repos["approvals"],
        checkpoints=repos["checkpoints"],
        tool_invocations=repos["tool_invocations"],
        usage=repos["usage"],
        capabilities=reg,
        prompt_provider=_PromptProvider(),  # type: ignore[arg-type]
        checkpointer=MemorySaver(),
    )
    engine = AgentEngine(policy=policy, deps=deps)
    graph_spy = _GraphSpy()
    engine._graph = graph_spy

    run = AgentRun(role="dev", objective="x", prompt_provider_version=None)
    await engine.start_run(run=run, plan=[{"capability": CapabilityName.summarize.value, "args": {"text": "hi"}}])

    assert run.prompt_provider_version is None
    assert graph_spy.invocations
    assert graph_spy.invocations[0][0] is not None  # Start run passes state


@pytest.mark.asyncio
async def test_start_run_emits_plan_created_event(repos, registry, policy: GlobalPolicy) -> None:
    reg, _cap = registry
    deps = EngineDeps(
        runs=repos["runs"],
        events=repos["events"],
        approvals=repos["approvals"],
        checkpoints=repos["checkpoints"],
        tool_invocations=repos["tool_invocations"],
        usage=repos["usage"],
        capabilities=reg,
        checkpointer=MemorySaver(),
    )
    engine = AgentEngine(policy=policy, deps=deps)

    graph_spy = _GraphSpy()
    engine._graph = graph_spy

    run = AgentRun(role="dev", objective="x")
    await engine.start_run(run=run, plan=[{"capability": CapabilityName.summarize.value, "args": {"text": "hi"}}])

    assert any(e.type == AgentEventType.plan_created for e in repos["events"].events)
    # Checkpoint is saved by LangGraph checkpointer, which is mocked out by _GraphSpy here.
    # So we don't expect checkpoint_saved event or repos["checkpoints"].saved in this unit test.
    plan_events = [e for e in repos["events"].events if e.type == AgentEventType.plan_created]
    assert len(plan_events) == 1
    assert plan_events[0].payload.get("plan")


class _RunsRepo:
    def __init__(self) -> None:
        self.by_id: Dict[str, AgentRun] = {}
        self.status_updates: List[tuple[str, str]] = []

    async def create(self, run: AgentRun) -> None:
        self.by_id[run.id] = run

    async def get(self, run_id: str) -> Optional[AgentRun]:
        return self.by_id.get(run_id)

    async def update_status(self, run_id: str, *, status: str) -> None:
        self.status_updates.append((run_id, status))
        run = self.by_id.get(run_id)
        if run is not None:
            run.status = AgentRunStatus(status)


class _EventsRepo:
    def __init__(self) -> None:
        self.events: List[AgentEvent] = []

    async def append(self, event: AgentEvent) -> None:
        self.events.append(event)


class _ApprovalsRepo:
    def __init__(self) -> None:
        self.by_id: Dict[str, Approval] = {}
        self.created: List[str] = []

    async def create(self, approval: Approval) -> None:
        self.by_id[approval.id] = approval
        self.created.append(approval.id)

    async def get(self, approval_id: str) -> Optional[Approval]:
        return self.by_id.get(approval_id)

    async def resolve(self, approval_id: str, *, decision: str, decided_by: Optional[str]) -> None:
        approval = self.by_id.get(approval_id)
        if approval is None:
            return
        approval.decision = ApprovalDecision(decision)
        approval.decided_by = decided_by


@dataclass
class _Checkpoint:
    id: str
    state: Dict[str, Any]
    node: str = "node"


class _CheckpointsRepo:
    def __init__(self) -> None:
        self.latest_by_run_id: Dict[str, _Checkpoint] = {}
        self.saved: List[_Checkpoint] = []

    async def save(self, cp) -> None:
        chk = _Checkpoint(id=cp.id, state=dict(cp.state), node=cp.node)
        self.latest_by_run_id[cp.run_id] = chk
        self.saved.append(chk)

    async def latest(self, run_id: str):
        return self.latest_by_run_id.get(run_id)


class _ToolInvocationsRepo:
    def __init__(self) -> None:
        self.invocations: List[Any] = []

    async def append(self, invocation) -> None:
        self.invocations.append(invocation)


class _UsageRepo:
    def __init__(self) -> None:
        self.entries: List[Any] = []

    async def append(self, usage) -> None:
        self.entries.append(usage)


class _DummyCapability(Capability):
    name = CapabilityName.summarize

    def __init__(self, *, ok: bool = True, output: Optional[Dict[str, Any]] = None) -> None:
        self._ok = ok
        self._output = output or {"result": "ok"}
        self.calls: List[Dict[str, Any]] = []

    async def execute(self, ctx: CapabilityContext, *, args: Dict[str, Any]) -> CapabilityResult:
        self.calls.append({"run_id": ctx.run.id, "args": dict(args)})
        return CapabilityResult(ok=self._ok, output=dict(self._output))


@pytest.fixture
def repos() -> Dict[str, Any]:
    return {
        "runs": _RunsRepo(),
        "events": _EventsRepo(),
        "approvals": _ApprovalsRepo(),
        "checkpoints": _CheckpointsRepo(),
        "tool_invocations": _ToolInvocationsRepo(),
        "usage": _UsageRepo(),
    }


@pytest.fixture
def registry() -> tuple[CapabilityRegistry, _DummyCapability]:
    reg = CapabilityRegistry()
    cap = _DummyCapability()
    reg.register(cap)
    return reg, cap


@pytest.fixture
def policy() -> GlobalPolicy:
    cfg = PolicyConfig()
    cfg.tool_policy.allowed_capabilities = {CapabilityName.summarize}
    return GlobalPolicy(cfg)


@pytest.fixture
def engine_runtime(repos, registry, policy: GlobalPolicy) -> AgentEngine:
    reg, _cap = registry
    deps = EngineDeps(
        runs=repos["runs"],
        events=repos["events"],
        approvals=repos["approvals"],
        checkpoints=repos["checkpoints"],
        tool_invocations=repos["tool_invocations"],
        usage=repos["usage"],
        capabilities=reg,
        checkpointer=MemorySaver(),
    )
    return AgentEngine(policy=policy, deps=deps)


@pytest.mark.asyncio
async def test_node_execute_next_persists_usage_when_capability_emits_usage(
    repos, registry, policy: GlobalPolicy
) -> None:
    reg, _cap = registry
    usage_cap = _DummyCapability(
        output={
            "result": "ok",
            "usage": {"provider": "p", "model": "m", "prompt_tokens": 1, "completion_tokens": 2, "cost_usd": 0.01},
        }
    )
    reg = CapabilityRegistry()
    reg.register(usage_cap)

    deps = EngineDeps(
        runs=repos["runs"],
        events=repos["events"],
        approvals=repos["approvals"],
        checkpoints=repos["checkpoints"],
        tool_invocations=repos["tool_invocations"],
        usage=repos["usage"],
        capabilities=reg,
        checkpointer=MemorySaver(),
    )
    engine = AgentEngine(policy=policy, deps=deps)

    run = AgentRun(role="dev", objective="x")
    await repos["runs"].create(run)

    state: _GraphState = {
        "run_id": run.id,
        "plan": [{"kind": "action", "capability": CapabilityName.summarize.value, "args": {"text": "hi"}}],
        "idx": 0,
        "awaiting_approval_id": None,
    }
    await engine._node_execute_next(state, config={})

    assert repos["usage"].entries
    assert any(e.type == AgentEventType.usage_recorded for e in repos["events"].events)


@pytest.mark.asyncio
async def test_node_execute_next_run_missing_raises(engine_runtime: AgentEngine) -> None:
    state: _GraphState = {"run_id": "missing", "plan": [], "idx": 0, "awaiting_approval_id": None}
    with pytest.raises(ValueError, match="run not found"):
        await engine_runtime._node_execute_next(state, config={})


@pytest.mark.asyncio
async def test_node_execute_next_marks_finished_when_idx_out_of_range(engine_runtime: AgentEngine, repos) -> None:
    run = AgentRun(role="dev", objective="x")
    await repos["runs"].create(run)

    state: _GraphState = {"run_id": run.id, "plan": [], "idx": 0, "awaiting_approval_id": None}
    out = await engine_runtime._node_execute_next(state, config={})
    assert out.get("_finished") is True


@pytest.mark.asyncio
async def test_node_execute_next_thought_step_emits_artifact_and_never_invokes_tools(
    engine_runtime: AgentEngine, repos
) -> None:
    run = AgentRun(role="dev", objective="x")
    await repos["runs"].create(run)

    plan = [{"kind": "thought", "thought": "design", "args": {"note": "n"}}]
    state: _GraphState = {"run_id": run.id, "plan": plan, "idx": 0, "awaiting_approval_id": None}
    out = await engine_runtime._node_execute_next(state, config={})

    assert out["idx"] == 1
    assert repos["approvals"].created == []
    assert repos["tool_invocations"].invocations == []
    assert any(e.type == AgentEventType.thought_executed for e in repos["events"].events)
    assert any(e.type == AgentEventType.artifact_created for e in repos["events"].events)


@pytest.mark.asyncio
async def test_node_execute_next_blocked_capability_fails_run(repos, registry) -> None:
    reg, _cap = registry
    cfg = PolicyConfig()
    cfg.tool_policy.allowed_capabilities = {CapabilityName.summarize}
    policy = GlobalPolicy(cfg)
    deps = EngineDeps(
        runs=repos["runs"],
        events=repos["events"],
        approvals=repos["approvals"],
        checkpoints=repos["checkpoints"],
        tool_invocations=repos["tool_invocations"],
        capabilities=reg,
        checkpointer=MemorySaver(),
    )
    engine_runtime = AgentEngine(policy=policy, deps=deps)

    run = AgentRun(role="dev", objective="x")
    await repos["runs"].create(run)

    plan = [{"kind": "action", "capability": CapabilityName.shell_exec.value, "args": {"cmd": "echo hi"}}]
    state: _GraphState = {"run_id": run.id, "plan": plan, "idx": 0, "awaiting_approval_id": None}
    out = await engine_runtime._node_execute_next(state, config={})

    assert out.get("_finished") is True
    assert out.get("_terminal_status") == AgentRunStatus.failed.value
    assert repos["runs"].status_updates[-1] == (run.id, AgentRunStatus.failed.value)


@pytest.mark.asyncio
async def test_node_execute_next_invalid_args_fails_run(repos, registry) -> None:
    reg, _cap = registry
    cfg = PolicyConfig()
    cfg.tool_policy.allowed_capabilities = {CapabilityName.summarize}
    cfg.safety_policy.max_tool_args_bytes = 1
    policy = GlobalPolicy(cfg)
    deps = EngineDeps(
        runs=repos["runs"],
        events=repos["events"],
        approvals=repos["approvals"],
        checkpoints=repos["checkpoints"],
        tool_invocations=repos["tool_invocations"],
        capabilities=reg,
        checkpointer=MemorySaver(),
    )
    engine_runtime = AgentEngine(policy=policy, deps=deps)

    run = AgentRun(role="dev", objective="x")
    await repos["runs"].create(run)

    plan = [{"kind": "action", "capability": CapabilityName.summarize.value, "args": {"text": "xx"}}]
    state: _GraphState = {"run_id": run.id, "plan": plan, "idx": 0, "awaiting_approval_id": None}
    out = await engine_runtime._node_execute_next(state, config={})

    assert out.get("_finished") is True
    assert out.get("_terminal_status") == AgentRunStatus.failed.value
    assert repos["runs"].status_updates[-1] == (run.id, AgentRunStatus.failed.value)
    assert any(e.type == AgentEventType.run_failed for e in repos["events"].events)


from langgraph.errors import NodeInterrupt

# ...

@pytest.mark.asyncio
async def test_node_execute_next_requires_approval_creates_checkpoint_and_pauses(repos, registry) -> None:
    reg, _cap = registry
    cfg = PolicyConfig()
    cfg.tool_policy.allowed_capabilities = {CapabilityName.summarize}
    cfg.approval_policy.require_for_risk_at_or_above = RiskLevel.low
    policy = GlobalPolicy(cfg)
    deps = EngineDeps(
        runs=repos["runs"],
        events=repos["events"],
        approvals=repos["approvals"],
        checkpoints=repos["checkpoints"],
        tool_invocations=repos["tool_invocations"],
        capabilities=reg,
        checkpointer=MemorySaver(),
    )
    engine_runtime = AgentEngine(policy=policy, deps=deps)

    run = AgentRun(role="dev", objective="x")
    await repos["runs"].create(run)

    plan = [{"kind": "action", "capability": CapabilityName.summarize.value, "args": {"text": "hello"}}]
    state: _GraphState = {"run_id": run.id, "plan": plan, "idx": 0, "awaiting_approval_id": None}
    
    # Execute next step - should detect approval needed and return state update
    out = await engine_runtime._node_execute_next(state, config={})
    
    # Verify state updated with approval ID
    assert out.get("awaiting_approval_id") is not None
    
    # Verify approval created
    assert repos["runs"].status_updates[-1] == (run.id, AgentRunStatus.paused_for_approval.value)
    assert repos["approvals"].created
    assert any(e.type == AgentEventType.approval_requested for e in repos["events"].events)
    
    # Note: Checkpoint is saved by LangGraph runtime (checkpointer) via the graph execution loop.


@pytest.mark.asyncio
async def test_node_wait_for_approval_raises_interrupt(engine_runtime: AgentEngine) -> None:
    state: _GraphState = {"run_id": "r", "plan": [], "idx": 0, "awaiting_approval_id": "app_1"}
    with pytest.raises(NodeInterrupt, match="Approval required: app_1"):
        await engine_runtime._node_wait_for_approval(state)


@pytest.mark.asyncio
async def test_node_wait_for_approval_does_nothing_if_no_id(engine_runtime: AgentEngine) -> None:
    state: _GraphState = {"run_id": "r", "plan": [], "idx": 0, "awaiting_approval_id": None}
    out = await engine_runtime._node_wait_for_approval(state)
    assert out == state



@pytest.mark.asyncio
async def test_node_execute_next_executes_capability_and_appends_tool_invocation(
    engine_runtime: AgentEngine,
    repos,
    registry,
) -> None:
    _reg, cap = registry
    run = AgentRun(role="dev", objective="x")
    await repos["runs"].create(run)

    plan = [{"kind": "action", "capability": CapabilityName.summarize.value, "args": {"text": "hello"}}]
    state: _GraphState = {"run_id": run.id, "plan": plan, "idx": 0, "awaiting_approval_id": None}
    out = await engine_runtime._node_execute_next(state, config={})

    assert out["idx"] == 1
    assert cap.calls == [{"run_id": run.id, "args": {"text": "hello"}}]
    assert len(repos["tool_invocations"].invocations) == 1
    assert any(e.type == AgentEventType.capability_requested for e in repos["events"].events)
    assert any(e.type == AgentEventType.capability_executed for e in repos["events"].events)


@pytest.mark.asyncio
async def test_node_finish_marks_succeeded_and_emits_completed_event(engine_runtime: AgentEngine, repos) -> None:
    run = AgentRun(role="dev", objective="x")
    await repos["runs"].create(run)

    state: _GraphState = {"run_id": run.id, "plan": [], "idx": 0, "awaiting_approval_id": None}
    await engine_runtime._node_finish(state)

    assert repos["runs"].status_updates[-1] == (run.id, AgentRunStatus.succeeded.value)
    assert any(e.type == AgentEventType.run_completed for e in repos["events"].events)


@pytest.mark.asyncio
async def test_node_finish_honors_terminal_status_failed(engine_runtime: AgentEngine, repos) -> None:
    run = AgentRun(role="dev", objective="x")
    await repos["runs"].create(run)

    state: _GraphState = {
        "run_id": run.id,
        "plan": [],
        "idx": 0,
        "awaiting_approval_id": None,
        "_terminal_status": AgentRunStatus.failed.value,
    }
    await engine_runtime._node_finish(state)

    assert repos["runs"].status_updates[-1] == (run.id, AgentRunStatus.failed.value)


def test_route_after_execute_prefers_finish_then_continue(engine_runtime: AgentEngine) -> None:
    # 'pause' route is no longer used; NodeInterrupt pauses execution inside the node.
    # So we only check for finish vs continue.
    assert (
        engine_runtime._route_after_execute(
            {"run_id": "r", "plan": [], "idx": 0, "awaiting_approval_id": None, "_finished": True}
        )
        == "finish"
    )
    assert (
        engine_runtime._route_after_execute({"run_id": "r", "plan": [], "idx": 0, "awaiting_approval_id": None})
        == "continue"
    )


@pytest.mark.asyncio
async def test_resume_run_validation_errors(repos, registry) -> None:
    reg, _cap = registry
    cfg = PolicyConfig()
    cfg.tool_policy.allowed_capabilities = {CapabilityName.summarize}
    policy = GlobalPolicy(cfg)
    deps = EngineDeps(
        runs=repos["runs"],
        events=repos["events"],
        approvals=repos["approvals"],
        checkpoints=repos["checkpoints"],
        tool_invocations=repos["tool_invocations"],
        capabilities=reg,
        checkpointer=MemorySaver(),
    )
    engine_runtime = AgentEngine(policy=policy, deps=deps)

    with pytest.raises(ValueError, match="approval not found"):
        await engine_runtime.resume_run(run_id="r", approval_id="missing")

    approval = Approval(run_id="r", risk=RiskLevel.low, capability=CapabilityName.summarize, reason="x")
    await repos["approvals"].create(approval)
    await repos["approvals"].resolve(approval.id, decision=ApprovalDecision.rejected.value, decided_by=None)

    with pytest.raises(ValueError, match="approval not approved"):
        await engine_runtime.resume_run(run_id="r", approval_id=approval.id)

    await repos["approvals"].resolve(approval.id, decision=ApprovalDecision.approved.value, decided_by="t")
    
    # Mock graph to raise error during update_state (simulating no checkpoint found)
    class _MockGraph:
        async def aupdate_state(self, *args, **kwargs):
            raise ValueError("no checkpoint")
        async def ainvoke(self, *args, **kwargs):
            pass
            
    engine_runtime._graph = _MockGraph()
    
    with pytest.raises(ValueError, match="no checkpoint"):
        await engine_runtime.resume_run(run_id="r", approval_id=approval.id)


@pytest.mark.asyncio
async def test_resume_run_happy_path_restores_checkpoint_and_invokes_graph(repos, registry) -> None:
    reg, _cap = registry
    cfg = PolicyConfig()
    cfg.tool_policy.allowed_capabilities = {CapabilityName.summarize}
    policy = GlobalPolicy(cfg)
    deps = EngineDeps(
        runs=repos["runs"],
        events=repos["events"],
        approvals=repos["approvals"],
        checkpoints=repos["checkpoints"],
        tool_invocations=repos["tool_invocations"],
        capabilities=reg,
        checkpointer=MemorySaver(),
    )
    engine_runtime = AgentEngine(policy=policy, deps=deps)

    graph_spy = _GraphSpy()
    engine_runtime._graph = graph_spy

    run = AgentRun(role="dev", objective="resume")
    await repos["runs"].create(run)

    approval = Approval(run_id=run.id, risk=RiskLevel.low, capability=CapabilityName.summarize, reason="ok")
    await repos["approvals"].create(approval)
    await repos["approvals"].resolve(approval.id, decision=ApprovalDecision.approved.value, decided_by="tester")

    restored_state: _GraphState = {
        "run_id": run.id,
        "plan": [{"kind": "action", "capability": CapabilityName.summarize.value, "args": {"text": "hello"}}],
        "idx": 0,
        "awaiting_approval_id": approval.id,
    }

    class _CheckpointObj:
        def __init__(self) -> None:
            self.id = "cp1"
            self.run_id = run.id
            self.state = dict(restored_state)
            self.node = "node"

    repos["checkpoints"].latest_by_run_id[run.id] = _Checkpoint(id="cp1", state=dict(restored_state), node="node")

    async def _latest(_run_id: str):
        if _run_id != run.id:
            return None
        return _CheckpointObj()

    repos["checkpoints"].latest = _latest

    await engine_runtime.resume_run(run_id=run.id, approval_id=approval.id)

    assert len(graph_spy.invocations) == 1
    state_arg, config_arg = graph_spy.invocations[0]
    
    # Resume calls ainvoke(None, config=...)
    assert state_arg is None
    assert config_arg["configurable"]["thread_id"] == run.id
    
    # Verify state update was called on the graph
    assert len(graph_spy.state_updates) == 1
    config_update, values_update = graph_spy.state_updates[0]
    assert config_update["configurable"]["thread_id"] == run.id
    assert values_update["awaiting_approval_id"] is None
    assert values_update["_resume_skip_approval"] is True
    
    assert any(e.type == AgentEventType.state_transition for e in repos["events"].events)
    assert any(e.type == AgentEventType.approval_resolved for e in repos["events"].events)
    resolved = [e for e in repos["events"].events if e.type == AgentEventType.approval_resolved][0]
    assert resolved.payload.get("approval_id") == approval.id
    assert resolved.payload.get("decision") == ApprovalDecision.approved
    assert resolved.payload.get("decided_by") == "tester"


@pytest.mark.asyncio
async def test_start_run_handles_existing_run_gracefully(repos, registry, policy: GlobalPolicy) -> None:
    """Test start_run proceeds if run already exists (simulating async-first API creation)."""
    reg, _cap = registry
    run = AgentRun(role="dev", objective="x")

    # 1. Pre-seed the run in the repo so get() returns it
    # We access the underlying _RunsRepo method directly to bypass our mock override below initially,
    # or just rely on the fact that our mock override is only for this test.
    # Actually, the repo is a simple class, we can just set state.
    repos["runs"].by_id[run.id] = run

    # 2. Mock create() to raise an exception (simulating DB unique constraint violation)
    async def create_raising(r: AgentRun) -> None:
        raise RuntimeError("Unique constraint violation")

    repos["runs"].create = create_raising

    deps = EngineDeps(
        runs=repos["runs"],
        events=repos["events"],
        approvals=repos["approvals"],
        checkpoints=repos["checkpoints"],
        tool_invocations=repos["tool_invocations"],
        usage=repos["usage"],
        capabilities=reg,
        checkpointer=MemorySaver(),
    )
    engine = AgentEngine(policy=policy, deps=deps)
    engine._graph = _GraphSpy()  # Mock graph to avoid execution

    # 3. Call start_run - should catch exception, check get(), and proceed without error
    await engine.start_run(run=run, plan=[{"capability": CapabilityName.summarize.value, "args": {"text": "hi"}}])

    # Verify run_started event is still emitted (logic flow continues)
    assert any(e.type == AgentEventType.run_started for e in repos["events"].events)


@pytest.mark.asyncio
async def test_start_run_raises_if_create_fails_and_run_missing(repos, registry, policy: GlobalPolicy) -> None:
    """Test start_run re-raises exception if create fails and run is not found."""
    reg, _cap = registry
    run = AgentRun(role="dev", objective="x")

    # 1. Ensure run is NOT in repo
    # repos["runs"].by_id is empty by default

    # 2. Mock create() to raise an exception
    async def create_raising(r: AgentRun) -> None:
        raise RuntimeError("Database connection error")

    repos["runs"].create = create_raising

    deps = EngineDeps(
        runs=repos["runs"],
        events=repos["events"],
        approvals=repos["approvals"],
        checkpoints=repos["checkpoints"],
        tool_invocations=repos["tool_invocations"],
        usage=repos["usage"],
        capabilities=reg,
        checkpointer=MemorySaver(),
    )
    engine = AgentEngine(policy=policy, deps=deps)

    # 3. Call start_run - should re-raise the exception
    with pytest.raises(RuntimeError, match="Database connection error"):
        await engine.start_run(run=run, plan=[])
