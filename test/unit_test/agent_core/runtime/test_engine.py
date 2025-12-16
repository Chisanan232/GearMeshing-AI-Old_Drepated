from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

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


class _CheckpointsRepo:
    def __init__(self) -> None:
        self.latest_by_run_id: Dict[str, _Checkpoint] = {}
        self.saved: List[_Checkpoint] = []

    async def save(self, cp) -> None:
        chk = _Checkpoint(id=cp.id, state=dict(cp.state))
        self.latest_by_run_id[cp.run_id] = chk
        self.saved.append(chk)

    async def latest(self, run_id: str):
        return self.latest_by_run_id.get(run_id)


class _ToolInvocationsRepo:
    def __init__(self) -> None:
        self.invocations: List[Any] = []

    async def append(self, invocation) -> None:
        self.invocations.append(invocation)


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
        capabilities=reg,
    )
    return AgentEngine(policy=policy, deps=deps)


@pytest.mark.asyncio
async def test_node_execute_next_run_missing_raises(engine_runtime: AgentEngine) -> None:
    state: _GraphState = {"run_id": "missing", "plan": [], "idx": 0, "awaiting_approval_id": None}
    with pytest.raises(ValueError, match="run not found"):
        await engine_runtime._node_execute_next(state)


@pytest.mark.asyncio
async def test_node_execute_next_marks_finished_when_idx_out_of_range(engine_runtime: AgentEngine, repos) -> None:
    run = AgentRun(role="dev", objective="x")
    await repos["runs"].create(run)

    state: _GraphState = {"run_id": run.id, "plan": [], "idx": 0, "awaiting_approval_id": None}
    out = await engine_runtime._node_execute_next(state)
    assert out.get("_finished") is True


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
    )
    engine_runtime = AgentEngine(policy=policy, deps=deps)

    run = AgentRun(role="dev", objective="x")
    await repos["runs"].create(run)

    plan = [{"capability": CapabilityName.shell_exec.value, "args": {"cmd": "echo hi"}}]
    state: _GraphState = {"run_id": run.id, "plan": plan, "idx": 0, "awaiting_approval_id": None}
    out = await engine_runtime._node_execute_next(state)

    assert out.get("_finished") is True
    assert out.get("_terminal_status") == AgentRunStatus.failed.value
    assert repos["runs"].status_updates[-1] == (run.id, AgentRunStatus.failed.value)
    assert any(e.type == AgentEventType.run_failed for e in repos["events"].events)


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
    )
    engine_runtime = AgentEngine(policy=policy, deps=deps)

    run = AgentRun(role="dev", objective="x")
    await repos["runs"].create(run)

    plan = [{"capability": CapabilityName.summarize.value, "args": {"text": "xx"}}]
    state: _GraphState = {"run_id": run.id, "plan": plan, "idx": 0, "awaiting_approval_id": None}
    out = await engine_runtime._node_execute_next(state)

    assert out.get("_finished") is True
    assert out.get("_terminal_status") == AgentRunStatus.failed.value
    assert repos["runs"].status_updates[-1] == (run.id, AgentRunStatus.failed.value)
    assert any(e.type == AgentEventType.run_failed for e in repos["events"].events)


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
    )
    engine_runtime = AgentEngine(policy=policy, deps=deps)

    run = AgentRun(role="dev", objective="x")
    await repos["runs"].create(run)

    plan = [{"capability": CapabilityName.summarize.value, "args": {"text": "hello"}}]
    state: _GraphState = {"run_id": run.id, "plan": plan, "idx": 0, "awaiting_approval_id": None}
    out = await engine_runtime._node_execute_next(state)

    assert out.get("awaiting_approval_id")
    assert repos["runs"].status_updates[-1] == (run.id, AgentRunStatus.paused_for_approval.value)
    assert repos["approvals"].created
    assert repos["checkpoints"].saved
    assert any(e.type == AgentEventType.approval_requested for e in repos["events"].events)
    assert any(e.type == AgentEventType.checkpoint_saved for e in repos["events"].events)


@pytest.mark.asyncio
async def test_node_execute_next_executes_capability_and_appends_tool_invocation(
    engine_runtime: AgentEngine,
    repos,
    registry,
) -> None:
    _reg, cap = registry
    run = AgentRun(role="dev", objective="x")
    await repos["runs"].create(run)

    plan = [{"capability": CapabilityName.summarize.value, "args": {"text": "hello"}}]
    state: _GraphState = {"run_id": run.id, "plan": plan, "idx": 0, "awaiting_approval_id": None}
    out = await engine_runtime._node_execute_next(state)

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


def test_route_after_execute_prefers_pause_then_finish_then_continue(engine_runtime: AgentEngine) -> None:
    assert engine_runtime._route_after_execute({"run_id": "r", "plan": [], "idx": 0, "awaiting_approval_id": "a"}) == "pause"
    assert engine_runtime._route_after_execute({"run_id": "r", "plan": [], "idx": 0, "awaiting_approval_id": None, "_finished": True}) == "finish"
    assert engine_runtime._route_after_execute({"run_id": "r", "plan": [], "idx": 0, "awaiting_approval_id": None}) == "continue"


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
    with pytest.raises(ValueError, match="no checkpoint"):
        await engine_runtime.resume_run(run_id="r", approval_id=approval.id)
