from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from gearmeshing_ai.agent_core.factory import build_default_registry, build_engine
from gearmeshing_ai.agent_core.policy.models import PolicyConfig
from gearmeshing_ai.agent_core.runtime import EngineDeps
from gearmeshing_ai.agent_core.schemas.domain import AgentRole, AgentRun, CapabilityName
from gearmeshing_ai.agent_core.service import AgentService, AgentServiceDeps


def test_build_default_registry_registers_all_builtin_capabilities() -> None:
    reg = build_default_registry()
    assert reg.has(CapabilityName.summarize)
    assert reg.has(CapabilityName.web_search)
    assert reg.has(CapabilityName.web_fetch)
    assert reg.has(CapabilityName.shell_exec)
    assert reg.has(CapabilityName.code_execution)
    assert reg.has(CapabilityName.codegen)


def test_build_engine_wires_policy_config_and_deps() -> None:
    cfg = PolicyConfig()
    deps = EngineDeps(
        runs=object(),
        events=object(),
        approvals=object(),
        checkpoints=object(),
        tool_invocations=object(),
        capabilities=build_default_registry(),
    )

    engine = build_engine(policy_config=cfg, deps=deps)
    assert engine._deps is deps  # type: ignore[attr-defined]
    assert engine._policy.config is cfg  # type: ignore[attr-defined]


@dataclass
class _FakePlanner:
    plan_result: list[dict[str, Any]]
    last_objective: str | None = None
    last_role: str | None = None

    async def plan(self, *, objective: str, role: str) -> list[dict[str, Any]]:
        self.last_objective = objective
        self.last_role = role
        return list(self.plan_result)


class _FakeEngine:
    def __init__(self) -> None:
        self.started: list[tuple[AgentRun, list[dict[str, Any]]]] = []
        self.resumed: list[tuple[str, str]] = []

    async def start_run(self, *, run: AgentRun, plan: list[dict[str, Any]]) -> str:
        self.started.append((run, plan))
        return run.id

    async def resume_run(self, *, run_id: str, approval_id: str) -> None:
        self.resumed.append((run_id, approval_id))


@pytest.mark.asyncio
async def test_agent_service_run_calls_planner_and_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_engine = _FakeEngine()

    import gearmeshing_ai.agent_core.factory as factory_mod

    def _build_engine(*, policy_config: PolicyConfig, deps: EngineDeps):
        return fake_engine

    monkeypatch.setattr(factory_mod, "build_engine", _build_engine)

    planner = _FakePlanner(plan_result=[{"capability": CapabilityName.summarize, "args": {"text": "x"}}])
    deps = AgentServiceDeps(
        engine_deps=EngineDeps(
            runs=object(),
            events=object(),
            approvals=object(),
            checkpoints=object(),
            tool_invocations=object(),
            capabilities=build_default_registry(),
        ),
        planner=planner,
    )

    svc = AgentService(policy_config=PolicyConfig(), deps=deps)

    run = AgentRun(role=AgentRole.dev, objective="do x")
    out = await svc.run(run=run)

    assert out == run.id
    assert planner.last_objective == "do x"
    assert planner.last_role == AgentRole.dev
    assert len(fake_engine.started) == 1
    started_run, started_plan = fake_engine.started[0]
    assert started_run is run
    assert started_plan == [{"capability": CapabilityName.summarize, "args": {"text": "x"}}]


@pytest.mark.asyncio
async def test_agent_service_resume_calls_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_engine = _FakeEngine()

    import gearmeshing_ai.agent_core.factory as factory_mod

    def _build_engine(*, policy_config: PolicyConfig, deps: EngineDeps):
        return fake_engine

    monkeypatch.setattr(factory_mod, "build_engine", _build_engine)

    deps = AgentServiceDeps(
        engine_deps=EngineDeps(
            runs=object(),
            events=object(),
            approvals=object(),
            checkpoints=object(),
            tool_invocations=object(),
            capabilities=build_default_registry(),
        ),
        planner=_FakePlanner(plan_result=[]),
    )

    svc = AgentService(policy_config=PolicyConfig(), deps=deps)
    await svc.resume(run_id="r1", approval_id="a1")

    assert fake_engine.resumed == [("r1", "a1")]
