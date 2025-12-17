from __future__ import annotations

from gearmeshing_ai.agent_core.policy.models import PolicyConfig
from gearmeshing_ai.agent_core.policy.provider import StaticPolicyProvider
from gearmeshing_ai.agent_core.schemas.domain import AgentRun, AutonomyProfile


def test_static_policy_provider_uses_default_when_no_ids() -> None:
    base = PolicyConfig(version="v-default")
    provider = StaticPolicyProvider(default=base)

    cfg = provider.get(AgentRun(role="dev", objective="x"))
    assert cfg.version == "v-default"


def test_static_policy_provider_prefers_tenant_then_workspace() -> None:
    base = PolicyConfig(version="v-default")
    tenant_cfg = PolicyConfig(version="v-tenant")
    workspace_cfg = PolicyConfig(version="v-workspace")

    provider = StaticPolicyProvider(
        default=base,
        by_tenant={"t1": tenant_cfg},
        by_workspace={"w1": workspace_cfg},
    )

    run_tenant = AgentRun(role="dev", objective="x", tenant_id="t1", workspace_id="w1")
    cfg = provider.get(run_tenant)
    assert cfg.version == "v-tenant"

    run_ws = AgentRun(role="dev", objective="x", tenant_id=None, workspace_id="w1")
    cfg2 = provider.get(run_ws)
    assert cfg2.version == "v-workspace"


def test_agent_service_overrides_autonomy_profile_from_run(monkeypatch) -> None:
    from gearmeshing_ai.agent_core.service import AgentService, AgentServiceDeps
    from gearmeshing_ai.agent_core.runtime import EngineDeps
    from gearmeshing_ai.agent_core.factory import build_default_registry
    from gearmeshing_ai.agent_core.planning import StructuredPlanner

    class _FakePlanner(StructuredPlanner):
        async def plan(self, *, objective: str, role: str):
            return []

    class _FakeRuns:
        async def get(self, _run_id: str):
            return None

    fake_deps = AgentServiceDeps(
        engine_deps=EngineDeps(
            runs=_FakeRuns(),  # type: ignore[arg-type]
            events=object(),  # type: ignore[arg-type]
            approvals=object(),  # type: ignore[arg-type]
            checkpoints=object(),  # type: ignore[arg-type]
            tool_invocations=object(),  # type: ignore[arg-type]
            capabilities=build_default_registry(),
            usage=None,
        ),
        planner=_FakePlanner(model=None),
    )

    base = PolicyConfig(autonomy_profile=AutonomyProfile.strict)
    provider = StaticPolicyProvider(default=base)

    svc = AgentService(policy_config=base, deps=fake_deps, policy_provider=provider)
    cfg = svc._policy_for_run(AgentRun(role="dev", objective="x", autonomy_profile=AutonomyProfile.unrestricted))
    assert cfg.autonomy_profile == AutonomyProfile.unrestricted
