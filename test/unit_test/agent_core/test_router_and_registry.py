from __future__ import annotations

from dataclasses import dataclass

import pytest
from langgraph.checkpoint.memory import MemorySaver

from gearmeshing_ai.agent_core.agent_registry import AgentRegistry
from gearmeshing_ai.agent_core.factory import build_agent_registry
from gearmeshing_ai.agent_core.planning.planner import StructuredPlanner
from gearmeshing_ai.agent_core.policy.models import PolicyConfig
from gearmeshing_ai.agent_core.role_provider import (
    CognitiveProfile,
    RoleDefinition,
    RolePermissions,
    StaticAgentRoleProvider,
)
from gearmeshing_ai.agent_core.router import Router
from gearmeshing_ai.agent_core.runtime.models import EngineDeps
from gearmeshing_ai.agent_core.schemas.domain import (
    AgentRole,
    AgentRun,
    AutonomyProfile,
    CapabilityName,
)
from gearmeshing_ai.agent_core.service import AgentServiceDeps


@dataclass
class _Svc:
    name: str


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        ("Fix bug in flow", "dev"),
        ("Please implement feature", "dev"),
        ("Run pytest and add unit test", "qa"),
        ("Deploy to kubernetes", "sre"),
        ("Competitor analysis", "market"),
        ("Something else", "planner"),
        ("", "planner"),
    ],
)
def test_router_infer_role_keyword_buckets(objective: str, expected: str) -> None:
    reg = AgentRegistry()
    router = Router(registry=reg, default_role="planner", enable_intent_routing=True)
    assert router._infer_role(run=AgentRun(role="", objective=objective)) == expected


def test_registry_register_and_has_and_get() -> None:
    reg = AgentRegistry()
    reg.register("dev", lambda _run: _Svc("dev"))  # type: ignore[arg-type, return-value]

    assert reg.has("dev")
    dev_svc = reg.get("dev")(AgentRun(role="dev", objective="x"))
    assert isinstance(dev_svc, _Svc)
    assert dev_svc.name == "dev"


@pytest.mark.asyncio
async def test_router_routes_to_registered_role() -> None:
    reg = AgentRegistry()

    async def async_factory(_run):
        return _Svc("dev")

    reg.register("dev", async_factory)

    router = Router(registry=reg, default_role="planner")
    svc = await router.route(run=AgentRun(role="dev", objective="x"))
    assert isinstance(svc, _Svc)
    assert svc.name == "dev"


@pytest.mark.asyncio
async def test_router_defaults_when_role_is_blank() -> None:
    reg = AgentRegistry()

    async def async_factory(_run):
        return _Svc("planner")

    reg.register("planner", async_factory)

    router = Router(registry=reg, default_role="planner")
    svc = await router.route(run=AgentRun(role="", objective="x"))
    assert isinstance(svc, _Svc)
    assert svc.name == "planner"


@pytest.mark.asyncio
async def test_router_intent_routing_selects_dev_when_enabled() -> None:
    reg = AgentRegistry()

    async def async_planner(_run):
        return _Svc("planner")

    async def async_dev(_run):
        return _Svc("dev")

    reg.register("planner", async_planner)
    reg.register("dev", async_dev)

    router = Router(registry=reg, default_role="planner", enable_intent_routing=True)
    svc = await router.route(run=AgentRun(role="", objective="Fix bug in payment flow"))
    assert isinstance(svc, _Svc)
    assert svc.name == "dev"


@pytest.mark.asyncio
async def test_router_intent_routing_falls_back_when_inferred_role_missing() -> None:
    reg = AgentRegistry()

    async def async_factory(_run):
        return _Svc("planner")

    reg.register("planner", async_factory)

    router = Router(registry=reg, default_role="planner", enable_intent_routing=True)
    svc = await router.route(run=AgentRun(role="", objective="Investigate incident and monitor metrics"))
    assert isinstance(svc, _Svc)
    assert svc.name == "planner"


@pytest.mark.parametrize(
    ("run_role", "objective", "enable_intent", "registered", "expected"),
    [
        ("dev", "x", False, {"dev", "planner"}, "dev"),
        (" dev ", "x", False, {"dev", "planner"}, "dev"),
        ("", "x", False, {"planner"}, "planner"),
        ("", "Fix bug", True, {"dev", "planner"}, "dev"),
        ("", "Competitor analysis", True, {"market", "planner"}, "market"),
        ("", "Fix bug", True, {"planner"}, "planner"),
    ],
)
@pytest.mark.asyncio
async def test_router_route_branching(
    run_role: str,
    objective: str,
    enable_intent: bool,
    registered: set[str],
    expected: str,
) -> None:
    reg = AgentRegistry()
    for r in registered:

        async def async_factory(_run, _r=r):
            return _Svc(_r)

        reg.register(r, async_factory)

    router = Router(registry=reg, default_role="planner", enable_intent_routing=enable_intent)
    svc = await router.route(run=AgentRun(role=run_role, objective=objective))
    assert isinstance(svc, _Svc)
    assert svc.name == expected


@pytest.mark.asyncio
async def test_router_route_raises_when_default_role_not_registered() -> None:
    reg = AgentRegistry()

    async def async_factory(_run):
        return _Svc("dev")

    reg.register("dev", async_factory)
    router = Router(registry=reg, default_role="planner", enable_intent_routing=True)

    with pytest.raises(KeyError, match="unknown role"):
        await router.route(run=AgentRun(role="", objective="Competitor analysis"))


@pytest.mark.asyncio
async def test_router_raises_for_unknown_role() -> None:
    reg = AgentRegistry()
    router = Router(registry=reg, default_role="planner")

    with pytest.raises(KeyError, match="unknown role"):
        await router.route(run=AgentRun(role="missing", objective="x"))


@pytest.mark.asyncio
async def test_build_agent_registry_builds_service_with_run_autonomy_profile() -> None:
    base_cfg = PolicyConfig()
    base_cfg.autonomy_profile = AutonomyProfile.strict

    deps = AgentServiceDeps(
        engine_deps=EngineDeps(
            runs=object(),  # type: ignore[arg-type]
            events=object(),  # type: ignore[arg-type]
            approvals=object(),  # type: ignore[arg-type]
            checkpoints=object(),  # type: ignore[arg-type]
            tool_invocations=object(),  # type: ignore[arg-type]
            capabilities=object(),  # type: ignore[arg-type]
            usage=None,
            checkpointer=MemorySaver(),
        ),
        planner=StructuredPlanner(model=None),
    )

    registry = build_agent_registry(base_policy_config=base_cfg, deps=deps)
    router = Router(registry=registry)

    run = AgentRun(role="planner", objective="x", autonomy_profile=AutonomyProfile.unrestricted)
    svc = await router.route(run=run)
    assert svc._policy_config.autonomy_profile == AutonomyProfile.unrestricted
    assert base_cfg.autonomy_profile == AutonomyProfile.strict


@pytest.mark.asyncio
async def test_build_agent_registry_applies_role_capabilities_and_intersects_with_base() -> None:
    base_cfg = PolicyConfig()
    base_cfg.tool_policy.allowed_capabilities = {CapabilityName.summarize}

    provider = StaticAgentRoleProvider(
        definitions={
            AgentRole.dev: RoleDefinition(
                role=AgentRole.dev,
                cognitive=CognitiveProfile(system_prompt_key="dev/system"),
                permissions=RolePermissions(allowed_capabilities={CapabilityName.summarize, CapabilityName.codegen}),
            )
        }
    )

    deps = AgentServiceDeps(
        engine_deps=EngineDeps(
            runs=object(),  # type: ignore[arg-type]
            events=object(),  # type: ignore[arg-type]
            approvals=object(),  # type: ignore[arg-type]
            checkpoints=object(),  # type: ignore[arg-type]
            tool_invocations=object(),  # type: ignore[arg-type]
            capabilities=object(),  # type: ignore[arg-type]
            usage=None,
            checkpointer=MemorySaver(),
        ),
        planner=StructuredPlanner(model=None),
    )

    registry = build_agent_registry(base_policy_config=base_cfg, deps=deps, role_provider=provider)
    router = Router(registry=registry)

    run = AgentRun(role="dev", objective="x")
    svc = await router.route(run=run)
    assert svc._policy_config.tool_policy.allowed_capabilities == {CapabilityName.summarize}


@pytest.mark.asyncio
async def test_build_agent_registry_applies_role_tools_and_intersects_with_base() -> None:
    base_cfg = PolicyConfig()
    base_cfg.tool_policy.allowed_tools = {"scm.create_pr", "scm.merge_pr"}

    provider = StaticAgentRoleProvider(
        definitions={
            AgentRole.dev: RoleDefinition(
                role=AgentRole.dev,
                cognitive=CognitiveProfile(system_prompt_key="dev/system"),
                permissions=RolePermissions(
                    allowed_capabilities={CapabilityName.summarize},
                    allowed_tools={"scm.create_pr"},
                ),
            )
        }
    )

    deps = AgentServiceDeps(
        engine_deps=EngineDeps(
            runs=object(),  # type: ignore[arg-type]
            events=object(),  # type: ignore[arg-type]
            approvals=object(),  # type: ignore[arg-type]
            checkpoints=object(),  # type: ignore[arg-type]
            tool_invocations=object(),  # type: ignore[arg-type]
            capabilities=object(),  # type: ignore[arg-type]
            usage=None,
            checkpointer=MemorySaver(),
        ),
        planner=StructuredPlanner(model=None),
    )

    registry = build_agent_registry(base_policy_config=base_cfg, deps=deps, role_provider=provider)
    router = Router(registry=registry)

    run = AgentRun(role="dev", objective="x")
    svc = await router.route(run=run)
    assert svc._policy_config.tool_policy.allowed_tools == {"scm.create_pr"}


@pytest.mark.parametrize(
    ("base_allowed_tools", "role_allowed_tools", "expected"),
    [
        (None, {"scm.create_pr"}, {"scm.create_pr"}),
        ({"scm.create_pr", "scm.merge_pr"}, {"scm.create_pr"}, {"scm.create_pr"}),
        ({"scm.create_pr", "scm.merge_pr"}, {"scm.rebase"}, set()),
    ],
)
@pytest.mark.asyncio
async def test_build_agent_registry_allowed_tools_assignment_and_intersection(
    base_allowed_tools: set[str] | None,
    role_allowed_tools: set[str],
    expected: set[str],
) -> None:
    base_cfg = PolicyConfig()
    base_cfg.tool_policy.allowed_tools = base_allowed_tools

    provider = StaticAgentRoleProvider(
        definitions={
            AgentRole.dev: RoleDefinition(
                role=AgentRole.dev,
                cognitive=CognitiveProfile(system_prompt_key="dev/system"),
                permissions=RolePermissions(
                    allowed_capabilities={CapabilityName.summarize},
                    allowed_tools=role_allowed_tools,
                ),
            )
        }
    )

    deps = AgentServiceDeps(
        engine_deps=EngineDeps(
            runs=object(),  # type: ignore[arg-type]
            events=object(),  # type: ignore[arg-type]
            approvals=object(),  # type: ignore[arg-type]
            checkpoints=object(),  # type: ignore[arg-type]
            tool_invocations=object(),  # type: ignore[arg-type]
            capabilities=object(),  # type: ignore[arg-type]
            usage=None,
            checkpointer=MemorySaver(),
        ),
        planner=StructuredPlanner(model=None),
    )

    registry = build_agent_registry(base_policy_config=base_cfg, deps=deps, role_provider=provider)
    svc = await Router(registry=registry).route(run=AgentRun(role="dev", objective="x"))
    assert svc._policy_config.tool_policy.allowed_tools == expected
