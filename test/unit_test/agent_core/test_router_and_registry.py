from __future__ import annotations

from dataclasses import dataclass

import pytest

from gearmeshing_ai.agent_core.agent_registry import AgentRegistry
from gearmeshing_ai.agent_core.factory import build_agent_registry
from gearmeshing_ai.agent_core.router import Router
from gearmeshing_ai.agent_core.schemas.domain import AgentRole, AgentRun
from gearmeshing_ai.agent_core.policy.models import PolicyConfig
from gearmeshing_ai.agent_core.schemas.domain import AutonomyProfile
from gearmeshing_ai.agent_core.runtime.models import EngineDeps
from gearmeshing_ai.agent_core.service import AgentServiceDeps
from gearmeshing_ai.agent_core.planning.planner import StructuredPlanner
from gearmeshing_ai.agent_core.role_provider import (
    CognitiveProfile,
    RoleDefinition,
    RolePermissions,
    StaticAgentRoleProvider,
)
from gearmeshing_ai.agent_core.schemas.domain import CapabilityName


@dataclass
class _Svc:
    name: str


def test_registry_register_and_has_and_get() -> None:
    reg = AgentRegistry()
    reg.register("dev", lambda _run: _Svc("dev"))  # type: ignore[arg-type]

    assert reg.has("dev")
    assert reg.get("dev")(AgentRun(role="dev", objective="x")).name == "dev"  # type: ignore[union-attr]


def test_router_routes_to_registered_role() -> None:
    reg = AgentRegistry()
    reg.register("dev", lambda _run: _Svc("dev"))  # type: ignore[arg-type]

    router = Router(registry=reg, default_role="planner")
    svc = router.route(run=AgentRun(role="dev", objective="x"))
    assert svc.name == "dev"  # type: ignore[union-attr]


def test_router_defaults_when_role_is_blank() -> None:
    reg = AgentRegistry()
    reg.register("planner", lambda _run: _Svc("planner"))  # type: ignore[arg-type]

    router = Router(registry=reg, default_role="planner")
    svc = router.route(run=AgentRun(role="", objective="x"))
    assert svc.name == "planner"  # type: ignore[union-attr]


def test_router_intent_routing_selects_dev_when_enabled() -> None:
    reg = AgentRegistry()
    reg.register("planner", lambda _run: _Svc("planner"))  # type: ignore[arg-type]
    reg.register("dev", lambda _run: _Svc("dev"))  # type: ignore[arg-type]

    router = Router(registry=reg, default_role="planner", enable_intent_routing=True)
    svc = router.route(run=AgentRun(role="", objective="Fix bug in payment flow"))
    assert svc.name == "dev"  # type: ignore[union-attr]


def test_router_intent_routing_falls_back_when_inferred_role_missing() -> None:
    reg = AgentRegistry()
    reg.register("planner", lambda _run: _Svc("planner"))  # type: ignore[arg-type]

    router = Router(registry=reg, default_role="planner", enable_intent_routing=True)
    svc = router.route(run=AgentRun(role="", objective="Investigate incident and monitor metrics"))
    assert svc.name == "planner"  # type: ignore[union-attr]


def test_router_raises_for_unknown_role() -> None:
    reg = AgentRegistry()
    router = Router(registry=reg, default_role="planner")

    with pytest.raises(KeyError, match="unknown role"):
        router.route(run=AgentRun(role="missing", objective="x"))


def test_build_agent_registry_builds_service_with_run_autonomy_profile() -> None:
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
        ),
        planner=StructuredPlanner(model=None),
    )

    registry = build_agent_registry(base_policy_config=base_cfg, deps=deps)
    router = Router(registry=registry)

    run = AgentRun(role="planner", objective="x", autonomy_profile=AutonomyProfile.unrestricted)
    svc = router.route(run=run)
    assert svc._policy_config.autonomy_profile == AutonomyProfile.unrestricted
    assert base_cfg.autonomy_profile == AutonomyProfile.strict


def test_build_agent_registry_applies_role_capabilities_and_intersects_with_base() -> None:
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
        ),
        planner=StructuredPlanner(model=None),
    )

    registry = build_agent_registry(base_policy_config=base_cfg, deps=deps, role_provider=provider)
    router = Router(registry=registry)

    run = AgentRun(role="dev", objective="x")
    svc = router.route(run=run)
    assert svc._policy_config.tool_policy.allowed_capabilities == {CapabilityName.summarize}


def test_build_agent_registry_applies_role_tools_and_intersects_with_base() -> None:
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
        ),
        planner=StructuredPlanner(model=None),
    )

    registry = build_agent_registry(base_policy_config=base_cfg, deps=deps, role_provider=provider)
    router = Router(registry=registry)

    run = AgentRun(role="dev", objective="x")
    svc = router.route(run=run)
    assert svc._policy_config.tool_policy.allowed_tools == {"scm.create_pr"}
