from __future__ import annotations

import pytest

from gearmeshing_ai.agent_core.roles import ROLE_CAPABILITIES, ROLE_SPECS, coerce_role, get_role_spec
from gearmeshing_ai.agent_core.schemas.domain import AgentRole, CapabilityName


@pytest.mark.parametrize(
    ("role", "allowed", "system_prompt_key", "done_when"),
    [
        (
            AgentRole.planner,
            {CapabilityName.summarize, CapabilityName.docs_read, CapabilityName.web_search},
            "planner/system",
            None,
        ),
        (AgentRole.market, {CapabilityName.web_search, CapabilityName.summarize}, "market/system", None),
        (
            AgentRole.dev,
            {CapabilityName.codegen, CapabilityName.mcp_call, CapabilityName.summarize},
            "dev/system",
            None,
        ),
        (
            AgentRole.dev_lead,
            {CapabilityName.codegen, CapabilityName.mcp_call, CapabilityName.summarize},
            "dev_lead/system",
            None,
        ),
        (AgentRole.qa, {CapabilityName.docs_read, CapabilityName.summarize}, "qa/system", None),
        (
            AgentRole.sre,
            {CapabilityName.shell_exec, CapabilityName.mcp_call, CapabilityName.summarize},
            "sre/system",
            None,
        ),
    ],
)
def test_role_specs_are_correct(
    role: AgentRole,
    allowed: set[CapabilityName],
    system_prompt_key: str,
    done_when: str | None,
) -> None:
    spec = ROLE_SPECS[role]
    assert spec.role == role
    assert set(spec.allowed_capabilities) == allowed
    assert spec.system_prompt_key == system_prompt_key
    assert spec.done_when == done_when


@pytest.mark.parametrize("role", list(AgentRole))
def test_get_role_spec_accepts_agentrole_and_string(role: AgentRole) -> None:
    assert get_role_spec(role) == ROLE_SPECS[role]
    assert get_role_spec(role.value) == ROLE_SPECS[role]


@pytest.mark.parametrize("role", list(AgentRole))
def test_coerce_role_accepts_agentrole_and_string(role: AgentRole) -> None:
    assert coerce_role(role) == role
    assert coerce_role(role.value) == role


def test_role_capabilities_mapping_matches_role_specs() -> None:
    expected = {r.value: set(spec.allowed_capabilities) for r, spec in ROLE_SPECS.items()}
    assert ROLE_CAPABILITIES == expected
