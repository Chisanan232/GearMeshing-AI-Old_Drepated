from __future__ import annotations

from typing import Dict, Set

from pydantic import Field

from .schemas.base import BaseSchema
from .schemas.domain import AgentRole, CapabilityName


class RoleSpec(BaseSchema):
    role: AgentRole
    allowed_capabilities: Set[CapabilityName] = Field(default_factory=set)
    system_prompt_key: str
    done_when: str | None = None


ROLE_SPECS: Dict[AgentRole, RoleSpec] = {
    AgentRole.planner: RoleSpec(
        role=AgentRole.planner,
        allowed_capabilities={CapabilityName.summarize, CapabilityName.docs_read, CapabilityName.web_search},
        system_prompt_key="planner/system",
    ),
    AgentRole.market: RoleSpec(
        role=AgentRole.market,
        allowed_capabilities={CapabilityName.web_search, CapabilityName.summarize},
        system_prompt_key="market/system",
    ),
    AgentRole.dev: RoleSpec(
        role=AgentRole.dev,
        allowed_capabilities={CapabilityName.codegen, CapabilityName.mcp_call, CapabilityName.summarize},
        system_prompt_key="dev/system",
    ),
    AgentRole.dev_lead: RoleSpec(
        role=AgentRole.dev_lead,
        allowed_capabilities={CapabilityName.codegen, CapabilityName.mcp_call, CapabilityName.summarize},
        system_prompt_key="dev_lead/system",
    ),
    AgentRole.qa: RoleSpec(
        role=AgentRole.qa,
        allowed_capabilities={CapabilityName.docs_read, CapabilityName.summarize},
        system_prompt_key="qa/system",
    ),
    AgentRole.sre: RoleSpec(
        role=AgentRole.sre,
        allowed_capabilities={CapabilityName.shell_exec, CapabilityName.mcp_call, CapabilityName.summarize},
        system_prompt_key="sre/system",
    ),
}


def coerce_role(role: AgentRole | str) -> AgentRole:
    if isinstance(role, AgentRole):
        return role
    return AgentRole(str(role))


def get_role_spec(role: AgentRole | str) -> RoleSpec:
    return ROLE_SPECS[coerce_role(role)]


# Backward compatible mapping
ROLE_CAPABILITIES: Dict[str, Set[CapabilityName]] = {
    r.value: set(spec.allowed_capabilities) for r, spec in ROLE_SPECS.items()
}
