from __future__ import annotations

from typing import Dict, Set

from pydantic import Field

from .schemas.base import BaseSchema
from .schemas.domain import AgentRole, CapabilityName
from .role_provider import DEFAULT_ROLE_PROVIDER


class RoleSpec(BaseSchema):
    role: AgentRole
    allowed_capabilities: Set[CapabilityName] = Field(default_factory=set)
    system_prompt_key: str
    done_when: str | None = None


ROLE_SPECS: Dict[AgentRole, RoleSpec] = {
    r: RoleSpec(
        role=r,
        allowed_capabilities=set(DEFAULT_ROLE_PROVIDER.get(r).permissions.allowed_capabilities),
        system_prompt_key=DEFAULT_ROLE_PROVIDER.get(r).cognitive.system_prompt_key,
        done_when=DEFAULT_ROLE_PROVIDER.get(r).cognitive.done_when,
    )
    for r in DEFAULT_ROLE_PROVIDER.list_roles()
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
