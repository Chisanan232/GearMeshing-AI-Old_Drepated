"""
Role specifications and utilities.

This module defines the runtime specification for agent roles, including
their allowed capabilities and system prompts. It acts as a bridge between
the raw ``RoleDefinition`` provided by ``AgentRoleProvider`` and the runtime's
need for concrete configuration.
"""

from __future__ import annotations

from typing import Dict, Set

from pydantic import Field

from .role_provider import DEFAULT_ROLE_PROVIDER
from .schemas.base import BaseSchema
from .schemas.domain import AgentRole, CapabilityName


class RoleSpec(BaseSchema):
    """
    Runtime specification for an agent role.

    Derived from ``RoleDefinition``, this model flattens the configuration
    into a shape readily consumable by the runtime engine and policy layer.

    Attributes:
        role: The specific agent role (e.g. planner, dev).
        allowed_capabilities: Set of capabilities this role is permitted to use.
        system_prompt_key: Key used to resolve the system prompt template.
        done_when: Optional string describing the termination condition for the role.
    """

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
    """Helper to ensure a role is an AgentRole enum member."""
    if isinstance(role, AgentRole):
        return role
    return AgentRole(str(role))


def get_role_spec(role: AgentRole | str) -> RoleSpec:
    """Retrieve the RoleSpec for a given role identifier."""
    return ROLE_SPECS[coerce_role(role)]


# Backward compatible mapping
ROLE_CAPABILITIES: Dict[str, Set[CapabilityName]] = {
    r.value: set(spec.allowed_capabilities) for r, spec in ROLE_SPECS.items()
}
