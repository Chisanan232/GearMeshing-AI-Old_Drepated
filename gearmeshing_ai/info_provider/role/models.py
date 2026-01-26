"""Data models and enums for role provider.

This module contains all the data models, enums, and schemas used by the role
provider system. These models define the structure of agent roles, their
cognitive profiles, permissions, and runtime specifications.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Set

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    """Enumeration of supported agent roles."""

    planner = "planner"
    market = "market"
    dev = "dev"
    dev_lead = "dev_lead"
    qa = "qa"
    sre = "sre"


class CapabilityName(str, Enum):
    """Enumeration of supported capabilities."""

    summarize = "summarize"
    docs_read = "docs_read"
    web_search = "web_search"
    web_fetch = "web_fetch"
    codegen = "codegen"
    mcp_call = "mcp_call"
    shell_exec = "shell_exec"
    code_execution = "code_execution"


class BaseSchema(BaseModel):
    """Base schema for role provider models."""


class CognitiveProfile(BaseSchema):
    """
    Configuration for an agent's cognitive behavior.

    Attributes:
        system_prompt_key: Identifier for the prompt template (e.g. 'dev/system').
        done_when: Optional description of when the agent should consider its task complete.
    """

    system_prompt_key: str
    done_when: str | None = None


class RolePermissions(BaseSchema):
    """
    Permission set for an agent role.

    Attributes:
        allowed_capabilities: Set of high-level capabilities the role can invoke.
        allowed_tools: Set of specific tool names the role can invoke (for finer granularity).
    """

    allowed_capabilities: Set[CapabilityName] = Field(default_factory=set)
    allowed_tools: Set[str] = Field(default_factory=set)


class RoleDefinition(BaseSchema):
    """
    Complete definition of an agent role.

    Combines the role identity, cognitive settings, and permissions.
    """

    role: AgentRole
    cognitive: CognitiveProfile
    permissions: RolePermissions


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


# ============================================================================
# DEFAULT ROLE DEFINITIONS
# ============================================================================

# Minimal, builtin role configurations for basic/local usage. These are
# intentionally conservative and generic. In production deployments these
# are usually overridden by database configurations or custom providers.

DEFAULT_ROLE_DEFINITIONS: Dict[AgentRole, RoleDefinition] = {
    AgentRole.planner: RoleDefinition(
        role=AgentRole.planner,
        cognitive=CognitiveProfile(system_prompt_key="planner/system"),
        permissions=RolePermissions(
            allowed_capabilities={CapabilityName.summarize, CapabilityName.docs_read, CapabilityName.web_search}
        ),
    ),
    AgentRole.market: RoleDefinition(
        role=AgentRole.market,
        cognitive=CognitiveProfile(system_prompt_key="market/system"),
        permissions=RolePermissions(allowed_capabilities={CapabilityName.web_search, CapabilityName.summarize}),
    ),
    AgentRole.dev: RoleDefinition(
        role=AgentRole.dev,
        cognitive=CognitiveProfile(system_prompt_key="dev/system"),
        permissions=RolePermissions(
            allowed_capabilities={CapabilityName.codegen, CapabilityName.mcp_call, CapabilityName.summarize}
        ),
    ),
    AgentRole.dev_lead: RoleDefinition(
        role=AgentRole.dev_lead,
        cognitive=CognitiveProfile(system_prompt_key="dev_lead/system"),
        permissions=RolePermissions(
            allowed_capabilities={CapabilityName.codegen, CapabilityName.mcp_call, CapabilityName.summarize}
        ),
    ),
    AgentRole.qa: RoleDefinition(
        role=AgentRole.qa,
        cognitive=CognitiveProfile(system_prompt_key="qa/system"),
        permissions=RolePermissions(allowed_capabilities={CapabilityName.docs_read, CapabilityName.summarize}),
    ),
    AgentRole.sre: RoleDefinition(
        role=AgentRole.sre,
        cognitive=CognitiveProfile(system_prompt_key="sre/system"),
        permissions=RolePermissions(
            allowed_capabilities={CapabilityName.shell_exec, CapabilityName.mcp_call, CapabilityName.summarize}
        ),
    ),
}


# ============================================================================
# ROLE SPECIFICATIONS
# ============================================================================

ROLE_SPECS: Dict[AgentRole, RoleSpec] = {
    r: RoleSpec(
        role=r,
        allowed_capabilities=set(DEFAULT_ROLE_DEFINITIONS[r].permissions.allowed_capabilities),
        system_prompt_key=DEFAULT_ROLE_DEFINITIONS[r].cognitive.system_prompt_key,
        done_when=DEFAULT_ROLE_DEFINITIONS[r].cognitive.done_when,
    )
    for r in DEFAULT_ROLE_DEFINITIONS.keys()
}

# Backward compatible mapping
ROLE_CAPABILITIES: Dict[str, Set[CapabilityName]] = {
    r.value: set(spec.allowed_capabilities) for r, spec in ROLE_SPECS.items()
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def coerce_role(role: AgentRole | str) -> AgentRole:
    """Helper to ensure a role is an AgentRole enum member."""
    if isinstance(role, AgentRole):
        return role
    return AgentRole(str(role))


def get_role_spec(role: AgentRole | str) -> RoleSpec:
    """Retrieve the RoleSpec for a given role identifier."""
    return ROLE_SPECS[coerce_role(role)]
