"""
Provider abstractions for agent role definitions.

This module defines the data models and provider protocol for retrieving
agent role configurations. Roles determine an agent's persona (system prompt),
permissions (allowed capabilities/tools), and termination criteria.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Iterable, Protocol, Set

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
    pass


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


class StaticAgentRoleProvider:
    """
    Role provider backed by an in-memory dictionary.

    Useful for default configurations and testing.
    """

    def __init__(self, *, definitions: Dict[AgentRole, RoleDefinition]) -> None:
        self._definitions = dict(definitions)

    def get(self, role: AgentRole | str) -> RoleDefinition:
        if isinstance(role, AgentRole):
            key = role
        else:
            key = AgentRole(str(role))
        return self._definitions[key]

    def list_roles(self) -> Iterable[AgentRole]:
        return self._definitions.keys()


DEFAULT_ROLE_PROVIDER: AgentRoleProvider = StaticAgentRoleProvider(
    definitions={
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
)
