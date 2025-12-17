from __future__ import annotations

from typing import Dict, Iterable, Protocol, Set

from pydantic import Field

from .schemas.base import BaseSchema
from .schemas.domain import AgentRole, CapabilityName


class CognitiveProfile(BaseSchema):
    system_prompt_key: str
    done_when: str | None = None


class RolePermissions(BaseSchema):
    allowed_capabilities: Set[CapabilityName] = Field(default_factory=set)
    allowed_tools: Set[str] = Field(default_factory=set)


class RoleDefinition(BaseSchema):
    role: AgentRole
    cognitive: CognitiveProfile
    permissions: RolePermissions


class AgentRoleProvider(Protocol):
    def get(self, role: AgentRole | str) -> RoleDefinition: ...

    def list_roles(self) -> Iterable[AgentRole]: ...


class StaticAgentRoleProvider:
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
            permissions=RolePermissions(allowed_capabilities={CapabilityName.codegen, CapabilityName.mcp_call, CapabilityName.summarize}),
        ),
        AgentRole.dev_lead: RoleDefinition(
            role=AgentRole.dev_lead,
            cognitive=CognitiveProfile(system_prompt_key="dev_lead/system"),
            permissions=RolePermissions(allowed_capabilities={CapabilityName.codegen, CapabilityName.mcp_call, CapabilityName.summarize}),
        ),
        AgentRole.qa: RoleDefinition(
            role=AgentRole.qa,
            cognitive=CognitiveProfile(system_prompt_key="qa/system"),
            permissions=RolePermissions(allowed_capabilities={CapabilityName.docs_read, CapabilityName.summarize}),
        ),
        AgentRole.sre: RoleDefinition(
            role=AgentRole.sre,
            cognitive=CognitiveProfile(system_prompt_key="sre/system"),
            permissions=RolePermissions(allowed_capabilities={CapabilityName.shell_exec, CapabilityName.mcp_call, CapabilityName.summarize}),
        ),
    }
)
