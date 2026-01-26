"""Provider integrations used by the agent system.

The ``info_provider`` package contains integration layers that supply external
information and tooling metadata to higher-level components.

Subpackages
-----------

- ``info_provider.prompt``:

  Prompt loading and prompt-provider composition (builtin prompts, stacked
  providers, hot reload wrapper, and configuration-based loader).

- ``info_provider.mcp``:

  MCP tool metadata discovery providers, schemas, and strategy implementations
  (direct and gateway-backed). This layer focuses on discovery and policy-aware
  filtering; execution and streaming are handled by lower-level transports.

- ``info_provider.role``:

  AI agent role provider implementation and utilities for managing role
  definitions, cognitive profiles, permissions, and runtime specifications.

This package intentionally keeps provider concerns isolated from the core agent
runtime so deployments can swap implementations without changing engine logic.
"""

from .mcp.schemas.config import MCPConfig
from .role import (
    AgentRole,
    CapabilityName,
    CognitiveProfile,
    DatabaseRoleProvider,
    DEFAULT_ROLE_PROVIDER,
    ROLE_CAPABILITIES,
    ROLE_SPECS,
    RoleDefinition,
    RolePermissions,
    RoleProvider,
    RoleSpec,
    StaticAgentRoleProvider,
    coerce_role,
    get_database_role_provider,
    get_hardcoded_role_provider,
    get_role_spec,
    load_role_provider,
    load_role_provider_with_session,
)

__all__ = [
    "MCPConfig",
    # Role provider exports
    "AgentRole",
    "CapabilityName",
    "RoleProvider",
    "HardcodedRoleProvider",
    "DatabaseRoleProvider",
    "StaticAgentRoleProvider",
    "DEFAULT_ROLE_PROVIDER",
    "RoleDefinition",
    "CognitiveProfile",
    "RolePermissions",
    "RoleSpec",
    "ROLE_SPECS",
    "ROLE_CAPABILITIES",
    "coerce_role",
    "get_role_spec",
    "get_hardcoded_role_provider",
    "get_database_role_provider",
    "load_role_provider",
    "load_role_provider_with_session",
]
