"""Role provider integration.

This sub-package contains the role provider implementation and utilities
for managing AI agent role definitions, including their cognitive profiles,
permissions, and runtime specifications.

The role provider is responsible for:
- Defining agent roles with their cognitive profiles and permissions
- Providing role specifications for runtime execution
- Managing role-based access control for capabilities and tools
"""

from .role_provider import (
    AgentRole,
    AgentRoleProvider,
    CapabilityName,
    CognitiveProfile,
    DEFAULT_ROLE_PROVIDER,
    RoleDefinition,
    RolePermissions,
    StaticAgentRoleProvider,
)
from .roles import (
    ROLE_CAPABILITIES,
    ROLE_SPECS,
    RoleSpec,
    coerce_role,
    get_role_spec,
)

__all__ = [
    # Core types
    "AgentRole",
    "CapabilityName",
    # Role provider types
    "AgentRoleProvider",
    "StaticAgentRoleProvider",
    "DEFAULT_ROLE_PROVIDER",
    # Role definition types
    "RoleDefinition",
    "CognitiveProfile",
    "RolePermissions",
    # Role specification types
    "RoleSpec",
    "ROLE_SPECS",
    "ROLE_CAPABILITIES",
    # Utility functions
    "coerce_role",
    "get_role_spec",
]
