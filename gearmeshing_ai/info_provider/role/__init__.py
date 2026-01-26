"""Role provider integration.

This sub-package contains the role provider implementation and utilities
for managing AI agent role definitions, including their cognitive profiles,
permissions, and runtime specifications.

The role provider is responsible for:
- Defining agent roles with their cognitive profiles and permissions
- Providing role specifications for runtime execution
- Managing role-based access control for capabilities and tools
- Supporting multiple configuration sources (hardcoded, database, external)

Provider Architecture:
- HardcodedRoleProvider: Static, built-in role definitions (default)
- DatabaseRoleProvider: Dynamic role configurations from database
- External providers: Pluggable providers via entry points
- RoleProviderLoader: Environment-driven provider selection
"""

from .base import RoleProvider
from .loader import load_role_provider, load_role_provider_with_session
from .models import (
    AgentRole,
    CapabilityName,
    CognitiveProfile,
    DEFAULT_ROLE_DEFINITIONS,
    ROLE_CAPABILITIES,
    ROLE_SPECS,
    RoleDefinition,
    RolePermissions,
    RoleSpec,
    coerce_role,
    get_role_spec,
)
from .provider import (
    DatabaseRoleProvider,
    DEFAULT_ROLE_PROVIDER,
    HardcodedRoleProvider,
    StaticAgentRoleProvider,
    StackedRoleProvider,
    HotReloadRoleWrapper,
    get_database_role_provider,
    get_hardcoded_role_provider,
)

__all__ = [
    # Core types
    "AgentRole",
    "CapabilityName",
    # Provider interfaces
    "RoleProvider",
    # Provider implementations
    "HardcodedRoleProvider",
    "DatabaseRoleProvider",
    "StaticAgentRoleProvider",
    "StackedRoleProvider",
    "HotReloadRoleWrapper",
    "DEFAULT_ROLE_PROVIDER",
    "DEFAULT_ROLE_DEFINITIONS",
    # Provider factories
    "get_hardcoded_role_provider",
    "get_database_role_provider",
    "load_role_provider",
    "load_role_provider_with_session",
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
