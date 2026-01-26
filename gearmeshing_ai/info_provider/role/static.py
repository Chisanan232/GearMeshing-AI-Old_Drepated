"""Static/hardcoded role provider implementation.

This module provides the default, built-in role provider with hardcoded
role definitions. This is the fallback provider used when no other
provider is configured or when external providers fail.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from .base import RoleProvider
from .role_provider import (
    AgentRole,
    CapabilityName,
    CognitiveProfile,
    RoleDefinition,
    RolePermissions,
    StaticAgentRoleProvider,
    DEFAULT_ROLE_PROVIDER,
)


class HardcodedRoleProvider(RoleProvider):
    """
    Role provider backed by hardcoded static definitions.

    This is the default provider that contains the built-in role definitions.
    It's always available as a safe fallback.
    """

    def __init__(self, definitions: Dict[AgentRole, RoleDefinition] | None = None) -> None:
        """
        Initialize the hardcoded role provider.

        Args:
            definitions: Optional custom role definitions. If None, uses defaults.
        """
        if definitions is None:
            # Use the default definitions from the original static provider
            self._provider = DEFAULT_ROLE_PROVIDER
        else:
            self._provider = StaticAgentRoleProvider(definitions=definitions)

    def get(self, role: str, tenant: Optional[str] = None) -> RoleDefinition:
        """
        Get role definition.

        Args:
            role: Name of the role.
            tenant: Optional tenant identifier (ignored for static provider).

        Returns:
            RoleDefinition for the role.

        Raises:
            KeyError: If the role is not found.
        """
        return self._provider.get(role)

    def list_roles(self, tenant: Optional[str] = None) -> list[str]:
        """
        List all available roles.

        Args:
            tenant: Optional tenant identifier (ignored for static provider).

        Returns:
            List of role names.
        """
        return [str(role) for role in self._provider.list_roles()]

    def version(self) -> str:
        """Return the version identifier of this provider."""
        return "hardcoded-v1"

    def refresh(self) -> None:
        """Refresh the provider state.
        
        For static provider, this is a no-op.
        """
        # No-op for static provider
        pass


def get_hardcoded_role_provider() -> HardcodedRoleProvider:
    """
    Get a hardcoded role provider instance.

    Returns:
        HardcodedRoleProvider instance with default definitions.
    """
    return HardcodedRoleProvider()
