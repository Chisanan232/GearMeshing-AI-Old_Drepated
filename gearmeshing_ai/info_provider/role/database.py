"""Database-based role provider implementation.

This module provides a role provider that reads role configurations
from a database, allowing runtime updates via APIs.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlmodel import Session, select

from .base import RoleProvider
from .role_provider import RoleDefinition, AgentRole, CognitiveProfile, RolePermissions, CapabilityName

logger = logging.getLogger(__name__)


class DatabaseRoleProvider(RoleProvider):
    """
    Provides role configuration from database instead of hardcoded values.

    Supports role-based configuration with tenant-specific overrides.
    """

    def __init__(self, session: Session):
        """
        Initialize the database role provider.

        Args:
            session: SQLModel database session.
        """
        self.session = session
        self._version = "database-v1"

    def get(self, role: str, tenant: Optional[str] = None) -> RoleDefinition:
        """
        Get role definition from database.

        Args:
            role: Name of the role (e.g., 'dev', 'planner').
            tenant: Optional tenant identifier for tenant-specific overrides.

        Returns:
            RoleDefinition with cognitive profile and permissions.

        Raises:
            ValueError: If role configuration not found.
        """
        from gearmeshing_ai.server.models.agent_config import AgentConfig

        # Try tenant-specific config first
        if tenant:
            statement = select(AgentConfig).where(
                (AgentConfig.role_name == role)
                & (AgentConfig.tenant_id == tenant)
                & (AgentConfig.is_active == True)
            )
            config = self.session.exec(statement).first()
            if config:
                return self._config_to_role_definition(config)

        # Fall back to default (no tenant) config
        statement = select(AgentConfig).where(
            (AgentConfig.role_name == role) & (AgentConfig.tenant_id == None) & (AgentConfig.is_active == True)
        )
        config = self.session.exec(statement).first()
        if not config:
            raise ValueError(f"Role '{role}' not found in database configuration")

        return self._config_to_role_definition(config)

    def list_roles(self, tenant: Optional[str] = None) -> list[str]:
        """
        List all available roles.

        Args:
            tenant: Optional tenant identifier to filter by.

        Returns:
            List of role names.
        """
        from gearmeshing_ai.server.models.agent_config import AgentConfig

        statement = select(AgentConfig.role_name).where(AgentConfig.is_active == True).distinct()
        if tenant:
            statement = statement.where(AgentConfig.tenant_id == tenant)

        roles = self.session.exec(statement).all()
        return list(roles)

    def version(self) -> str:
        """Return the version identifier of this provider."""
        return self._version

    def refresh(self) -> None:
        """Refresh the provider state.
        
        For database provider, this is a no-op since queries are always live.
        """
        # No-op for database provider as queries are always live
        pass

    def _config_to_role_definition(self, config) -> RoleDefinition:
        """Convert AgentConfig to RoleDefinition."""
        # Convert role string to AgentRole enum
        try:
            agent_role = AgentRole(config.role_name)
        except ValueError:
            # If role is not in enum, create a string-based role
            agent_role = config.role_name

        # Parse capabilities from JSON
        allowed_capabilities = set()
        if config.role_config and config.role_config.get("permissions"):
            permissions = config.role_config["permissions"]
            if isinstance(permissions, dict) and "allowed_capabilities" in permissions:
                caps = permissions["allowed_capabilities"]
                if isinstance(caps, list):
                    allowed_capabilities = {
                        CapabilityName(cap) for cap in caps if cap in CapabilityName.__members__
                    }

        # Parse cognitive profile
        system_prompt_key = "dev/system"  # default
        done_when = None
        if config.role_config and config.role_config.get("cognitive"):
            cognitive = config.role_config["cognitive"]
            if isinstance(cognitive, dict):
                system_prompt_key = cognitive.get("system_prompt_key", system_prompt_key)
                done_when = cognitive.get("done_when")

        return RoleDefinition(
            role=agent_role,
            cognitive=CognitiveProfile(
                system_prompt_key=system_prompt_key,
                done_when=done_when
            ),
            permissions=RolePermissions(
                allowed_capabilities=allowed_capabilities,
                allowed_tools=set()  # TODO: Parse from config if needed
            )
        )


def get_database_role_provider(session: Session) -> DatabaseRoleProvider:
    """
    Get a database role provider instance.

    Args:
        session: SQLModel database session.

    Returns:
        DatabaseRoleProvider instance.
    """
    return DatabaseRoleProvider(session)
