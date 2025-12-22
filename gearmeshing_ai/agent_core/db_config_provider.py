"""
Database-based configuration provider for agent settings.

This module replaces the YAML-based configuration system with database-driven
configuration, allowing runtime updates via APIs.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlmodel import Session, select

from gearmeshing_ai.agent_core.schemas.config import ModelConfig, RoleConfig

logger = logging.getLogger(__name__)


class DatabaseConfigProvider:
    """
    Provides agent configuration from database instead of YAML files.

    Supports role-based configuration with tenant-specific overrides.
    """

    def __init__(self, session: Session):
        """
        Initialize the database config provider.

        Args:
            session: SQLModel database session.
        """
        self.session = session

    def get_model_config(
        self,
        role_name: str,
        tenant_id: Optional[str] = None,
    ) -> ModelConfig:
        """
        Get model configuration for a role.

        Args:
            role_name: Name of the role (e.g., 'dev', 'planner').
            tenant_id: Optional tenant identifier for tenant-specific overrides.

        Returns:
            ModelConfig domain model with provider and parameters.

        Raises:
            ValueError: If role configuration not found.
        """
        from gearmeshing_ai.server.models.agent_config import AgentConfig

        # Try tenant-specific config first
        if tenant_id:
            statement = select(AgentConfig).where(
                (AgentConfig.role_name == role_name)
                & (AgentConfig.tenant_id == tenant_id)
                & (AgentConfig.is_active == True)
            )
            config = self.session.exec(statement).first()
            if config:
                return config.to_model_config()

        # Fall back to default (no tenant) config
        statement = select(AgentConfig).where(
            (AgentConfig.role_name == role_name) & (AgentConfig.tenant_id == None) & (AgentConfig.is_active == True)
        )
        config = self.session.exec(statement).first()
        if not config:
            raise ValueError(f"Role '{role_name}' not found in database configuration")

        return config.to_model_config()

    def get_role_config(self, role_name: str, tenant_id: Optional[str] = None) -> RoleConfig:
        """
        Get complete role configuration.

        Args:
            role_name: Name of the role.
            tenant_id: Optional tenant identifier.

        Returns:
            RoleConfig domain model with complete role settings.

        Raises:
            ValueError: If role not found.
        """
        from gearmeshing_ai.server.models.agent_config import AgentConfig

        # Try tenant-specific config first
        if tenant_id:
            statement = select(AgentConfig).where(
                (AgentConfig.role_name == role_name)
                & (AgentConfig.tenant_id == tenant_id)
                & (AgentConfig.is_active == True)
            )
            config = self.session.exec(statement).first()
            if config:
                return config.to_role_config()

        # Fall back to default config
        statement = select(AgentConfig).where(
            (AgentConfig.role_name == role_name) & (AgentConfig.tenant_id == None) & (AgentConfig.is_active == True)
        )
        config = self.session.exec(statement).first()
        if not config:
            raise ValueError(f"Role '{role_name}' not found in database configuration")

        return config.to_role_config()

    def list_roles(self, tenant_id: Optional[str] = None) -> list[str]:
        """
        List all available roles.

        Args:
            tenant_id: Optional tenant identifier to filter by.

        Returns:
            List of role names.
        """
        from gearmeshing_ai.server.models.agent_config import AgentConfig

        statement = select(AgentConfig.role_name).where(AgentConfig.is_active == True).distinct()
        if tenant_id:
            statement = statement.where(AgentConfig.tenant_id == tenant_id)

        roles = self.session.exec(statement).all()
        return list(roles)


def get_db_config_provider(session: Session) -> DatabaseConfigProvider:
    """
    Get a database configuration provider instance.

    Args:
        session: SQLModel database session.

    Returns:
        DatabaseConfigProvider instance.
    """
    return DatabaseConfigProvider(session)
