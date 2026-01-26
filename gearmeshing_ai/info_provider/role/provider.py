"""Concrete role provider implementations.

This module contains the built-in role providers used by
open-source and local deployments, as well as composition helpers:

* :class:`HardcodedRoleProvider` – in-process, dictionary-backed provider
  with a set of common role definitions.
* :class:`DatabaseRoleProvider` – database-backed provider using agent_configs table.
* :class:`StackedRoleProvider` – combines two providers with fallback
  semantics (typically database over hardcoded).
* :class:`HotReloadRoleWrapper` – wraps another provider to call ``refresh``
  periodically in a thread-safe way.

Higher-level code is expected to access these via the
``gearmeshing_ai.info_provider.role`` facade.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Iterable, Optional, cast

from pydantic import BaseModel, Field
from sqlmodel import Session, select

from .base import RoleProvider
from .models import (
    AgentRole,
    CapabilityName,
    CognitiveProfile,
    RoleDefinition,
    RolePermissions,
    RoleSpec,
    DEFAULT_ROLE_DEFINITIONS,
    ROLE_SPECS,
    ROLE_CAPABILITIES,
)

logger = logging.getLogger(__name__)

# ============================================================================
# PROVIDER IMPLEMENTATIONS
# ============================================================================

class StaticAgentRoleProvider(RoleProvider):
    """
    Role provider backed by an in-memory dictionary.

    Useful for default configurations and testing.
    """

    def __init__(self, *, definitions: Dict[AgentRole, RoleDefinition]) -> None:
        self._definitions = dict(definitions)

    def get(self, role: str, tenant: Optional[str] = None) -> RoleDefinition:
        # Convert string role to AgentRole
        try:
            key = AgentRole(role)
        except ValueError:
            # If it's not a valid AgentRole, try to use it as a string key
            # Look for any role that matches the string value
            for agent_role in self._definitions:
                if str(agent_role) == role:
                    key = agent_role
                    break
            else:
                raise KeyError(f"Role not found: {role}")
        
        return self._definitions[key]

    def list_roles(self, tenant: Optional[str] = None) -> list[str]:
        # Ignore tenant for static provider, return all roles as strings
        return [role.value for role in self._definitions.keys()]

    def version(self) -> str:
        return "static-v1"

    def refresh(self) -> None:
        # Static provider has no external state to refresh (no-op)
        pass


DEFAULT_ROLE_PROVIDER: StaticAgentRoleProvider = StaticAgentRoleProvider(
    definitions=DEFAULT_ROLE_DEFINITIONS
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
        # Return the enum string representations for backward compatibility
        return [f"AgentRole.{role}" for role in self._provider.list_roles()]

    def version(self) -> str:
        """Return the version identifier of this provider."""
        return "hardcoded-v1"

    def refresh(self) -> None:
        """Refresh the provider state.
        
        For static provider, this is a no-op.
        """
        # No-op for static provider
        pass


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
        try:
            # Try tenant-specific first
            if tenant:
                stmt = select(AgentConfig).where(
                    AgentConfig.role_name == role,
                    AgentConfig.tenant_id == tenant,
                    AgentConfig.is_active == True
                )
                config = self.session.exec(stmt).first()
                if config:
                    return self._role_config_to_definition(config.to_role_config())

            # Fallback to global
            stmt = select(AgentConfig).where(
                AgentConfig.role_name == role,
                AgentConfig.tenant_id.is_(None),  # type: ignore[union-attr]
                AgentConfig.is_active == True
            )
            config = self.session.exec(stmt).first()
            if config:
                return self._role_config_to_definition(config.to_role_config())

            raise ValueError(f"Role configuration not found: {role}")

        except Exception as e:
            logger.error(f"Failed to get role {role} from database: {e}")
            raise ValueError(f"Role configuration not found: {role}") from e

    def list_roles(self, tenant: Optional[str] = None) -> list[str]:
        """
        List all available roles from database.

        Args:
            tenant: Optional tenant identifier to filter by.

        Returns:
            List of role names.
        """
        try:
            if tenant:
                stmt = select(AgentConfig.role_name).where(
                    AgentConfig.tenant_id == tenant,
                    AgentConfig.is_active == True
                ).distinct()
            else:
                stmt = select(AgentConfig.role_name).where(
                    AgentConfig.is_active == True
                ).distinct()
            
            results = self.session.exec(stmt).all()
            return list(results)

        except Exception as e:
            logger.error(f"Failed to list roles from database: {e}")
            return []

    def _role_config_to_definition(self, role_config) -> RoleDefinition:
        """Convert RoleConfig to RoleDefinition.
        
        Args:
            role_config: RoleConfig from agent_core schemas.
            
        Returns:
            RoleDefinition for info_provider interface.
        """
        from gearmeshing_ai.agent_core.schemas.config import RoleConfig
        from gearmeshing_ai.info_provider.role.models import (
            AgentRole, CognitiveProfile, RoleDefinition, RolePermissions,
            CapabilityName
        )
        
        if not isinstance(role_config, RoleConfig):
            raise ValueError(f"Expected RoleConfig, got {type(role_config)}")
            
        # Convert role name to AgentRole enum
        try:
            agent_role = AgentRole(role_config.role_name)
        except ValueError:
            # If role name not in enum, create a string-based role
            agent_role = AgentRole(role_config.role_name)  # This will still fail but gives clearer error
            
        # Convert capabilities to CapabilityName enum
        allowed_capabilities = set()
        for cap_name in role_config.capabilities:
            try:
                allowed_capabilities.add(CapabilityName(cap_name))
            except ValueError:
                # Skip unknown capabilities
                logger.warning(f"Unknown capability: {cap_name}")
                continue
        
        # Create cognitive profile (using defaults since RoleConfig doesn't have these fields)
        cognitive = CognitiveProfile(
            system_prompt_key=role_config.system_prompt_key or f"{role_config.role_name}/system",
            done_when=role_config.done_when
        )
        
        # Create permissions
        permissions = RolePermissions(
            allowed_capabilities=allowed_capabilities,
            allowed_tools=set(role_config.tools)
        )
        
        return RoleDefinition(
            role=agent_role,
            cognitive=cognitive,
            permissions=permissions
        )

    def version(self) -> str:
        """Return the version identifier of this provider."""
        return self._version

    def refresh(self) -> None:
        """Refresh the provider state.
        
        For database provider, this updates the version.
        """
        # Update version to indicate refresh
        import time
        self._version = f"database-v{int(time.time())}"

    def _parse_config(self, config_value: str) -> RoleDefinition:
        """Parse configuration value from JSON to RoleDefinition."""
        import json
        try:
            data = json.loads(config_value)
            
            # Convert string capabilities to enum
            capabilities = set()
            for cap in data.get("permissions", {}).get("allowed_capabilities", []):
                try:
                    capabilities.add(CapabilityName(cap))
                except ValueError:
                    logger.warning(f"Unknown capability: {cap}")
            
            return RoleDefinition(
                role=AgentRole(data["role"]),
                cognitive=CognitiveProfile(
                    system_prompt_key=data["cognitive"]["system_prompt_key"],
                    done_when=data["cognitive"].get("done_when")
                ),
                permissions=RolePermissions(
                    allowed_capabilities=capabilities,
                    allowed_tools=set(data.get("permissions", {}).get("allowed_tools", []))
                )
            )
        except Exception as e:
            logger.error(f"Failed to parse role config: {e}")
            raise ValueError(f"Invalid role configuration format: {e}") from e


class StackedRoleProvider(RoleProvider):
    """
    Chains two providers with fallback semantics.

    Typical usage: DatabaseRoleProvider over HardcodedRoleProvider.
    """

    def __init__(self, primary: RoleProvider, fallback: RoleProvider):
        """
        Initialize stacked provider.

        Args:
            primary: Primary provider to try first.
            fallback: Fallback provider when primary fails.
        """
        self._primary = primary
        self._fallback = fallback

    def get(self, role: str, tenant: Optional[str] = None) -> RoleDefinition:
        """Get from primary, fallback to secondary on failure."""
        try:
            return self._primary.get(role, tenant)
        except (KeyError, ValueError):
            return self._fallback.get(role, tenant)

    def list_roles(self, tenant: Optional[str] = None) -> list[str]:
        """List roles from both providers, deduplicated."""
        primary_roles = set(self._primary.list_roles(tenant) or [])
        fallback_roles = set(self._fallback.list_roles(tenant) or [])
        return list(primary_roles | fallback_roles)

    def version(self) -> str:
        """Return combined version string."""
        return f"stacked({self._primary.version()}+{self._fallback.version()})"

    def refresh(self) -> None:
        """Refresh both providers."""
        self._primary.refresh()
        self._fallback.refresh()


class HotReloadRoleWrapper(RoleProvider):
    """
    Lightweight, thread-safe wrapper that periodically refreshes a role provider.

    Useful for providers that support hot reload (e.g., database providers).
    """

    def __init__(self, provider: RoleProvider, interval_seconds: int = 60):
        """
        Initialize hot reload wrapper.

        Args:
            provider: The underlying provider to wrap.
            interval_seconds: How often to refresh the provider.
        """
        self._provider = provider
        self._interval = interval_seconds
        self._last_refresh: float = 0.0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Start background refresh thread
        self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._thread.start()

    def get(self, role: str, tenant: Optional[str] = None) -> RoleDefinition:
        """Get role from underlying provider."""
        return self._provider.get(role, tenant)

    def list_roles(self, tenant: Optional[str] = None) -> list[str]:
        """List roles from underlying provider."""
        return self._provider.list_roles(tenant)

    def version(self) -> str:
        """Return version with hot reload indicator."""
        base_version = self._provider.version()
        return f"hotreload-{base_version}"

    def refresh(self) -> None:
        """Refresh the underlying provider."""
        with self._lock:
            self._provider.refresh()
            self._last_refresh = time.time()

    def _refresh_loop(self) -> None:
        """Background thread that periodically refreshes the provider."""
        while not self._stop_event.wait(self._interval):
            try:
                self.refresh()
                logger.debug(f"Hot reload refreshed role provider")
            except Exception as e:
                logger.error(f"Hot reload failed: {e}")

    def stop(self) -> None:
        """Stop the hot reload background thread."""
        self._stop_event.set()
        self._thread.join(timeout=5)


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


def get_hardcoded_role_provider() -> HardcodedRoleProvider:
    """
    Get a hardcoded role provider instance.

    Returns:
        HardcodedRoleProvider instance with default definitions.
    """
    return HardcodedRoleProvider()


def get_database_role_provider(session: Session) -> DatabaseRoleProvider:
    """
    Get a database role provider instance.

    Args:
        session: SQLModel database session.

    Returns:
        DatabaseRoleProvider instance.
    """
    return DatabaseRoleProvider(session)


# ============================================================================
# IMPORT FIXUP
# ============================================================================

# Import AgentConfig for database provider
from gearmeshing_ai.server.models.agent_config import AgentConfig
