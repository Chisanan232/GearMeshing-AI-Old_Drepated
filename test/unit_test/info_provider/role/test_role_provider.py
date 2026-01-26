"""Tests for role provider implementations."""

from __future__ import annotations

import json
import logging
import pytest
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from sqlmodel import Session

from gearmeshing_ai.info_provider import (
    DEFAULT_ROLE_PROVIDER,
    StaticAgentRoleProvider,
    AgentRole,
    CapabilityName,
)
from gearmeshing_ai.info_provider.role.provider import (
    HardcodedRoleProvider,
    DatabaseRoleProvider,
    StackedRoleProvider,
    HotReloadRoleWrapper,
    coerce_role,
    get_role_spec,
    get_hardcoded_role_provider,
    get_database_role_provider,
)
from gearmeshing_ai.info_provider.role.base import RoleProvider
from gearmeshing_ai.info_provider.role.models import (
    RoleDefinition,
    CognitiveProfile,
    RolePermissions,
    DEFAULT_ROLE_DEFINITIONS,
)


class TestStaticAgentRoleProvider:
    """Test the StaticAgentRoleProvider class."""

    def test_static_provider_initialization(self):
        """Test StaticAgentRoleProvider initialization."""
        definitions = {AgentRole.dev: DEFAULT_ROLE_DEFINITIONS[AgentRole.dev]}
        provider = StaticAgentRoleProvider(definitions=definitions)
        
        assert provider._definitions == definitions

    def test_static_provider_get_with_enum(self):
        """Test getting role definition using enum."""
        provider = DEFAULT_ROLE_PROVIDER
        role_def = provider.get(AgentRole.dev)
        
        assert role_def.role == AgentRole.dev
        assert isinstance(role_def.cognitive, CognitiveProfile)
        assert isinstance(role_def.permissions, RolePermissions)

    def test_static_provider_get_with_string(self):
        """Test getting role definition using string."""
        provider = DEFAULT_ROLE_PROVIDER
        role_def = provider.get("dev")
        
        assert role_def.role == AgentRole.dev
        assert role_def.cognitive.system_prompt_key

    def test_static_provider_list_roles(self):
        """Test listing all available roles."""
        provider = DEFAULT_ROLE_PROVIDER
        roles = provider.list_roles()
        
        assert "dev" in roles
        assert "planner" in roles
        assert len(roles) >= 2  # At least dev and planner

    def test_static_provider_rejects_unknown_role(self):
        """Test that unknown role raises exception."""
        provider = StaticAgentRoleProvider(definitions={})
        with pytest.raises(KeyError):  # Our implementation raises KeyError
            provider.get("missing")

    def test_static_provider_rejects_unknown_role_enum(self):
        """Test that unknown role enum raises exception."""
        provider = StaticAgentRoleProvider(definitions={})
        with pytest.raises(KeyError):
            provider.get(AgentRole.dev)  # Not in definitions


class TestHardcodedRoleProvider:
    """Test the HardcodedRoleProvider class."""

    def test_hardcoded_provider_default_initialization(self):
        """Test HardcodedRoleProvider with default definitions."""
        provider = HardcodedRoleProvider()
        
        assert provider._provider == DEFAULT_ROLE_PROVIDER

    def test_hardcoded_provider_custom_definitions(self):
        """Test HardcodedRoleProvider with custom definitions."""
        custom_defs = {AgentRole.dev: DEFAULT_ROLE_DEFINITIONS[AgentRole.dev]}
        provider = HardcodedRoleProvider(definitions=custom_defs)
        
        assert isinstance(provider._provider, StaticAgentRoleProvider)
        assert provider._provider._definitions == custom_defs

    def test_hardcoded_provider_get_role(self):
        """Test getting role from hardcoded provider."""
        provider = HardcodedRoleProvider()
        role_def = provider.get("dev")
        
        assert role_def.role == AgentRole.dev
        assert role_def.cognitive.system_prompt_key

    def test_hardcoded_provider_get_with_tenant(self):
        """Test getting role with tenant parameter (ignored)."""
        provider = HardcodedRoleProvider()
        role_def = provider.get("dev", tenant="tenant1")
        
        assert role_def.role == AgentRole.dev

    def test_hardcoded_provider_list_roles(self):
        """Test listing roles from hardcoded provider."""
        provider = HardcodedRoleProvider()
        roles = provider.list_roles()
        
        assert isinstance(roles, list)
        assert "AgentRole.dev" in roles
        assert "AgentRole.planner" in roles

    def test_hardcoded_provider_list_roles_with_tenant(self):
        """Test listing roles with tenant parameter (ignored)."""
        provider = HardcodedRoleProvider()
        roles = provider.list_roles(tenant="tenant1")
        
        assert isinstance(roles, list)
        assert "AgentRole.dev" in roles

    def test_hardcoded_provider_version(self):
        """Test version method."""
        provider = HardcodedRoleProvider()
        version = provider.version()
        
        assert version == "hardcoded-v1"

    def test_hardcoded_provider_refresh(self):
        """Test refresh method (no-op)."""
        provider = HardcodedRoleProvider()
        
        # Should not raise any exception
        provider.refresh()


class TestDatabaseRoleProvider:
    """Test the DatabaseRoleProvider class."""

    def test_database_provider_initialization(self):
        """Test DatabaseRoleProvider initialization."""
        mock_session = Mock(spec=Session)
        provider = DatabaseRoleProvider(mock_session)
        
        assert provider.session == mock_session
        assert provider._version == "database-v1"

    def test_database_provider_get_tenant_specific_success(self):
        """Test getting tenant-specific role successfully."""
        mock_session = Mock(spec=Session)
        provider = DatabaseRoleProvider(mock_session)
        
        # Mock successful tenant-specific query
        mock_config = Mock()
        # Mock the to_role_config method to return a proper RoleConfig
        from gearmeshing_ai.agent_core.schemas.config import RoleConfig
        mock_role_config = Mock(spec=RoleConfig)
        mock_role_config.role_name = "dev"
        mock_role_config.system_prompt_key = "dev.system_prompt"
        mock_role_config.done_when = "task_completed"
        mock_role_config.capabilities = ["docs_read", "codegen"]
        mock_role_config.tools = ["editor"]
        mock_config.to_role_config.return_value = mock_role_config
        
        mock_session.exec.return_value.first.return_value = mock_config
        
        with patch('gearmeshing_ai.info_provider.role.provider.select') as mock_select:
            with patch('gearmeshing_ai.info_provider.role.provider.AgentConfig'):
                role_def = provider.get("dev", tenant="tenant1")
                
                assert role_def.role == AgentRole.dev
                assert role_def.cognitive.system_prompt_key == "dev.system_prompt"
                assert CapabilityName.docs_read in role_def.permissions.allowed_capabilities

    def test_database_provider_get_global_fallback(self):
        """Test getting global role when tenant-specific not found."""
        mock_session = Mock(spec=Session)
        provider = DatabaseRoleProvider(mock_session)
        
        # Mock tenant query returns None, global query returns config
        mock_config = Mock()
        # Mock the to_role_config method to return a proper RoleConfig
        from gearmeshing_ai.agent_core.schemas.config import RoleConfig
        mock_role_config = Mock(spec=RoleConfig)
        mock_role_config.role_name = "planner"
        mock_role_config.system_prompt_key = "planner.prompt"
        mock_role_config.done_when = None
        mock_role_config.capabilities = ["plan"]
        mock_role_config.tools = []
        mock_config.to_role_config.return_value = mock_role_config
        
        # First call (tenant-specific) returns None, second call (global) returns config
        mock_session.exec.return_value.first.side_effect = [None, mock_config]
        
        with patch('gearmeshing_ai.info_provider.role.provider.select') as mock_select:
            with patch('gearmeshing_ai.info_provider.role.provider.AgentConfig'):
                role_def = provider.get("planner", tenant="tenant1")
                
                assert role_def.role == AgentRole.planner
                assert mock_session.exec.call_count == 2

    def test_database_provider_get_role_not_found(self):
        """Test getting role that doesn't exist."""
        mock_session = Mock(spec=Session)
        provider = DatabaseRoleProvider(mock_session)
        
        # Mock both queries return None
        mock_session.exec.return_value.first.return_value = None
        
        with patch('gearmeshing_ai.info_provider.role.provider.select'):
            with patch('gearmeshing_ai.info_provider.role.provider.AgentConfig'):
                with pytest.raises(ValueError, match="Role configuration not found: missing"):
                    provider.get("missing")

    def test_database_provider_get_database_error(self):
        """Test handling database errors."""
        mock_session = Mock(spec=Session)
        provider = DatabaseRoleProvider(mock_session)
        
        mock_session.exec.side_effect = Exception("Database error")
        
        with patch('gearmeshing_ai.info_provider.role.provider.select'):
            with patch('gearmeshing_ai.info_provider.role.provider.AgentConfig'):
                with patch('gearmeshing_ai.info_provider.role.provider.logger') as mock_logger:
                    with pytest.raises(ValueError, match="Role configuration not found: dev"):
                        provider.get("dev")
                    mock_logger.error.assert_called()

    def test_database_provider_list_roles_with_tenant(self):
        """Test listing roles with tenant filter."""
        mock_session = Mock(spec=Session)
        provider = DatabaseRoleProvider(mock_session)
        
        mock_session.exec.return_value.all.return_value = ["dev", "planner"]
        
        with patch('gearmeshing_ai.info_provider.role.provider.select'):
            with patch('gearmeshing_ai.info_provider.role.provider.AgentConfig'):
                roles = provider.list_roles(tenant="tenant1")
                
                assert roles == ["dev", "planner"]

    def test_database_provider_list_roles_global(self):
        """Test listing all roles globally."""
        mock_session = Mock(spec=Session)
        provider = DatabaseRoleProvider(mock_session)
        
        mock_session.exec.return_value.all.return_value = ["dev", "planner", "reviewer"]
        
        with patch('gearmeshing_ai.info_provider.role.provider.select'):
            with patch('gearmeshing_ai.info_provider.role.provider.AgentConfig'):
                roles = provider.list_roles()
                
                assert roles == ["dev", "planner", "reviewer"]

    def test_database_provider_list_roles_error(self):
        """Test listing roles with database error."""
        mock_session = Mock(spec=Session)
        provider = DatabaseRoleProvider(mock_session)
        
        mock_session.exec.side_effect = Exception("Database error")
        
        with patch('gearmeshing_ai.info_provider.role.provider.select'):
            with patch('gearmeshing_ai.info_provider.role.provider.AgentConfig'):
                with patch('gearmeshing_ai.info_provider.role.provider.logger') as mock_logger:
                    roles = provider.list_roles()
                    
                    assert roles == []
                    mock_logger.error.assert_called()

    def test_database_provider_version(self):
        """Test version method."""
        mock_session = Mock(spec=Session)
        provider = DatabaseRoleProvider(mock_session)
        
        version = provider.version()
        assert version == "database-v1"

    def test_database_provider_refresh(self):
        """Test refresh method updates version."""
        mock_session = Mock(spec=Session)
        provider = DatabaseRoleProvider(mock_session)
        
        old_version = provider.version()
        time.sleep(0.01)  # Small delay to ensure different timestamp
        provider.refresh()
        new_version = provider.version()
        
        assert new_version != old_version
        assert new_version.startswith("database-v")

    def test_database_provider_parse_config_invalid_json(self):
        """Test parsing invalid JSON configuration."""
        mock_session = Mock(spec=Session)
        provider = DatabaseRoleProvider(mock_session)
        
        with pytest.raises(ValueError, match="Invalid role configuration format"):
            provider._parse_config("invalid json")

    def test_database_provider_parse_config_missing_fields(self):
        """Test parsing configuration with missing required fields."""
        mock_session = Mock(spec=Session)
        provider = DatabaseRoleProvider(mock_session)
        
        incomplete_config = json.dumps({"role": "dev"})  # Missing cognitive and permissions
        
        with pytest.raises(ValueError, match="Invalid role configuration format"):
            provider._parse_config(incomplete_config)

    def test_database_provider_parse_config_unknown_capability(self):
        """Test parsing configuration with unknown capability."""
        mock_session = Mock(spec=Session)
        provider = DatabaseRoleProvider(mock_session)
        
        config_with_unknown_cap = json.dumps({
            "role": "dev",
            "cognitive": {"system_prompt_key": "dev.prompt"},
            "permissions": {
                "allowed_capabilities": ["docs_read", "unknown_capability"],
                "allowed_tools": []
            }
        })
        
        with patch('gearmeshing_ai.info_provider.role.provider.logger') as mock_logger:
            role_def = provider._parse_config(config_with_unknown_cap)
            
            # Should parse successfully but warn about unknown capability
            assert CapabilityName.docs_read in role_def.permissions.allowed_capabilities
            assert "unknown_capability" not in [cap.value for cap in role_def.permissions.allowed_capabilities]
            mock_logger.warning.assert_called()


class TestStackedRoleProvider:
    """Test the StackedRoleProvider class."""

    def test_stacked_provider_initialization(self):
        """Test StackedRoleProvider initialization."""
        primary = Mock(spec=RoleProvider)
        fallback = Mock(spec=RoleProvider)
        
        provider = StackedRoleProvider(primary, fallback)
        
        assert provider._primary == primary
        assert provider._fallback == fallback

    def test_stacked_provider_get_primary_success(self):
        """Test getting role from primary provider successfully."""
        primary = Mock(spec=RoleProvider)
        fallback = Mock(spec=RoleProvider)
        expected_role = Mock(spec=RoleDefinition)
        primary.get.return_value = expected_role
        
        provider = StackedRoleProvider(primary, fallback)
        result = provider.get("dev", "tenant1")
        
        assert result == expected_role
        primary.get.assert_called_once_with("dev", "tenant1")
        fallback.get.assert_not_called()

    def test_stacked_provider_get_fallback_success(self):
        """Test getting role from fallback when primary fails."""
        primary = Mock(spec=RoleProvider)
        fallback = Mock(spec=RoleProvider)
        expected_role = Mock(spec=RoleDefinition)
        
        primary.get.side_effect = KeyError("Not found")
        fallback.get.return_value = expected_role
        
        provider = StackedRoleProvider(primary, fallback)
        result = provider.get("dev", "tenant1")
        
        assert result == expected_role
        primary.get.assert_called_once_with("dev", "tenant1")
        fallback.get.assert_called_once_with("dev", "tenant1")

    def test_stacked_provider_get_fallback_value_error(self):
        """Test getting role from fallback when primary raises ValueError."""
        primary = Mock(spec=RoleProvider)
        fallback = Mock(spec=RoleProvider)
        expected_role = Mock(spec=RoleDefinition)
        
        primary.get.side_effect = ValueError("Invalid role")
        fallback.get.return_value = expected_role
        
        provider = StackedRoleProvider(primary, fallback)
        result = provider.get("dev", "tenant1")
        
        assert result == expected_role

    def test_stacked_provider_list_roles(self):
        """Test listing roles from both providers."""
        primary = Mock(spec=RoleProvider)
        fallback = Mock(spec=RoleProvider)
        
        primary.list_roles.return_value = ["dev", "planner"]
        fallback.list_roles.return_value = ["planner", "reviewer"]
        
        provider = StackedRoleProvider(primary, fallback)
        roles = provider.list_roles("tenant1")
        
        assert set(roles) == {"dev", "planner", "reviewer"}
        primary.list_roles.assert_called_once_with("tenant1")
        fallback.list_roles.assert_called_once_with("tenant1")

    def test_stacked_provider_list_roles_empty_lists(self):
        """Test listing roles when both providers return empty lists."""
        primary = Mock(spec=RoleProvider)
        fallback = Mock(spec=RoleProvider)
        
        primary.list_roles.return_value = []
        fallback.list_roles.return_value = []
        
        provider = StackedRoleProvider(primary, fallback)
        roles = provider.list_roles()
        
        assert roles == []

    def test_stacked_provider_version(self):
        """Test version method combines both versions."""
        primary = Mock(spec=RoleProvider)
        fallback = Mock(spec=RoleProvider)
        primary.version.return_value = "v1"
        fallback.version.return_value = "v2"
        
        provider = StackedRoleProvider(primary, fallback)
        version = provider.version()
        
        assert version == "stacked(v1+v2)"

    def test_stacked_provider_refresh(self):
        """Test refresh method calls both providers."""
        primary = Mock(spec=RoleProvider)
        fallback = Mock(spec=RoleProvider)
        
        provider = StackedRoleProvider(primary, fallback)
        provider.refresh()
        
        primary.refresh.assert_called_once()
        fallback.refresh.assert_called_once()


class TestHotReloadRoleWrapper:
    """Test the HotReloadRoleWrapper class."""

    def test_hot_reload_wrapper_initialization(self):
        """Test HotReloadRoleWrapper initialization."""
        inner_provider = Mock(spec=RoleProvider)
        
        provider = HotReloadRoleWrapper(inner_provider, interval_seconds=30)
        
        assert provider._provider == inner_provider
        assert provider._interval == 30
        assert provider._last_refresh == 0
        assert provider._lock is not None
        assert provider._stop_event is not None
        assert provider._thread is not None
        assert provider._thread.daemon

    def test_hot_reload_wrapper_get_role(self):
        """Test getting role delegates to inner provider."""
        inner_provider = Mock(spec=RoleProvider)
        expected_role = Mock(spec=RoleDefinition)
        inner_provider.get.return_value = expected_role
        
        provider = HotReloadRoleWrapper(inner_provider)
        result = provider.get("dev", "tenant1")
        
        assert result == expected_role
        inner_provider.get.assert_called_once_with("dev", "tenant1")

    def test_hot_reload_wrapper_list_roles(self):
        """Test listing roles delegates to inner provider."""
        inner_provider = Mock(spec=RoleProvider)
        inner_provider.list_roles.return_value = ["dev", "planner"]
        
        provider = HotReloadRoleWrapper(inner_provider)
        result = provider.list_roles("tenant1")
        
        assert result == ["dev", "planner"]
        inner_provider.list_roles.assert_called_once_with("tenant1")

    def test_hot_reload_wrapper_version(self):
        """Test version method includes hot reload indicator."""
        inner_provider = Mock(spec=RoleProvider)
        inner_provider.version.return_value = "base-v1"
        
        provider = HotReloadRoleWrapper(inner_provider)
        version = provider.version()
        
        assert version == "hotreload-base-v1"

    def test_hot_reload_wrapper_refresh(self):
        """Test refresh method updates timestamp."""
        inner_provider = Mock(spec=RoleProvider)
        
        provider = HotReloadRoleWrapper(inner_provider)
        old_timestamp = provider._last_refresh
        
        time.sleep(0.01)  # Small delay
        provider.refresh()
        
        assert provider._last_refresh > old_timestamp
        inner_provider.refresh.assert_called_once()

    def test_hot_reload_wrapper_stop(self):
        """Test stopping the background thread."""
        inner_provider = Mock(spec=RoleProvider)
        
        provider = HotReloadRoleWrapper(inner_provider, interval_seconds=1)
        
        # Should stop without hanging
        provider.stop()
        assert provider._stop_event.is_set()


class TestUtilityFunctions:
    """Test utility functions."""

    def test_coerce_role_with_enum(self):
        """Test coerce_role with AgentRole enum."""
        result = coerce_role(AgentRole.dev)
        assert result == AgentRole.dev

    def test_coerce_role_with_string(self):
        """Test coerce_role with string."""
        result = coerce_role("dev")
        assert result == AgentRole.dev

    def test_get_role_spec(self):
        """Test get_role_spec function."""
        spec = get_role_spec(AgentRole.dev)
        
        assert spec is not None
        assert hasattr(spec, 'system_prompt_key')
        assert hasattr(spec, 'allowed_capabilities')
        assert spec.role == AgentRole.dev

    def test_get_hardcoded_role_provider(self):
        """Test get_hardcoded_role_provider function."""
        provider = get_hardcoded_role_provider()
        
        assert isinstance(provider, HardcodedRoleProvider)

    def test_get_database_role_provider(self):
        """Test get_database_role_provider function."""
        mock_session = Mock(spec=Session)
        
        provider = get_database_role_provider(mock_session)
        
        assert isinstance(provider, DatabaseRoleProvider)
        assert provider.session == mock_session


# Keep the original tests for backward compatibility
@pytest.mark.parametrize("role", list(AgentRole))
def test_default_role_provider_supports_all_roles(role: AgentRole) -> None:
    d = DEFAULT_ROLE_PROVIDER.get(role)
    assert d.role == role
    assert d.cognitive.system_prompt_key


def test_static_role_provider_rejects_unknown_role() -> None:
    p = StaticAgentRoleProvider(definitions={})
    with pytest.raises(Exception):
        p.get("missing")
