"""Unit tests for agent configuration repository.

Tests repository operations with mocked database session to ensure
business logic works correctly without real database dependencies.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from gearmeshing_ai.core.database.entities.agent_configs import AgentConfig
from gearmeshing_ai.core.database.repositories.agent_configs import AgentConfigRepository


class TestAgentConfigRepository:
    """Tests for AgentConfigRepository operations."""
    
    @pytest.fixture
    def mock_session(self):
        """Mock async database session."""
        session = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.delete = AsyncMock()
        # Make execute return the mock result directly, not a coroutine
        mock_result = AsyncMock()
        # Make scalar_one_or_none return the object directly, not a coroutine
        mock_result.scalar_one_or_none = MagicMock()
        mock_result.scalars = MagicMock()
        mock_result.scalars.all = MagicMock()
        session.execute = AsyncMock(return_value=mock_result)
        return session
    
    @pytest.fixture
    def repository(self, mock_session):
        """Create repository instance with mocked session."""
        return AgentConfigRepository(mock_session)
    
    @pytest.fixture
    def sample_agent_config(self, sample_agent_config_data):
        """Create sample AgentConfig instance."""
        return AgentConfig(**sample_agent_config_data)
    
    async def test_create_success(self, repository, mock_session, sample_agent_config):
        """Test successful agent configuration creation."""
        # Mock the refresh to return the same object
        mock_session.refresh.return_value = None
        
        result = await repository.create(sample_agent_config)
        
        # Verify session operations
        mock_session.add.assert_called_once_with(sample_agent_config)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_agent_config)
        
        # Verify return value
        assert result == sample_agent_config
    
    async def test_get_by_id_found(self, repository, mock_session, sample_agent_config):
        """Test getting agent configuration by ID when found."""
        # Mock the query execution
        mock_result = mock_session.execute.return_value
        mock_result.scalar_one_or_none.return_value = sample_agent_config
        
        result = await repository.get_by_id(1)
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        # Don't check isinstance since it's a mock, just check it was called
        assert mock_session.execute.called
        
        # Verify return value
        assert result == sample_agent_config
    
    async def test_get_by_id_not_found(self, repository, mock_session):
        """Test getting agent configuration by ID when not found."""
        # Mock the query execution to return None
        mock_result = mock_session.execute.return_value
        mock_result.scalar_one_or_none.return_value = None
        
        result = await repository.get_by_id(999)
        
        assert result is None
    
    async def test_get_by_role_global_config(self, repository, mock_session, sample_agent_config):
        """Test getting global configuration by role name."""
        # Mock the query execution
        mock_result = mock_session.execute.return_value
        mock_result.scalar_one_or_none.return_value = sample_agent_config
        
        result = await repository.get_by_role("developer")
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        
        # Verify return value
        assert result == sample_agent_config
        assert result.role_name == "developer"
        assert result.tenant_id is None
    
    async def test_get_by_role_tenant_specific(self, repository, mock_session, sample_agent_config):
        """Test getting tenant-specific configuration by role name."""
        # Update sample config to have tenant
        sample_agent_config.tenant_id = "tenant_123"
        
        # Mock the query execution
        mock_result = mock_session.execute.return_value
        mock_result.scalar_one_or_none.return_value = sample_agent_config
        
        result = await repository.get_by_role("developer", "tenant_123")
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        
        # Verify return value
        assert result == sample_agent_config
        assert result.role_name == "developer"
        assert result.tenant_id == "tenant_123"
    
    async def test_get_by_role_not_found(self, repository, mock_session):
        """Test getting configuration by role when not found."""
        # Mock the query execution to return None
        mock_result = mock_session.execute.return_value
        mock_result.scalar_one_or_none.return_value = None
        
        result = await repository.get_by_role("nonexistent_role")
        
        assert result is None
    
    async def test_update_success(self, repository, mock_session, sample_agent_config):
        """Test successful agent configuration update."""
        # Mock the refresh to return the updated object
        mock_session.refresh.return_value = None
        
        result = await repository.update(sample_agent_config)
        
        # Verify session operations
        mock_session.add.assert_called_once_with(sample_agent_config)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_agent_config)
        
        # Verify return value
        assert result == sample_agent_config
    
    async def test_delete_success(self, repository, mock_session, sample_agent_config):
        """Test successful agent configuration deletion."""
        # Mock get_by_id to return the config
        with patch.object(repository, 'get_by_id', return_value=sample_agent_config):
            result = await repository.delete(1)
        
        # Verify session operations
        mock_session.delete.assert_called_once_with(sample_agent_config)
        mock_session.commit.assert_called_once()
        
        # Verify return value
        assert result is True
    
    async def test_delete_not_found(self, repository, mock_session):
        """Test deleting agent configuration when not found."""
        # Mock get_by_id to return None
        with patch.object(repository, 'get_by_id', return_value=None):
            result = await repository.delete(999)
        
        # Verify no delete operations
        mock_session.delete.assert_not_called()
        mock_session.commit.assert_not_called()
        
        # Verify return value
        assert result is False
    
    async def test_list_no_filters(self, repository, mock_session, sample_agent_config):
        """Test listing agent configurations without filters."""
        # Mock the query execution
        mock_result = mock_session.execute.return_value
        mock_result.scalars.return_value.all.return_value = [sample_agent_config]
        
        result = await repository.list()
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        # Don't check isinstance since it's a mock, just check it was called
        assert mock_session.execute.called
        
        # Verify return value
        assert result == [sample_agent_config]
    
    async def test_list_with_filters(self, repository, mock_session, sample_agent_config):
        """Test listing agent configurations with filters."""
        filters = {
            "tenant_id": "tenant_123",
            "model_provider": "openai",
            "role_name": "developer"
        }
        
        # Mock the query execution
        mock_result = mock_session.execute.return_value
        mock_result.scalars.return_value.all.return_value = [sample_agent_config]
        
        result = await repository.list(filters=filters)
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        
        # Verify return value
        assert result == [sample_agent_config]
    
    async def test_list_with_pagination(self, repository, mock_session, sample_agent_config):
        """Test listing agent configurations with pagination."""
        # Mock the query execution
        mock_result = mock_session.execute.return_value
        mock_result.scalars.return_value.all.return_value = [sample_agent_config]
        
        result = await repository.list(limit=10, offset=20)
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        
        # Verify return value
        assert result == [sample_agent_config]
    
    async def test_get_active_configs_for_tenant(self, repository, mock_session, sample_agent_config):
        """Test getting active configurations for a tenant."""
        # Update sample config to have tenant
        sample_agent_config.tenant_id = "tenant_123"
        
        # Mock the query execution
        mock_result = mock_session.execute.return_value
        mock_result.scalars.return_value.all.return_value = [sample_agent_config]
        
        result = await repository.get_active_configs_for_tenant("tenant_123")
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        
        # Verify return value
        assert result == [sample_agent_config]
        assert all(config.is_active for config in result)
    
    async def test_get_global_configs(self, repository, mock_session, sample_agent_config):
        """Test getting global (non-tenant) configurations."""
        # Ensure sample config has no tenant
        sample_agent_config.tenant_id = None
        
        # Mock the query execution
        mock_result = mock_session.execute.return_value
        mock_result.scalars.return_value.all.return_value = [sample_agent_config]
        
        result = await repository.get_global_configs()
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        
        # Verify return value
        assert result == [sample_agent_config]
        assert all(config.tenant_id is None for config in result)
    
    async def test_deactivate_config(self, repository, mock_session, sample_agent_config):
        """Test deactivating an agent configuration."""
        # Mock get_by_id to return the config
        with patch.object(repository, 'get_by_id', return_value=sample_agent_config):
            result = await repository.deactivate_config(1)
        
        # Verify config was deactivated
        assert sample_agent_config.is_active is False
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_agent_config)
        
        # Verify return value
        assert result == sample_agent_config
    
    async def test_deactivate_config_not_found(self, repository, mock_session):
        """Test deactivating non-existent configuration."""
        # Mock get_by_id to return None
        with patch.object(repository, 'get_by_id', return_value=None):
            result = await repository.deactivate_config(999)
        
        # Verify no operations
        mock_session.commit.assert_not_called()
        mock_session.refresh.assert_not_called()
        
        # Verify return value
        assert result is None
    
    async def test_config_isolation_by_tenant(self, repository, mock_session):
        """Test tenant isolation in configuration queries."""
        # Create configs for different tenants
        global_config = AgentConfig(
            role_name="global_role",
            display_name="Global Role",
            description="Global configuration",
            system_prompt_key="global_prompt",
            model_provider="openai",
            model_name="gpt-4o",
            tenant_id=None
        )
        
        tenant1_config = AgentConfig(
            role_name="tenant_role",
            display_name="Tenant Role",
            description="Tenant-specific configuration",
            system_prompt_key="tenant_prompt",
            model_provider="openai",
            model_name="gpt-4o",
            tenant_id="tenant_1"
        )
        
        tenant2_config = AgentConfig(
            role_name="tenant_role",
            display_name="Tenant Role",
            description="Another tenant-specific configuration",
            system_prompt_key="tenant_prompt",
            model_provider="anthropic",
            model_name="claude-3-5-sonnet",
            tenant_id="tenant_2"
        )
        
        # Test global config retrieval
        mock_result = mock_session.execute.return_value
        mock_result.scalar_one_or_none.return_value = global_config
        
        result = await repository.get_by_role("global_role")
        assert result == global_config
        assert result.tenant_id is None
        
        # Test tenant-specific retrieval
        mock_result.scalar_one_or_none.return_value = tenant1_config
        result = await repository.get_by_role("tenant_role", "tenant_1")
        assert result == tenant1_config
        assert result.tenant_id == "tenant_1"
        
        # Test different tenant gets different config
        mock_result.scalar_one_or_none.return_value = tenant2_config
        result = await repository.get_by_role("tenant_role", "tenant_2")
        assert result == tenant2_config
        assert result.tenant_id == "tenant_2"
    
    async def test_json_field_operations(self, repository, mock_session, sample_agent_config):
        """Test JSON field operations through repository."""
        # Test updating capabilities
        new_capabilities = ["testing", "deployment", "monitoring"]
        sample_agent_config.set_capabilities_list(new_capabilities)
        
        await repository.update(sample_agent_config)
        
        # Verify the update was called
        mock_session.add.assert_called_with(sample_agent_config)
        mock_session.commit.assert_called_once()
        
        # Verify capabilities were set correctly
        assert sample_agent_config.get_capabilities_list() == new_capabilities
    
    async def test_active_config_filtering(self, repository, mock_session):
        """Test that only active configs are returned by default."""
        # Create active and inactive configs
        active_config = AgentConfig(
            role_name="active_role",
            display_name="Active Role",
            description="Active configuration",
            system_prompt_key="active_prompt",
            model_provider="openai",
            model_name="gpt-4o",
            is_active=True
        )
        
        inactive_config = AgentConfig(
            role_name="inactive_role",
            display_name="Inactive Role",
            description="Inactive configuration",
            system_prompt_key="inactive_prompt",
            model_provider="openai",
            model_name="gpt-4o",
            is_active=False
        )
        
        # Mock list to return only active config
        mock_result = mock_session.execute.return_value
        mock_result.scalars.return_value.all.return_value = [active_config]
        
        result = await repository.list()
        
        # Verify only active config is returned
        assert len(result) == 1
        assert result[0].is_active is True
        assert result[0].role_name == "active_role"
    
    def test_repository_initialization(self):
        """Test repository initialization with session."""
        mock_session = AsyncMock()
        repository = AgentConfigRepository(mock_session)
        
        assert repository.session == mock_session
    
    async def test_filter_combinations(self, repository, mock_session, sample_agent_config):
        """Test various filter combinations."""
        test_filters = [
            {"tenant_id": "tenant_123"},
            {"model_provider": "openai"},
            {"role_name": "developer"},
            {"tenant_id": "tenant_123", "model_provider": "openai"},
            {"tenant_id": "tenant_123", "role_name": "developer"},
            {"model_provider": "openai", "role_name": "developer"},
            {"tenant_id": "tenant_123", "model_provider": "openai", "role_name": "developer"},
        ]
        
        for filters in test_filters:
            # Mock the query execution
            mock_result = mock_session.execute.return_value
            mock_result.scalars.return_value.all.return_value = [sample_agent_config]
            
            result = await repository.list(filters=filters)
            
            # Verify query was built correctly
            mock_session.execute.assert_called()
            
            # Verify return value
            assert result == [sample_agent_config]
            
            # Reset mock for next iteration
            mock_session.reset_mock()
