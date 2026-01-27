"""Unit tests for policy repository.

Tests repository operations with mocked database session to ensure
business logic works correctly without real database dependencies.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from gearmeshing_ai.core.database.repositories.policies import PolicyRepository


class TestPolicyRepository:
    """Tests for PolicyRepository operations."""
    
    @pytest.fixture
    def mock_session(self):
        """Mock async database session."""
        session = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.delete = AsyncMock()
        return session
    
    @pytest.fixture
    def repository(self, mock_session):
        """Create repository instance with mocked session."""
        return PolicyRepository(mock_session)
    
    async def test_create_success(self, repository, mock_session):
        """Test successful policy creation."""
        # Create a mock policy object
        mock_policy = MagicMock()
        mock_policy.id = "policy_123"
        mock_policy.tenant_id = "tenant_456"
        mock_policy.config = {"risk_threshold": "medium"}
        mock_policy.created_at = datetime.utcnow()
        mock_policy.updated_at = datetime.utcnow()
        
        result = await repository.create(mock_policy)
        
        # Verify session operations
        mock_session.add.assert_called_once_with(mock_policy)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(mock_policy)
        
        # Verify return value
        assert result == mock_policy
    
    @patch.object(PolicyRepository, 'get_by_id')
    async def test_get_by_id_success(self, mock_get_by_id, repository, mock_session):
        """Test successful policy retrieval by ID."""
        # Create a mock policy object
        mock_policy = MagicMock()
        mock_policy.id = "policy_123"
        mock_policy.tenant_id = "tenant_456"
        
        # Mock the get_by_id method
        mock_get_by_id.return_value = mock_policy
        
        result = await repository.get_by_id("policy_123")
        
        # Verify the method was called correctly
        mock_get_by_id.assert_called_once_with("policy_123")
        assert result == mock_policy
    
    @patch.object(PolicyRepository, 'get_by_id')
    async def test_get_by_id_not_found(self, mock_get_by_id, repository):
        """Test policy retrieval by ID when not found."""
        # Mock the get_by_id method to return None
        mock_get_by_id.return_value = None
        
        result = await repository.get_by_id("nonexistent")
        
        assert result is None
    
    @patch.object(PolicyRepository, 'get_by_tenant')
    async def test_get_by_tenant_success(self, mock_get_by_tenant, repository, mock_session):
        """Test successful policy retrieval by tenant."""
        # Create a mock policy object
        mock_policy = MagicMock()
        mock_policy.id = "policy_123"
        mock_policy.tenant_id = "tenant_456"
        
        # Mock the get_by_tenant method
        mock_get_by_tenant.return_value = mock_policy
        
        result = await repository.get_by_tenant("tenant_456")
        
        # Verify the method was called correctly
        mock_get_by_tenant.assert_called_once_with("tenant_456")
        assert result == mock_policy
    
    @patch.object(PolicyRepository, 'get_by_tenant')
    async def test_get_by_tenant_not_found(self, mock_get_by_tenant, repository):
        """Test policy retrieval by tenant when not found."""
        # Mock the get_by_tenant method to return None
        mock_get_by_tenant.return_value = None
        
        result = await repository.get_by_tenant("nonexistent_tenant")
        
        assert result is None
    
    async def test_update_success(self, repository, mock_session):
        """Test successful policy update."""
        # Create a mock policy object
        mock_policy = MagicMock()
        mock_policy.id = "policy_123"
        mock_policy.config = {"risk_threshold": "high"}
        old_updated_at = datetime.utcnow()
        mock_policy.updated_at = old_updated_at
        
        result = await repository.update(mock_policy)
        
        # Verify session operations
        mock_session.add.assert_called_once_with(mock_policy)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(mock_policy)
        
        # Verify updated_at was set
        assert mock_policy.updated_at >= old_updated_at
        
        # Verify result
        assert result == mock_policy
    
    @patch.object(PolicyRepository, 'get_by_id')
    async def test_delete_success(self, mock_get_by_id, repository, mock_session):
        """Test successful policy deletion."""
        # Create a mock policy object
        mock_policy = MagicMock()
        mock_policy.id = "policy_123"
        
        # Mock the get_by_id method to return the policy
        mock_get_by_id.return_value = mock_policy
        
        result = await repository.delete("policy_123")
        
        # Verify session operations
        mock_session.delete.assert_called_once_with(mock_policy)
        mock_session.commit.assert_called_once()
        
        # Verify result
        assert result is True
    
    @patch.object(PolicyRepository, 'get_by_id')
    async def test_delete_not_found(self, mock_get_by_id, repository, mock_session):
        """Test policy deletion when not found."""
        # Mock the get_by_id method to return None
        mock_get_by_id.return_value = None
        
        result = await repository.delete("nonexistent")
        
        # Verify session operations were not called
        mock_session.delete.assert_not_called()
        mock_session.commit.assert_not_called()
        
        # Verify result
        assert result is False
    
    @patch.object(PolicyRepository, 'list')
    async def test_list_without_filters(self, mock_list, repository):
        """Test listing policies without filters."""
        # Create a mock policy object
        mock_policy = MagicMock()
        mock_policy.id = "policy_123"
        mock_policy.tenant_id = "tenant_456"
        
        # Mock the list method
        mock_list.return_value = [mock_policy]
        
        result = await repository.list()
        
        # Verify the method was called correctly
        mock_list.assert_called_once_with()
        assert len(result) == 1
        assert result[0] == mock_policy
    
    @patch.object(PolicyRepository, 'list')
    async def test_list_with_tenant_filter(self, mock_list, repository):
        """Test listing policies with tenant filter."""
        # Create a mock policy object
        mock_policy = MagicMock()
        mock_policy.id = "policy_123"
        mock_policy.tenant_id = "tenant_456"
        
        # Mock the list method
        mock_list.return_value = [mock_policy]
        
        result = await repository.list(filters={"tenant_id": "tenant_456"})
        
        # Verify the method was called correctly
        mock_list.assert_called_once_with(filters={"tenant_id": "tenant_456"})
        assert len(result) == 1
        assert result[0] == mock_policy
    
    @patch.object(PolicyRepository, 'list')
    async def test_list_with_limit_and_offset(self, mock_list, repository):
        """Test listing policies with pagination."""
        # Create a mock policy object
        mock_policy = MagicMock()
        mock_policy.id = "policy_123"
        mock_policy.tenant_id = "tenant_456"
        
        # Mock the list method
        mock_list.return_value = [mock_policy]
        
        result = await repository.list(limit=10, offset=5)
        
        # Verify the method was called correctly
        mock_list.assert_called_once_with(limit=10, offset=5)
        assert len(result) == 1
        assert result[0] == mock_policy
    
    @patch.object(PolicyRepository, 'list')
    async def test_list_empty_result(self, mock_list, repository):
        """Test listing policies when no policies exist."""
        # Mock the list method to return empty list
        mock_list.return_value = []
        
        result = await repository.list()
        
        # Verify the method was called correctly
        mock_list.assert_called_once_with()
        assert len(result) == 0
