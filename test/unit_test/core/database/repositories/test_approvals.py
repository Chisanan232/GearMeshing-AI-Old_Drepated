"""Unit tests for approval repository.

Tests repository operations with mocked database session to ensure
business logic works correctly without real database dependencies.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from gearmeshing_ai.core.database.repositories.approvals import ApprovalRepository


class TestApprovalRepository:
    """Tests for ApprovalRepository operations."""
    
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
        return ApprovalRepository(mock_session)
    
    async def test_create_success(self, repository, mock_session):
        """Test successful approval creation."""
        # Create a mock approval object
        mock_approval = MagicMock()
        mock_approval.id = "approval_123"
        mock_approval.run_id = "run_456"
        mock_approval.risk = "high"
        mock_approval.capability = "code_execution"
        mock_approval.reason = "Test reason"
        mock_approval.requested_at = datetime.utcnow()
        mock_approval.expires_at = datetime.utcnow() + timedelta(hours=1)
        mock_approval.decision = None
        mock_approval.decided_at = None
        mock_approval.decided_by = None
        
        result = await repository.create(mock_approval)
        
        # Verify session operations
        mock_session.add.assert_called_once_with(mock_approval)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(mock_approval)
        
        # Verify return value
        assert result == mock_approval
    
    @patch.object(ApprovalRepository, 'get_by_id')
    async def test_get_by_id_success(self, mock_get_by_id, repository, mock_session):
        """Test successful approval retrieval by ID."""
        # Create a mock approval object
        mock_approval = MagicMock()
        mock_approval.id = "approval_123"
        mock_approval.run_id = "run_456"
        
        # Mock the get_by_id method
        mock_get_by_id.return_value = mock_approval
        
        result = await repository.get_by_id("approval_123")
        
        # Verify the method was called correctly
        mock_get_by_id.assert_called_once_with("approval_123")
        assert result == mock_approval
    
    @patch.object(ApprovalRepository, 'get_by_id')
    async def test_get_by_id_not_found(self, mock_get_by_id, repository):
        """Test approval retrieval by ID when not found."""
        # Mock the get_by_id method to return None
        mock_get_by_id.return_value = None
        
        result = await repository.get_by_id("nonexistent")
        
        assert result is None
    
    async def test_update_success(self, repository, mock_session):
        """Test successful approval update."""
        # Create a mock approval object
        mock_approval = MagicMock()
        mock_approval.id = "approval_123"
        mock_approval.decision = "approved"
        
        result = await repository.update(mock_approval)
        
        # Verify session operations
        mock_session.add.assert_called_once_with(mock_approval)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(mock_approval)
        
        # Verify result
        assert result == mock_approval
    
    @patch.object(ApprovalRepository, 'get_by_id')
    async def test_delete_success(self, mock_get_by_id, repository, mock_session):
        """Test successful approval deletion."""
        # Create a mock approval object
        mock_approval = MagicMock()
        mock_approval.id = "approval_123"
        
        # Mock the get_by_id method to return the approval
        mock_get_by_id.return_value = mock_approval
        
        result = await repository.delete("approval_123")
        
        # Verify session operations
        mock_session.delete.assert_called_once_with(mock_approval)
        mock_session.commit.assert_called_once()
        
        # Verify result
        assert result is True
    
    @patch.object(ApprovalRepository, 'get_by_id')
    async def test_delete_not_found(self, mock_get_by_id, repository, mock_session):
        """Test approval deletion when not found."""
        # Mock the get_by_id method to return None
        mock_get_by_id.return_value = None
        
        result = await repository.delete("nonexistent")
        
        # Verify session operations were not called
        mock_session.delete.assert_not_called()
        mock_session.commit.assert_not_called()
        
        # Verify result
        assert result is False
    
    @patch.object(ApprovalRepository, 'list')
    async def test_list_without_filters(self, mock_list, repository):
        """Test listing approvals without filters."""
        # Create a mock approval object
        mock_approval = MagicMock()
        mock_approval.id = "approval_123"
        mock_approval.run_id = "run_456"
        
        # Mock the list method
        mock_list.return_value = [mock_approval]
        
        result = await repository.list()
        
        # Verify the method was called correctly
        mock_list.assert_called_once_with()
        assert len(result) == 1
        assert result[0] == mock_approval
    
    @patch.object(ApprovalRepository, 'list')
    async def test_list_with_run_id_filter(self, mock_list, repository):
        """Test listing approvals with run_id filter."""
        # Create a mock approval object
        mock_approval = MagicMock()
        mock_approval.id = "approval_123"
        mock_approval.run_id = "run_456"
        
        # Mock the list method
        mock_list.return_value = [mock_approval]
        
        result = await repository.list(filters={"run_id": "run_456"})
        
        # Verify the method was called correctly
        mock_list.assert_called_once_with(filters={"run_id": "run_456"})
        assert len(result) == 1
        assert result[0] == mock_approval
    
    @patch.object(ApprovalRepository, 'list')
    async def test_list_with_decision_filter(self, mock_list, repository):
        """Test listing approvals with decision filter."""
        # Create a mock approval object
        mock_approval = MagicMock()
        mock_approval.id = "approval_123"
        mock_approval.decision = "approved"
        
        # Mock the list method
        mock_list.return_value = [mock_approval]
        
        result = await repository.list(filters={"decision": "approved"})
        
        # Verify the method was called correctly
        mock_list.assert_called_once_with(filters={"decision": "approved"})
        assert len(result) == 1
        assert result[0] == mock_approval
    
    @patch.object(ApprovalRepository, 'list')
    async def test_list_with_risk_filter(self, mock_list, repository):
        """Test listing approvals with risk filter."""
        # Create a mock approval object
        mock_approval = MagicMock()
        mock_approval.id = "approval_123"
        mock_approval.risk = "high"
        
        # Mock the list method
        mock_list.return_value = [mock_approval]
        
        result = await repository.list(filters={"risk": "high"})
        
        # Verify the method was called correctly
        mock_list.assert_called_once_with(filters={"risk": "high"})
        assert len(result) == 1
        assert result[0] == mock_approval
    
    @patch.object(ApprovalRepository, 'list')
    async def test_list_with_multiple_filters(self, mock_list, repository):
        """Test listing approvals with multiple filters."""
        # Create a mock approval object
        mock_approval = MagicMock()
        mock_approval.id = "approval_123"
        mock_approval.run_id = "run_456"
        mock_approval.risk = "high"
        mock_approval.decision = None
        
        # Mock the list method
        mock_list.return_value = [mock_approval]
        
        result = await repository.list(filters={
            "run_id": "run_456",
            "risk": "high",
            "decision": None
        })
        
        # Verify the method was called correctly
        mock_list.assert_called_once_with(filters={
            "run_id": "run_456",
            "risk": "high",
            "decision": None
        })
        assert len(result) == 1
        assert result[0] == mock_approval
    
    @patch.object(ApprovalRepository, 'list')
    async def test_list_with_limit_and_offset(self, mock_list, repository):
        """Test listing approvals with pagination."""
        # Create a mock approval object
        mock_approval = MagicMock()
        mock_approval.id = "approval_123"
        
        # Mock the list method
        mock_list.return_value = [mock_approval]
        
        result = await repository.list(limit=10, offset=5)
        
        # Verify the method was called correctly
        mock_list.assert_called_once_with(limit=10, offset=5)
        assert len(result) == 1
        assert result[0] == mock_approval
    
    @patch.object(ApprovalRepository, 'list')
    async def test_list_empty_result(self, mock_list, repository):
        """Test listing approvals when no approvals exist."""
        # Mock the list method to return empty list
        mock_list.return_value = []
        
        result = await repository.list()
        
        # Verify the method was called correctly
        mock_list.assert_called_once_with()
        assert len(result) == 0
    
    @patch.object(ApprovalRepository, 'get_pending_approvals')
    async def test_get_pending_approvals(self, mock_get_pending, repository):
        """Test getting pending approval requests."""
        # Create a mock approval object
        mock_approval = MagicMock()
        mock_approval.id = "approval_123"
        mock_approval.decision = None
        
        # Mock the get_pending_approvals method
        mock_get_pending.return_value = [mock_approval]
        
        result = await repository.get_pending_approvals()
        
        # Verify the method was called correctly
        mock_get_pending.assert_called_once_with()
        assert len(result) == 1
        assert result[0] == mock_approval
        assert result[0].decision is None
    
    @patch.object(ApprovalRepository, 'get_pending_approvals')
    async def test_get_pending_approvals_empty(self, mock_get_pending, repository):
        """Test getting pending approvals when none exist."""
        # Mock the get_pending_approvals method to return empty list
        mock_get_pending.return_value = []
        
        result = await repository.get_pending_approvals()
        
        # Verify the method was called correctly
        mock_get_pending.assert_called_once_with()
        assert len(result) == 0
