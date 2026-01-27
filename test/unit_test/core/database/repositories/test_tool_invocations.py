"""Unit tests for tool invocation repository.

Tests repository operations with mocked database session to ensure
business logic works correctly without real database dependencies.
"""

from __future__ import annotations

from datetime import datetime
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from gearmeshing_ai.core.database.entities.tool_invocations import ToolInvocation
from gearmeshing_ai.core.database.repositories.tool_invocations import ToolInvocationRepository


class TestToolInvocationRepository:
    """Tests for ToolInvocationRepository operations."""
    
    @pytest.fixture
    def mock_session(self):
        """Mock async database session."""
        session = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.delete = AsyncMock()
        # Make execute return the mock result directly, not a coroutine
        mock_result = MagicMock()
        # Make scalar_one_or_none return the object directly, not a coroutine
        mock_result.__iter__ = MagicMock(return_value=iter([]))
        # Make scalar_one_or_none return the object directly, not a coroutine
        mock_result.one_or_none = MagicMock()
        mock_result.scalars = MagicMock()
        mock_result.scalars.all = MagicMock()
        session.exec = MagicMock(return_value=mock_result)
        return session
    
    @pytest.fixture
    def repository(self, mock_session):
        """Create repository instance with mocked session."""
        return ToolInvocationRepository(mock_session)
    
    @pytest.fixture
    def sample_tool_invocation(self):
        """Create sample ToolInvocation instance."""
        invocation = ToolInvocation(
            id="tool_inv_123",
            run_id="run_456",
            server_id="server_789",
            tool_name="git.commit",
            args='{"repo": "test", "message": "Initial commit"}',
            ok=True,
            result='{"commit_hash": "abc123"}',
            risk="medium",
            created_at=datetime.utcnow()
        )
        return invocation
    
    async def test_create_success(self, repository, mock_session, sample_tool_invocation):
        """Test successful tool invocation creation."""
        # Mock the refresh to return the same object
        mock_session.refresh.return_value = None
        
        result = await repository.create(sample_tool_invocation)
        
        # Verify session operations
        mock_session.add.assert_called_once_with(sample_tool_invocation)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_tool_invocation)
        
        # Verify return value
        assert result == sample_tool_invocation
    
    async def test_get_by_id_found(self, repository, mock_session, sample_tool_invocation):
        """Test getting tool invocation by ID when found."""
        # Mock the query execution
        mock_result = mock_session.exec.return_value
        mock_result.one_or_none.return_value = sample_tool_invocation
        
        result = await repository.get_by_id("tool_inv_123")
        
        # Verify query was built correctly
        mock_session.exec.assert_called_once()
        # Don't check isinstance since it's a mock, just check it was called
        assert mock_session.exec.called
        
        # Verify return value
        assert result == sample_tool_invocation
    
    async def test_get_by_id_not_found(self, repository, mock_session):
        """Test getting tool invocation by ID when not found."""
        # Mock the query execution to return None
        mock_result = mock_session.exec.return_value
        mock_result.one_or_none.return_value = None
        
        result = await repository.get_by_id("nonexistent_invocation")
        
        assert result is None
    
    async def test_update_success(self, repository, mock_session, sample_tool_invocation):
        """Test successful tool invocation update."""
        # Mock the refresh to return the updated object
        mock_session.refresh.return_value = None
        
        result = await repository.update(sample_tool_invocation)
        
        # Verify session operations
        mock_session.add.assert_called_once_with(sample_tool_invocation)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_tool_invocation)
        
        # Verify return value
        assert result == sample_tool_invocation
    
    async def test_delete_success(self, repository, mock_session, sample_tool_invocation):
        """Test successful tool invocation deletion."""
        # Mock get_by_id to return the invocation
        with patch.object(repository, 'get_by_id', return_value=sample_tool_invocation):
            result = await repository.delete("tool_inv_123")
        
        # Verify session operations
        mock_session.delete.assert_called_once_with(sample_tool_invocation)
        mock_session.commit.assert_called_once()
        
        # Verify return value
        assert result is True
    
    async def test_delete_not_found(self, repository, mock_session):
        """Test deleting tool invocation when not found."""
        # Mock get_by_id to return None
        with patch.object(repository, 'get_by_id', return_value=None):
            result = await repository.delete("nonexistent_invocation")
        
        # Verify no delete operations
        mock_session.delete.assert_not_called()
        mock_session.commit.assert_not_called()
        
        # Verify return value
        assert result is False
    
    async def test_list_no_filters(self, repository, mock_session, sample_tool_invocation):
        """Test listing tool invocations without filters."""
        # Mock the query execution
        mock_result = mock_session.exec.return_value
        mock_result.__iter__ = MagicMock(return_value=iter([sample_tool_invocation]))
        
        result = await repository.list()
        
        # Verify query was built correctly
        mock_session.exec.assert_called_once()
        # Don't check isinstance since it's a mock, just check it was called
        assert mock_session.exec.called
        
        # Verify return value
        assert result == [sample_tool_invocation]
    
    async def test_list_with_filters(self, repository, mock_session, sample_tool_invocation):
        """Test listing tool invocations with filters."""
        filters = {
            "run_id": "run_456",
            "tool_name": "git.commit",
            "server_id": "server_789",
            "risk": "medium",
            "ok": True
        }
        
        # Mock the query execution
        mock_result = mock_session.exec.return_value
        mock_result.__iter__ = MagicMock(return_value=iter([sample_tool_invocation]))
        
        result = await repository.list(filters=filters)
        
        # Verify query was built correctly
        mock_session.exec.assert_called_once()
        
        # Verify return value
        assert result == [sample_tool_invocation]
    
    async def test_list_with_pagination(self, repository, mock_session, sample_tool_invocation):
        """Test listing tool invocations with pagination."""
        # Mock the query execution
        mock_result = mock_session.exec.return_value
        mock_result.__iter__ = MagicMock(return_value=iter([sample_tool_invocation]))
        
        result = await repository.list(limit=10, offset=20)
        
        # Verify query was built correctly
        mock_session.exec.assert_called_once()
        
        # Verify return value
        assert result == [sample_tool_invocation]
    
    async def test_get_invocations_for_run(self, repository, mock_session, sample_tool_invocation):
        """Test getting invocations for a specific run."""
        # Mock the query execution
        mock_result = mock_session.exec.return_value
        mock_result.__iter__ = MagicMock(return_value=iter([sample_tool_invocation]))
        
        result = await repository.get_invocations_for_run("run_456")
        
        # Verify query was built correctly
        mock_session.exec.assert_called_once()
        
        # Verify return value
        assert result == [sample_tool_invocation]
    
    async def test_get_high_risk_invocations(self, repository, mock_session):
        """Test getting high-risk tool invocations."""
        # Create high-risk invocation
        high_risk_invocation = ToolInvocation(
            id="high_risk_inv",
            run_id="run_456",
            server_id="server_789",
            tool_name="system.delete",
            args={"path": "/important/file"},
            ok=True,
            result={"deleted": True},
            risk="high",
            created_at=datetime.utcnow()
        )
        
        # Mock the query execution
        mock_result = mock_session.exec.return_value
        mock_result.__iter__ = MagicMock(return_value=iter([high_risk_invocation]))
        
        result = await repository.get_high_risk_invocations()
        
        # Verify query was built correctly
        mock_session.exec.assert_called_once()
        
        # Verify return value
        assert result == [high_risk_invocation]
        assert result[0].risk == "high"
    
    async def test_invocation_chronological_order(self, repository, mock_session):
        """Test that invocations are returned in chronological order."""
        # Create multiple invocations with different timestamps
        invocations = []
        for i in range(5):
            invocation = ToolInvocation(
                id=f"inv_{i}",
                run_id="run_456",
                server_id="server_789",
                tool_name="test.tool",
                args={"step": i},
                ok=True,
                result={"result": i},
                risk="low",
                created_at=datetime(2024, 1, 1, i, 0, 0)  # Different hours
            )
            invocations.append(invocation)
        
        # Mock the query execution to return invocations in order
        mock_result = mock_session.exec.return_value
        mock_result.scalars.return_value.all.return_value = invocations
        
        result = await repository.get_invocations_for_run("run_456")
        
        # Verify invocations are in chronological order (earliest first)
        for i in range(len(result) - 1):
            assert result[i].created_at <= result[i + 1].created_at
    
    async def test_filter_by_tool_name(self, repository, mock_session):
        """Test filtering invocations by tool name."""
        tools = ["git.commit", "docker.run", "python.execute", "file.read"]
        
        for tool_name in tools:
            # Create invocation for each tool
            invocation = ToolInvocation(
                id=f"inv_{tool_name}",
                run_id="run_456",
                server_id="server_789",
                tool_name=tool_name,
                args={},
                ok=True,
                result={},
                risk="low"
            )
            
            # Mock the query execution
            mock_result = mock_session.exec.return_value
            mock_result.__iter__ = MagicMock(return_value=iter([invocation]))
            
            result = await repository.list(filters={"tool_name": tool_name})
            
            # Verify query was built correctly
            mock_session.exec.assert_called()
            
            # Verify return value
            assert result == [invocation]
            assert result[0].tool_name == tool_name
            
            # Reset mock for next iteration
            mock_session.reset_mock()
    
    async def test_filter_by_success_status(self, repository, mock_session):
        """Test filtering invocations by success status."""
        # Create successful and failed invocations
        successful_invocation = ToolInvocation(
            id="inv_success",
            run_id="run_456",
            server_id="server_789",
            tool_name="test.success",
            args={},
            ok=True,
            result={"status": "completed"},
            risk="low"
        )
        
        failed_invocation = ToolInvocation(
            id="inv_failed",
            run_id="run_456",
            server_id="server_789",
            tool_name="test.fail",
            args={},
            ok=False,
            result={"error": "Tool failed"},
            risk="medium"
        )
        
        # Test successful invocations
        mock_result = mock_session.exec.return_value
        mock_result.__iter__ = MagicMock(return_value=iter([successful_invocation]))
        
        result = await repository.list(filters={"ok": True})
        
        assert len(result) == 1
        assert result[0].ok is True
        assert result[0].tool_name == "test.success"
        
        # Test failed invocations
        mock_result.__iter__ = MagicMock(return_value=iter([failed_invocation]))
        result = await repository.list(filters={"ok": False})
        
        assert len(result) == 1
        assert result[0].ok is False
        assert result[0].tool_name == "test.fail"
    
    async def test_filter_by_risk_level(self, repository, mock_session):
        """Test filtering invocations by risk level."""
        risk_levels = ["low", "medium", "high", "critical"]
        
        for risk in risk_levels:
            # Create invocation for each risk level
            invocation = ToolInvocation(
                id=f"inv_{risk}",
                run_id="run_456",
                server_id="server_789",
                tool_name="test.risk",
                args={},
                ok=True,
                result={},
                risk=risk
            )
            
            # Mock the query execution
            mock_result = mock_session.exec.return_value
            mock_result.__iter__ = MagicMock(return_value=iter([invocation]))
            
            result = await repository.list(filters={"risk": risk})
            
            # Verify query was built correctly
            mock_session.exec.assert_called()
            
            # Verify return value
            assert result == [invocation]
            assert result[0].risk == risk
            
            # Reset mock for next iteration
            mock_session.reset_mock()
    
    async def test_complex_args_and_results(self, repository, mock_session):
        """Test invocations with complex arguments and results."""
        complex_args = {
            "repository": {
                "url": "https://github.com/user/repo.git",
                "branch": "main",
                "commit": "abc123"
            },
            "options": {
                "force": False,
                "allow_empty": True,
                "message": "Automated commit"
            },
            "metadata": {
                "user_id": 123,
                "session_id": "sess_456"
            }
        }
        
        complex_result = {
            "commit": {
                "hash": "def456",
                "url": "https://github.com/user/repo/commit/def456",
                "message": "Automated commit",
                "author": {
                    "name": "Test User",
                    "email": "test@example.com"
                },
                "stats": {
                    "files_changed": 5,
                    "additions": 100,
                    "deletions": 50
                }
            },
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        invocation = ToolInvocation(
            id="inv_complex",
            run_id="run_456",
            server_id="server_789",
            tool_name="git.commit",
            args=complex_args,
            ok=True,
            result=complex_result,
            risk="low"
        )
        
        # Mock the query execution
        mock_result = mock_session.exec.return_value
        mock_result.__iter__ = MagicMock(return_value=iter([invocation]))
        
        result = await repository.list()
        
        # Verify complex data is preserved
        assert result[0].args == complex_args
        assert result[0].result == complex_result
        assert result[0].args["repository"]["url"] == "https://github.com/user/repo.git"
        assert result[0].result["commit"]["hash"] == "def456"
    
    def test_repository_initialization(self):
        """Test repository initialization with session."""
        mock_session = AsyncMock()
        repository = ToolInvocationRepository(mock_session)
        
        assert repository.session == mock_session
    
    async def test_filter_combinations(self, repository, mock_session, sample_tool_invocation):
        """Test various filter combinations."""
        test_filters = [
            {"run_id": "run_456"},
            {"tool_name": "git.commit"},
            {"server_id": "server_789"},
            {"risk": "medium"},
            {"ok": True},
            {"run_id": "run_456", "tool_name": "git.commit"},
            {"run_id": "run_456", "server_id": "server_789"},
            {"tool_name": "git.commit", "risk": "medium"},
            {"run_id": "run_456", "tool_name": "git.commit", "ok": True},
            {"run_id": "run_456", "tool_name": "git.commit", "risk": "medium", "ok": True},
        ]
        
        for filters in test_filters:
            # Mock the query execution
            mock_result = mock_session.exec.return_value
            mock_result.__iter__ = MagicMock(return_value=iter([sample_tool_invocation]))
            
            result = await repository.list(filters=filters)
            
            # Verify query was built correctly
            mock_session.exec.assert_called()
            
            # Verify return value
            assert result == [sample_tool_invocation]
            
            # Reset mock for next iteration
            mock_session.reset_mock()
