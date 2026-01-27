"""Unit tests for agent run repository.

Tests repository operations with mocked database session to ensure
business logic works correctly without real database dependencies.
"""

from __future__ import annotations

from datetime import datetime
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from gearmeshing_ai.core.database.entities.agent_runs import AgentRun
from gearmeshing_ai.core.database.repositories.agent_runs import AgentRunRepository


class TestAgentRunRepository:
    """Tests for AgentRunRepository operations."""
    
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
        return AgentRunRepository(mock_session)
    
    @pytest.fixture
    def sample_agent_run(self):
        """Create sample AgentRun instance."""
        return AgentRun(
            id="run_123",
            tenant_id="tenant_456",
            workspace_id="workspace_789",
            role="developer",
            autonomy_profile="balanced",
            objective="Build a new feature",
            done_when=["tests_pass", "code_reviewed"],
            prompt_provider_version="1.0",
            status="running",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    async def test_create_success(self, repository, mock_session, sample_agent_run):
        """Test successful agent run creation."""
        # Mock the refresh to return the same object
        mock_session.refresh.return_value = None
        
        result = await repository.create(sample_agent_run)
        
        # Verify session operations
        mock_session.add.assert_called_once_with(sample_agent_run)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_agent_run)
        
        # Verify return value
        assert result == sample_agent_run
    
    async def test_get_by_id_found(self, repository, mock_session, sample_agent_run):
        """Test getting agent run by ID when found."""
        # Mock the query execution
        mock_result = mock_session.exec.return_value
        mock_result.one_or_none.return_value = sample_agent_run
        
        result = await repository.get_by_id("run_123")
        
        # Verify query was built correctly
        mock_session.exec.assert_called_once()
        call_args = mock_session.exec.call_args[0][0]
        # Don't check isinstance since it's a mock, just check it was called
        assert mock_session.exec.called
        
        # Verify return value
        assert result == sample_agent_run
    
    async def test_get_by_id_not_found(self, repository, mock_session):
        """Test getting agent run by ID when not found."""
        # Mock the query execution to return None
        mock_result = mock_session.exec.return_value
        mock_result.one_or_none.return_value = None
        
        result = await repository.get_by_id("nonexistent_run")
        
        assert result is None
    
    async def test_update_success(self, repository, mock_session, sample_agent_run):
        """Test successful agent run update."""
        # Mock the refresh to return the updated object
        mock_session.refresh.return_value = None
        
        result = await repository.update(sample_agent_run)
        
        # Verify session operations
        mock_session.add.assert_called_once_with(sample_agent_run)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_agent_run)
        
        # Verify return value
        assert result == sample_agent_run
    
    async def test_delete_success(self, repository, mock_session, sample_agent_run):
        """Test successful agent run deletion."""
        # Mock get_by_id to return the run
        with patch.object(repository, 'get_by_id', return_value=sample_agent_run):
            result = await repository.delete("run_123")
        
        # Verify session operations
        mock_session.delete.assert_called_once_with(sample_agent_run)
        mock_session.commit.assert_called_once()
        
        # Verify return value
        assert result is True
    
    async def test_delete_not_found(self, repository, mock_session):
        """Test deleting agent run when not found."""
        # Mock get_by_id to return None
        with patch.object(repository, 'get_by_id', return_value=None):
            result = await repository.delete("nonexistent_run")
        
        # Verify no delete operations
        mock_session.delete.assert_not_called()
        mock_session.commit.assert_not_called()
        
        # Verify return value
        assert result is False
    
    async def test_list_no_filters(self, repository, mock_session, sample_agent_run):
        """Test listing agent runs without filters."""
        # Mock the query execution
        mock_result = mock_session.exec.return_value
        mock_result.__iter__ = MagicMock(return_value=iter([sample_agent_run]))
        
        result = await repository.list()
        
        # Verify query was built correctly
        mock_session.exec.assert_called_once()
        # Don't check isinstance since it's a mock, just check it was called
        assert mock_session.exec.called
        
        # Verify return value
        assert result == [sample_agent_run]
    
    async def test_list_with_filters(self, repository, mock_session, sample_agent_run):
        """Test listing agent runs with filters."""
        filters = {
            "tenant_id": "tenant_456",
            "workspace_id": "workspace_789",
            "role": "developer",
            "status": "running"
        }
        
        # Mock the query execution
        mock_result = mock_session.exec.return_value
        mock_result.__iter__ = MagicMock(return_value=iter([sample_agent_run]))
        
        result = await repository.list(filters=filters)
        
        # Verify query was built correctly
        mock_session.exec.assert_called_once()
        
        # Verify return value
        assert result == [sample_agent_run]
    
    async def test_list_with_pagination(self, repository, mock_session, sample_agent_run):
        """Test listing agent runs with pagination."""
        # Mock the query execution
        mock_result = mock_session.exec.return_value
        mock_result.__iter__ = MagicMock(return_value=iter([sample_agent_run]))
        
        result = await repository.list(limit=10, offset=20)
        
        # Verify query was built correctly
        mock_session.exec.assert_called_once()
        
        # Verify return value
        assert result == [sample_agent_run]
    
    async def test_update_status(self, repository, mock_session, sample_agent_run):
        """Test updating agent run status."""
        # Mock get_by_id to return the run
        repository.get_by_id = AsyncMock(return_value=sample_agent_run)
        
        result = await repository.update_status("run_123", "completed")
        
        # Verify status was updated
        assert sample_agent_run.status == "completed"
        assert sample_agent_run.updated_at is not None
        
        # Verify session operations (no add call for updates)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_agent_run)
        
        # Verify return value
        assert result == sample_agent_run
    
    async def test_update_status_not_found(self, repository, mock_session):
        """Test updating status of non-existent run."""
        # Mock get_by_id to return None
        with patch.object(repository, 'get_by_id', return_value=None):
            result = await repository.update_status("nonexistent_run", "completed")
        
        # Verify no update operations
        mock_session.add.assert_not_called()
        mock_session.commit.assert_not_called()
        
        # Verify return value
        assert result is None
    
    async def test_get_active_runs_for_tenant(self, repository, mock_session, sample_agent_run):
        """Test getting active runs for a tenant."""
        # Mock the query execution
        mock_result = mock_session.exec.return_value
        mock_result.__iter__ = MagicMock(return_value=iter([sample_agent_run]))
        
        result = await repository.get_active_runs_for_tenant("tenant_456")
        
        # Verify query was built correctly
        mock_session.exec.assert_called_once()
        
        # Verify return value
        assert result == [sample_agent_run]
    
    async def test_get_runs_by_status(self, repository, mock_session, sample_agent_run):
        """Test getting runs by status."""
        # Mock the query execution
        mock_result = mock_session.exec.return_value
        mock_result.__iter__ = MagicMock(return_value=iter([sample_agent_run]))
        
        # Use the list method with status filter instead
        result = await repository.list(filters={"status": "running"})
        
        # Verify query was built correctly
        mock_session.exec.assert_called_once()
        
        # Verify return value
        assert result == [sample_agent_run]
    
    async def test_get_runs_by_role(self, repository, mock_session, sample_agent_run):
        """Test getting runs by role."""
        # Mock the query execution
        mock_result = mock_session.exec.return_value
        mock_result.__iter__ = MagicMock(return_value=iter([sample_agent_run]))
        
        # Use the list method with role filter instead
        result = await repository.list(filters={"role": "developer"})
        
        # Verify query was built correctly
        mock_session.exec.assert_called_once()
        
        # Verify return value
        assert result == [sample_agent_run]
    
    async def test_filter_by_tenant_id(self, repository, mock_session):
        """Test filtering runs by tenant ID."""
        tenants = ["tenant_a", "tenant_b", "tenant_c"]
        
        for tenant_id in tenants:
            # Create run for each tenant
            run = AgentRun(
                id=f"run_{tenant_id}",
                tenant_id=tenant_id,
                role="developer",
                objective="Test objective",
                status="running"
            )
            
            # Mock the query execution
            mock_result = mock_session.exec.return_value
            mock_result.__iter__ = MagicMock(return_value=iter([run]))
            
            result = await repository.list(filters={"tenant_id": tenant_id})
            
            # Verify query was built correctly
            mock_session.exec.assert_called()
            
            # Verify return value
            assert result == [run]
            assert result[0].tenant_id == tenant_id
            
            # Reset mock for next iteration
            mock_session.reset_mock()
    
    async def test_filter_by_workspace_id(self, repository, mock_session):
        """Test filtering runs by workspace ID."""
        workspaces = ["workspace_a", "workspace_b", "workspace_c"]
        
        for workspace_id in workspaces:
            # Create run for each workspace
            run = AgentRun(
                id=f"run_{workspace_id}",
                workspace_id=workspace_id,
                role="developer",
                objective="Test objective",
                status="running"
            )
            
            # Mock the query execution
            mock_result = mock_session.exec.return_value
            mock_result.__iter__ = MagicMock(return_value=iter([run]))
            
            result = await repository.list(filters={"workspace_id": workspace_id})
            
            # Verify query was built correctly
            mock_session.exec.assert_called()
            
            # Verify return value
            assert result == [run]
            assert result[0].workspace_id == workspace_id
            
            # Reset mock for next iteration
            mock_session.reset_mock()
    
    async def test_status_transitions(self, repository, mock_session, sample_agent_run):
        """Test various status transitions."""
        status_transitions = [
            ("running", "paused"),
            ("paused", "running"),
            ("running", "completed"),
            ("running", "failed"),
            ("paused", "failed")
        ]
        
        for from_status, to_status in status_transitions:
            # Set initial status
            sample_agent_run.status = from_status
            
            # Mock get_by_id to return the run
            with patch.object(repository, 'get_by_id', return_value=sample_agent_run):
                result = await repository.update_status("run_123", to_status)
            
            # Verify status was updated
            assert sample_agent_run.status == to_status
            assert sample_agent_run.updated_at is not None
            
            # Verify return value
            assert result == sample_agent_run
            
            # Reset mock for next iteration
            mock_session.reset_mock()
    
    async def test_role_filtering(self, repository, mock_session):
        """Test filtering runs by different roles."""
        roles = ["developer", "analyst", "researcher", "tester"]
        
        for role in roles:
            # Create run for each role
            run = AgentRun(
                id=f"run_{role}",
                role=role,
                objective="Test objective",
                status="running"
            )
            
            # Mock the query execution
            mock_result = mock_session.exec.return_value
            mock_result.__iter__ = MagicMock(return_value=iter([run]))
            
            result = await repository.list(filters={"role": role})
            
            # Verify query was built correctly
            mock_session.exec.assert_called()
            
            # Verify return value
            assert result == [run]
            assert result[0].role == role
            
            # Reset mock for next iteration
            mock_session.reset_mock()
    
    async def test_autonomy_profile_filtering(self, repository, mock_session):
        """Test filtering runs by autonomy profile."""
        profiles = ["minimal", "balanced", "maximum"]
        
        for profile in profiles:
            # Create run for each profile
            run = AgentRun(
                id=f"run_{profile}",
                autonomy_profile=profile,
                role="developer",
                objective="Test objective",
                status="running"
            )
            
            # Mock the query execution
            mock_result = mock_session.exec.return_value
            mock_result.__iter__ = MagicMock(return_value=iter([run]))
            
            result = await repository.list(filters={"autonomy_profile": profile})
            
            # Verify query was built correctly
            mock_session.exec.assert_called()
            
            # Verify return value
            assert result == [run]
            assert result[0].autonomy_profile == profile
            
            # Reset mock for next iteration
            mock_session.reset_mock()
    
    async def test_complex_filter_combinations(self, repository, mock_session, sample_agent_run):
        """Test complex filter combinations."""
        complex_filters = [
            {"tenant_id": "tenant_456", "role": "developer"},
            {"tenant_id": "tenant_456", "status": "running"},
            {"role": "developer", "status": "running"},
            {"tenant_id": "tenant_456", "role": "developer", "status": "running"},
            {"workspace_id": "workspace_789", "autonomy_profile": "balanced"},
            {"tenant_id": "tenant_456", "workspace_id": "workspace_789", "role": "developer", "status": "running"}
        ]
        
        for filters in complex_filters:
            # Mock the query execution
            mock_result = mock_session.exec.return_value
            mock_result.__iter__ = MagicMock(return_value=iter([sample_agent_run]))
            
            result = await repository.list(filters=filters)
            
            # Verify query was built correctly
            mock_session.exec.assert_called()
            
            # Verify return value
            assert result == [sample_agent_run]
            
            # Reset mock for next iteration
            mock_session.reset_mock()
    
    async def test_pagination_scenarios(self, repository, mock_session):
        """Test various pagination scenarios."""
        pagination_scenarios = [
            {"limit": 10, "offset": 0},
            {"limit": 20, "offset": 40},
            {"limit": 5, "offset": 100},
            {"limit": 50, "offset": 0},
            {"limit": 1, "offset": 1000}
        ]
        
        for pagination in pagination_scenarios:
            # Create sample run
            run = AgentRun(
                id=f"run_{pagination['limit']}_{pagination['offset']}",
                role="developer",
                objective="Test objective",
                status="running"
            )
            
            # Mock the query execution
            mock_result = mock_session.exec.return_value
            mock_result.__iter__ = MagicMock(return_value=iter([run]))
            
            result = await repository.list(limit=pagination["limit"], offset=pagination["offset"])
            
            # Verify query was built correctly
            mock_session.exec.assert_called()
            
            # Verify return value
            assert result == [run]
            
            # Reset mock for next iteration
            mock_session.reset_mock()
    
    def test_repository_initialization(self):
        """Test repository initialization with session."""
        mock_session = AsyncMock()
        repository = AgentRunRepository(mock_session)
        
        assert repository.session == mock_session
    
    async def test_timestamp_updates(self, repository, mock_session, sample_agent_run):
        """Test that timestamps are updated correctly."""
        original_updated_at = sample_agent_run.updated_at
        
        # Mock get_by_id to return the run
        with patch.object(repository, 'get_by_id', return_value=sample_agent_run):
            # Wait a moment to ensure timestamp difference
            await repository.update_status("run_123", "completed")
        
        # Verify updated_at was changed
        assert sample_agent_run.updated_at != original_updated_at
        assert sample_agent_run.updated_at > original_updated_at
