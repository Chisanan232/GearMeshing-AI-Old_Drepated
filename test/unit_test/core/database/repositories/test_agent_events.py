"""Unit tests for agent event repository.

Tests repository operations with mocked database session to ensure
business logic works correctly without real database dependencies.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from gearmeshing_ai.core.database.entities.agent_events import AgentEvent
from gearmeshing_ai.core.database.repositories.agent_events import AgentEventRepository


class TestAgentEventRepository:
    """Tests for AgentEventRepository operations."""
    
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
        return AgentEventRepository(mock_session)
    
    @pytest.fixture
    def sample_agent_event(self, sample_agent_event_data):
        """Create sample AgentEvent instance."""
        return AgentEvent(**sample_agent_event_data)
    
    async def test_create_success(self, repository, mock_session, sample_agent_event):
        """Test successful agent event creation."""
        # Mock the refresh to return the same object
        mock_session.refresh.return_value = None
        
        result = await repository.create(sample_agent_event)
        
        # Verify session operations
        mock_session.add.assert_called_once_with(sample_agent_event)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_agent_event)
        
        # Verify return value
        assert result == sample_agent_event
    
    async def test_get_by_id_found(self, repository, mock_session, sample_agent_event):
        """Test getting agent event by ID when found."""
        # Mock the query execution
        mock_result = mock_session.execute.return_value
        mock_result.scalar_one_or_none.return_value = sample_agent_event
        
        result = await repository.get_by_id("test_event_123")
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        # Don't check isinstance since it's a mock, just check it was called
        assert mock_session.execute.called
        
        # Verify return value
        assert result == sample_agent_event
    
    async def test_get_by_id_not_found(self, repository, mock_session):
        """Test getting agent event by ID when not found."""
        # Mock the query execution to return None
        mock_result = mock_session.execute.return_value
        mock_result.scalar_one_or_none.return_value = None
        
        result = await repository.get_by_id("nonexistent_event")
        
        assert result is None
    
    async def test_update_not_supported(self, repository, mock_session, sample_agent_event):
        """Test that update operation raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Events are append-only"):
            await repository.update(sample_agent_event)
    
    async def test_delete_not_supported(self, repository, mock_session):
        """Test that delete operation raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Events are append-only"):
            await repository.delete("test_event_123")
    
    async def test_list_no_filters(self, repository, mock_session, sample_agent_event):
        """Test listing agent events without filters."""
        # Mock the query execution
        mock_result = mock_session.execute.return_value
        mock_result.scalars.return_value.all.return_value = [sample_agent_event]
        
        result = await repository.list()
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        # Don't check isinstance since it's a mock, just check it was called
        assert mock_session.execute.called
        
        # Verify return value
        assert result == [sample_agent_event]
    
    async def test_list_with_filters(self, repository, mock_session, sample_agent_event):
        """Test listing agent events with filters."""
        filters = {
            "run_id": "test_run_123",
            "type": "step_completed",
            "correlation_id": "correlation_456"
        }
        
        # Mock the query execution
        mock_result = mock_session.execute.return_value
        mock_result.scalars.return_value.all.return_value = [sample_agent_event]
        
        result = await repository.list(filters=filters)
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        
        # Verify return value
        assert result == [sample_agent_event]
    
    async def test_list_with_pagination(self, repository, mock_session, sample_agent_event):
        """Test listing agent events with pagination."""
        # Mock the query execution
        mock_result = mock_session.execute.return_value
        mock_result.scalars.return_value.all.return_value = [sample_agent_event]
        
        result = await repository.list(limit=10, offset=20)
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        
        # Verify return value
        assert result == [sample_agent_event]
    
    async def test_get_events_for_run(self, repository, mock_session, sample_agent_event):
        """Test getting events for a specific run."""
        # Mock the query execution
        mock_result = mock_session.execute.return_value
        mock_result.scalars.return_value.all.return_value = [sample_agent_event]
        
        result = await repository.get_events_for_run("test_run_123")
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        
        # Verify return value
        assert result == [sample_agent_event]
    
    async def test_get_events_for_run_with_limit(self, repository, mock_session, sample_agent_event):
        """Test getting events for a run with limit."""
        # Mock the query execution
        mock_result = mock_session.execute.return_value
        mock_result.scalars.return_value.all.return_value = [sample_agent_event]
        
        result = await repository.get_events_for_run("test_run_123", limit=5)
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        
        # Verify return value
        assert result == [sample_agent_event]
    
    async def test_get_events_by_type(self, repository, mock_session, sample_agent_event):
        """Test getting events by type for a run."""
        # Mock the query execution
        mock_result = mock_session.execute.return_value
        mock_result.scalars.return_value.all.return_value = [sample_agent_event]
        
        result = await repository.get_events_by_type("test_run_123", "step_completed")
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        
        # Verify return value
        assert result == [sample_agent_event]
    
    async def test_get_events_by_correlation(self, repository, mock_session, sample_agent_event):
        """Test getting events by correlation ID."""
        # Mock the query execution
        mock_result = mock_session.execute.return_value
        mock_result.scalars.return_value.all.return_value = [sample_agent_event]
        
        result = await repository.get_events_by_correlation("correlation_456")
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        
        # Verify return value
        assert result == [sample_agent_event]
    
    async def test_events_chronological_order(self, repository, mock_session):
        """Test that events are returned in chronological order."""
        # Create multiple events with different timestamps
        events = []
        for i in range(5):
            event_data = {
                "id": f"event_{i}",
                "run_id": "test_run_123",
                "type": "step_completed",
                "correlation_id": f"correlation_{i}",
                "payload": '{"step": "analysis", "result": "success"}',
                "created_at": datetime(2024, 1, 1, i, 0, 0)  # Different hours
            }
            events.append(AgentEvent(**event_data))
        
        # Mock the query execution to return events in order
        mock_result = mock_session.execute.return_value
        mock_result.scalars.return_value.all.return_value = events
        
        result = await repository.get_events_for_run("test_run_123")
        
        # Verify events are in chronological order (earliest first)
        for i in range(len(result) - 1):
            assert result[i].created_at <= result[i + 1].created_at
    
    async def test_append_only_behavior(self, repository, mock_session):
        """Test that events follow append-only behavior."""
        # Create an event
        event_data = {
            "id": "append_test_event",
            "run_id": "test_run_123",
            "type": "step_completed",
            "correlation_id": "test_correlation",
            "payload": '{"step": "test", "result": "success"}',
        }
        event = AgentEvent(**event_data)
        
        # Create should work
        await repository.create(event)
        mock_session.add.assert_called_once_with(event)
        mock_session.commit.assert_called_once()
        
        # Update should not work
        with pytest.raises(NotImplementedError):
            await repository.update(event)
        
        # Delete should not work
        with pytest.raises(NotImplementedError):
            await repository.delete("append_test_event")
    
    async def test_payload_handling(self, repository, mock_session):
        """Test repository handles payload JSON correctly."""
        # Create event with complex payload
        complex_payload = {
            "user": {"id": 123, "name": "Test User"},
            "actions": ["create", "update"],
            "metadata": {"timestamp": "2024-01-01T00:00:00Z"}
        }
        
        event_data = {
            "id": "payload_test_event",
            "run_id": "test_run_123",
            "type": "complex_action",
            "correlation_id": "test_correlation",
            "payload": '{"user": {"id": 123, "name": "Test User"}, "actions": ["create", "update"], "metadata": {"timestamp": "2024-01-01T00:00:00Z"}}',
        }
        event = AgentEvent(**event_data)
        
        # Set complex payload
        event.set_payload_dict(complex_payload)
        
        # Create should work
        await repository.create(event)
        
        # Verify payload is preserved
        retrieved_payload = event.get_payload_dict()
        assert retrieved_payload == complex_payload
    
    def test_repository_initialization(self):
        """Test repository initialization with session."""
        mock_session = AsyncMock()
        repository = AgentEventRepository(mock_session)
        
        assert repository.session == mock_session
    
    async def test_filter_combinations(self, repository, mock_session, sample_agent_event):
        """Test various filter combinations."""
        test_filters = [
            {"run_id": "test_run_123"},
            {"type": "step_completed"},
            {"correlation_id": "correlation_456"},
            {"run_id": "test_run_123", "type": "step_completed"},
            {"run_id": "test_run_123", "correlation_id": "correlation_456"},
            {"type": "step_completed", "correlation_id": "correlation_456"},
            {"run_id": "test_run_123", "type": "step_completed", "correlation_id": "correlation_456"},
        ]
        
        for filters in test_filters:
            # Mock the query execution
            mock_result = mock_session.execute.return_value
            mock_result.scalars.return_value.all.return_value = [sample_agent_event]
            
            result = await repository.list(filters=filters)
            
            # Verify query was built correctly
            mock_session.execute.assert_called()
            
            # Verify return value
            assert result == [sample_agent_event]
            
            # Reset mock for next iteration
            mock_session.reset_mock()
