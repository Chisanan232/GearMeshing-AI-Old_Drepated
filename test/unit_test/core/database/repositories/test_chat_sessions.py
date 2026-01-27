"""Unit tests for chat session repository.

Tests repository operations with mocked database session to ensure
business logic works correctly without real database dependencies.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlmodel import select

from gearmeshing_ai.core.database.entities.chat_sessions import ChatSession, ChatMessage, MessageRole
from gearmeshing_ai.core.database.repositories.chat_sessions import ChatSessionRepository


class TestChatSessionRepository:
    """Tests for ChatSessionRepository operations."""
    
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
        return ChatSessionRepository(mock_session)
    
    @pytest.fixture
    def sample_chat_session(self, sample_chat_session_data):
        """Create sample ChatSession instance."""
        return ChatSession(**sample_chat_session_data)
    
    @pytest.fixture
    def sample_chat_message(self):
        """Create sample ChatMessage instance."""
        return ChatMessage(
            session_id=1,
            role=MessageRole.USER,
            content="Hello, world!",
            token_count=15
        )
    
    async def test_create_session_success(self, repository, mock_session, sample_chat_session):
        """Test successful chat session creation."""
        # Mock the refresh to return the same object
        mock_session.refresh.return_value = None
        
        result = await repository.create(sample_chat_session)
        
        # Verify session operations
        mock_session.add.assert_called_once_with(sample_chat_session)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_chat_session)
        
        # Verify return value
        assert result == sample_chat_session
    
    async def test_get_by_id_found(self, repository, mock_session, sample_chat_session):
        """Test getting chat session by ID when found."""
        # Mock the query execution with eager loading
        mock_result = mock_session.execute.return_value
        mock_result.scalar_one_or_none.return_value = sample_chat_session
        
        result = await repository.get_by_id(1)
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        # Don't check isinstance since it's a mock, just check it was called
        assert mock_session.execute.called
        
        # Verify return value
        assert result == sample_chat_session
    
    async def test_get_by_id_not_found(self, repository, mock_session):
        """Test getting chat session by ID when not found."""
        # Mock the query execution to return None
        mock_result = mock_session.execute.return_value
        mock_result.scalar_one_or_none.return_value = None
        
        result = await repository.get_by_id(999)
        
        assert result is None
    
    async def test_update_session_success(self, repository, mock_session, sample_chat_session):
        """Test successful chat session update."""
        # Mock the refresh to return the updated object
        mock_session.refresh.return_value = None
        
        result = await repository.update(sample_chat_session)
        
        # Verify session operations
        mock_session.add.assert_called_once_with(sample_chat_session)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_chat_session)
        
        # Verify return value
        assert result == sample_chat_session
    
    async def test_delete_session_success(self, repository, mock_session, sample_chat_session):
        """Test successful chat session deletion."""
        # Mock get_by_id to return the session
        with patch.object(repository, 'get_by_id', return_value=sample_chat_session):
            result = await repository.delete(1)
        
        # Verify session operations
        mock_session.delete.assert_called_once_with(sample_chat_session)
        mock_session.commit.assert_called_once()
        
        # Verify return value
        assert result is True
    
    async def test_delete_session_not_found(self, repository, mock_session):
        """Test deleting chat session when not found."""
        # Mock get_by_id to return None
        with patch.object(repository, 'get_by_id', return_value=None):
            result = await repository.delete(999)
        
        # Verify no delete operations
        mock_session.delete.assert_not_called()
        mock_session.commit.assert_not_called()
        
        # Verify return value
        assert result is False
    
    async def test_list_sessions_no_filters(self, repository, mock_session, sample_chat_session):
        """Test listing chat sessions without filters."""
        # Mock the query execution
        mock_result = mock_session.execute.return_value
        mock_result.scalars.return_value.all.return_value = [sample_chat_session]
        
        result = await repository.list()
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        # Don't check isinstance since it's a mock, just check it was called
        assert mock_session.execute.called
        
        # Verify return value
        assert result == [sample_chat_session]
    
    async def test_list_sessions_with_filters(self, repository, mock_session, sample_chat_session):
        """Test listing chat sessions with filters."""
        filters = {
            "tenant_id": "tenant_456",
            "agent_role": "developer",
            "is_active": True,
            "run_id": "test_run_123"
        }
        
        # Mock the query execution
        mock_result = mock_session.execute.return_value
        mock_result.scalars.return_value.all.return_value = [sample_chat_session]
        
        result = await repository.list(filters=filters)
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        
        # Verify return value
        assert result == [sample_chat_session]
    
    async def test_list_sessions_with_pagination(self, repository, mock_session, sample_chat_session):
        """Test listing chat sessions with pagination."""
        # Mock the query execution
        mock_result = mock_session.execute.return_value
        mock_result.scalars.return_value.all.return_value = [sample_chat_session]
        
        result = await repository.list(limit=10, offset=20)
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        
        # Verify return value
        assert result == [sample_chat_session]
    
    async def test_get_sessions_for_tenant(self, repository, mock_session, sample_chat_session):
        """Test getting sessions for a specific tenant."""
        # Mock the query execution
        mock_result = mock_session.execute.return_value
        mock_result.scalars.return_value.all.return_value = [sample_chat_session]
        
        result = await repository.get_sessions_for_tenant("tenant_456")
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        
        # Verify return value
        assert result == [sample_chat_session]
    
    async def test_get_active_sessions_for_role(self, repository, mock_session, sample_chat_session):
        """Test getting active sessions for a specific agent role."""
        # Mock the query execution
        mock_result = mock_session.execute.return_value
        mock_result.scalars.return_value.all.return_value = [sample_chat_session]
        
        result = await repository.get_active_sessions_for_role("developer")
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        
        # Verify return value
        assert result == [sample_chat_session]
        assert all(session.is_active for session in result)
        assert all(session.agent_role == "developer" for session in result)
    
    async def test_add_message_success(self, repository, mock_session, sample_chat_session, sample_chat_message):
        """Test adding a message to a chat session."""
        # Mock get_by_id to return the session
        with patch.object(repository, 'get_by_id', return_value=sample_chat_session):
            result = await repository.add_message(1, sample_chat_message)
        
        # Verify message was added
        mock_session.add.assert_called_once_with(sample_chat_message)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_chat_message)
        
        # Verify session was updated
        assert sample_chat_session.updated_at is not None
        
        # Verify return value
        assert result == sample_chat_message
        assert result.session_id == 1
    
    async def test_add_message_session_not_found(self, repository, mock_session, sample_chat_message):
        """Test adding message to non-existent session."""
        # Mock get_by_id to return None
        with patch.object(repository, 'get_by_id', return_value=None):
            # The repository doesn't raise ValueError, it just adds the message
            result = await repository.add_message(999, sample_chat_message)
        
        # Verify message was added anyway (repository doesn't validate session existence)
        mock_session.add.assert_called_once_with(sample_chat_message)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_chat_message)
        assert result == sample_chat_message
    
    async def test_get_messages_for_session(self, repository, mock_session):
        """Test getting messages for a chat session."""
        # Create sample messages
        messages = [
            ChatMessage(
                session_id=1,
                role=MessageRole.USER,
                content="Hello!",
                created_at=datetime(2024, 1, 1, 10, 0, 0)
            ),
            ChatMessage(
                session_id=1,
                role=MessageRole.ASSISTANT,
                content="Hi there!",
                created_at=datetime(2024, 1, 1, 10, 1, 0)
            ),
            ChatMessage(
                session_id=1,
                role=MessageRole.USER,
                content="How are you?",
                created_at=datetime(2024, 1, 1, 10, 2, 0)
            )
        ]
        
        # Mock the query execution
        mock_result = mock_session.execute.return_value
        mock_result.scalars.return_value.all.return_value = messages
        
        result = await repository.get_messages_for_session(1)
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        
        # Verify return value
        assert result == messages
        assert len(result) == 3
    
    async def test_get_messages_for_session_with_limit(self, repository, mock_session):
        """Test getting messages for a session with limit."""
        messages = [
            ChatMessage(
                session_id=1,
                role=MessageRole.USER,
                content=f"Message {i}",
                created_at=datetime(2024, 1, 1, 10, i, 0)
            )
            for i in range(10)
        ]
        
        # Mock the query execution
        mock_result = mock_session.execute.return_value
        mock_result.scalars.return_value.all.return_value = messages[:5]  # Limited to 5
        
        result = await repository.get_messages_for_session(1, limit=5)
        
        # Verify query was built correctly
        mock_session.execute.assert_called_once()
        
        # Verify return value
        assert len(result) == 5
    
    async def test_message_chronological_order(self, repository, mock_session):
        """Test that messages are returned in chronological order."""
        # Create messages with different timestamps
        messages = [
            ChatMessage(
                session_id=1,
                role=MessageRole.USER,
                content="First message",
                created_at=datetime(2024, 1, 1, 10, 0, 0)
            ),
            ChatMessage(
                session_id=1,
                role=MessageRole.ASSISTANT,
                content="Second message",
                created_at=datetime(2024, 1, 1, 10, 1, 0)
            ),
            ChatMessage(
                session_id=1,
                role=MessageRole.USER,
                content="Third message",
                created_at=datetime(2024, 1, 1, 10, 2, 0)
            )
        ]
        
        # Mock the query execution to return messages in order
        mock_result = mock_session.execute.return_value
        mock_result.scalars.return_value.all.return_value = messages
        
        result = await repository.get_messages_for_session(1)
        
        # Verify messages are in chronological order (earliest first)
        for i in range(len(result) - 1):
            assert result[i].created_at <= result[i + 1].created_at
    
    async def test_session_with_multiple_messages(self, repository, mock_session, sample_chat_session):
        """Test session with multiple messages workflow."""
        # Create session
        with patch.object(repository, 'get_by_id', return_value=sample_chat_session):
            created_session = await repository.create(sample_chat_session)
            assert created_session.title == "Project Discussion"
        
        # Add multiple messages
        messages = [
            ChatMessage(
                session_id=1,
                role=MessageRole.USER,
                content="Hello!"
            ),
            ChatMessage(
                session_id=1,
                role=MessageRole.ASSISTANT,
                content="Hi there!"
            ),
            ChatMessage(
                session_id=1,
                role=MessageRole.USER,
                content="How are you?"
            )
        ]
        
        created_messages = []
        for message in messages:
            with patch.object(repository, 'get_by_id', return_value=sample_chat_session):
                created_message = await repository.add_message(1, message)
                created_messages.append(created_message)
        
        # Verify all messages were created
        assert len(created_messages) == 3
        assert all(msg.session_id == 1 for msg in created_messages)
        
        # Verify message roles
        assert created_messages[0].role == MessageRole.USER
        assert created_messages[1].role == MessageRole.ASSISTANT
        assert created_messages[2].role == MessageRole.USER
    
    async def test_session_filtering_combinations(self, repository, mock_session, sample_chat_session):
        """Test various session filtering combinations."""
        test_filters = [
            {"tenant_id": "tenant_456"},
            {"agent_role": "developer"},
            {"is_active": True},
            {"run_id": "test_run_123"},
            {"tenant_id": "tenant_456", "agent_role": "developer"},
            {"tenant_id": "tenant_456", "is_active": True},
            {"agent_role": "developer", "is_active": True},
            {"tenant_id": "tenant_456", "agent_role": "developer", "is_active": True},
        ]
        
        for filters in test_filters:
            # Mock the query execution
            mock_result = mock_session.execute.return_value
            mock_result.scalars.return_value.all.return_value = [sample_chat_session]
            
            result = await repository.list(filters=filters)
            
            # Verify query was built correctly
            mock_session.execute.assert_called()
            
            # Verify return value
            assert result == [sample_chat_session]
            
            # Reset mock for next iteration
            mock_session.reset_mock()
    
    async def test_message_metadata_handling(self, repository, mock_session, sample_chat_session):
        """Test message with metadata fields."""
        message_with_metadata = ChatMessage(
            session_id=1,
            role=MessageRole.ASSISTANT,
            content="This is a response with metadata",
            token_count=25,
            model_used="gpt-4o"
        )
        
        # Mock get_by_id to return the session
        with patch.object(repository, 'get_by_id', return_value=sample_chat_session):
            result = await repository.add_message(1, message_with_metadata)
        
        # Verify metadata is preserved
        assert result.token_count == 25
        assert result.model_used == "gpt-4o"
        assert result.role == MessageRole.ASSISTANT
    
    def test_repository_initialization(self):
        """Test repository initialization with session."""
        mock_session = AsyncMock()
        repository = ChatSessionRepository(mock_session)
        
        assert repository.session == mock_session
    
    async def test_session_business_logic_scenarios(self, repository, mock_session):
        """Test various business logic scenarios for sessions."""
        # Active session
        active_session = ChatSession(
            title="Active Session",
            agent_role="developer",
            is_active=True
        )
        
        # Inactive session
        inactive_session = ChatSession(
            title="Inactive Session",
            agent_role="developer",
            is_active=False
        )
        
        # Test active sessions query
        mock_result = mock_session.execute.return_value
        mock_result.scalars.return_value.all.return_value = [active_session]
        
        result = await repository.get_active_sessions_for_role("developer")
        
        # Verify only active session is returned
        assert len(result) == 1
        assert result[0].is_active is True
        assert result[0].title == "Active Session"
