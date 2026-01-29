"""
Unit tests for chat session models and data structures.

Tests cover:
- Chat session model creation and validation
- Chat message model creation and validation
- Chat history data structures
- Message role enumeration
"""

from __future__ import annotations

from datetime import datetime

from gearmeshing_ai.core.database.entities.chat_sessions import (
    ChatMessage,
    ChatSession,
    MessageRole,
)
from gearmeshing_ai.core.database.schemas.chat_sessions import (
    ChatMessageCreate,
    ChatSessionCreate,
    ChatSessionUpdate,
)


class TestChatSessionModels:
    """Tests for chat session model creation and validation."""

    def test_create_chat_session_model(self):
        """Test creating a ChatSessionCreate model."""
        session_data = ChatSessionCreate(
            title="Test Session",
            description="Test Description",
            agent_role="planner",
            tenant_id="tenant-1",
        )

        assert session_data.title == "Test Session"
        assert session_data.description == "Test Description"
        assert session_data.agent_role == "planner"
        assert session_data.tenant_id == "tenant-1"

    def test_chat_session_with_run_id(self):
        """Test ChatSession model with run_id."""
        session = ChatSession(
            id=1,
            title="Session 1",
            agent_role="planner",
            tenant_id="tenant-1",
            run_id="run-123",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        assert session.id == 1
        assert session.run_id == "run-123"
        assert session.agent_role == "planner"
        assert session.tenant_id == "tenant-1"

    def test_chat_session_update_model(self):
        """Test ChatSessionUpdate model."""
        update_data = ChatSessionUpdate(title="New Title")

        assert update_data.title == "New Title"

    def test_chat_session_without_run_id(self):
        """Test ChatSession model without run_id."""
        session = ChatSession(
            id=2,
            title="Session 2",
            agent_role="dev",
            tenant_id="tenant-1",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        assert session.id == 2
        assert session.run_id is None
        assert session.agent_role == "dev"

    def test_chat_session_is_active_default(self):
        """Test ChatSession is_active defaults to True."""
        session = ChatSession(
            id=3,
            title="Session 3",
            agent_role="executor",
            tenant_id="tenant-1",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        assert session.is_active is True

    def test_chat_session_with_metadata(self):
        """Test ChatSession with optional metadata."""
        session = ChatSession(
            id=4,
            title="Session 4",
            agent_role="planner",
            tenant_id="tenant-1",
            run_id="run-456",
            is_active=False,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        assert session.is_active is False
        assert session.run_id == "run-456"


class TestChatMessageModels:
    """Tests for chat message model creation and validation."""

    def test_create_user_message(self):
        """Test creating a user message."""
        message_data = ChatMessageCreate(
            session_id=1,
            role=MessageRole.USER,
            content="Hello agent!",
        )

        assert message_data.session_id == 1
        assert message_data.role == MessageRole.USER
        assert message_data.content == "Hello agent!"

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        message_data = ChatMessageCreate(
            session_id=1,
            role=MessageRole.ASSISTANT,
            content="Hello user!",
        )

        assert message_data.session_id == 1
        assert message_data.role == MessageRole.ASSISTANT
        assert message_data.content == "Hello user!"

    def test_create_system_message(self):
        """Test creating a system message."""
        message_data = ChatMessageCreate(
            session_id=1,
            role=MessageRole.SYSTEM,
            content="System: Run completed",
        )

        assert message_data.session_id == 1
        assert message_data.role == MessageRole.SYSTEM
        assert message_data.content == "System: Run completed"

    def test_chat_message_with_metadata(self):
        """Test ChatMessage with metadata."""
        message = ChatMessage(
            id=1,
            session_id=1,
            role=MessageRole.ASSISTANT,
            content="Operation completed",
            message_metadata='{"event_type": "capability_executed", "status": "success"}',
            created_at=datetime.utcnow(),
        )

        assert message.id == 1
        assert message.session_id == 1
        assert message.role == MessageRole.ASSISTANT
        assert message.message_metadata is not None
        assert "event_type" in message.message_metadata

    def test_chat_message_without_metadata(self):
        """Test ChatMessage without metadata."""
        message = ChatMessage(
            id=2,
            session_id=1,
            role=MessageRole.USER,
            content="Hello",
            created_at=datetime.utcnow(),
        )

        assert message.id == 2
        assert message.message_metadata is None

    def test_message_role_enumeration(self):
        """Test MessageRole enumeration values."""
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.SYSTEM.value == "system"

    def test_multiple_messages_in_session(self):
        """Test creating multiple messages for a session."""
        messages = [
            ChatMessage(
                id=1,
                session_id=1,
                role=MessageRole.USER,
                content="First message",
                created_at=datetime.utcnow(),
            ),
            ChatMessage(
                id=2,
                session_id=1,
                role=MessageRole.ASSISTANT,
                content="Response to first",
                created_at=datetime.utcnow(),
            ),
            ChatMessage(
                id=3,
                session_id=1,
                role=MessageRole.USER,
                content="Second message",
                created_at=datetime.utcnow(),
            ),
        ]

        assert len(messages) == 3
        assert all(msg.session_id == 1 for msg in messages)
        assert messages[0].role == MessageRole.USER
        assert messages[1].role == MessageRole.ASSISTANT
        assert messages[2].role == MessageRole.USER


class TestChatSessionDataIntegrity:
    """Tests for chat session data integrity and relationships."""

    def test_session_and_messages_relationship(self):
        """Test relationship between session and messages."""
        session = ChatSession(
            id=1,
            title="Test Session",
            agent_role="planner",
            tenant_id="tenant-1",
            run_id="run-123",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        messages = [
            ChatMessage(
                id=1,
                session_id=session.id,
                role=MessageRole.USER,
                content="Message 1",
                created_at=datetime.utcnow(),
            ),
            ChatMessage(
                id=2,
                session_id=session.id,
                role=MessageRole.ASSISTANT,
                content="Message 2",
                created_at=datetime.utcnow(),
            ),
        ]

        assert all(msg.session_id == session.id for msg in messages)
        assert len(messages) == 2

    def test_run_id_persistence_across_sessions(self):
        """Test that run_id is properly maintained across sessions."""
        run_id = "run-789"

        session1 = ChatSession(
            id=1,
            title="Session 1",
            agent_role="planner",
            tenant_id="tenant-1",
            run_id=run_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        session2 = ChatSession(
            id=2,
            title="Session 2",
            agent_role="executor",
            tenant_id="tenant-1",
            run_id=run_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        assert session1.run_id == run_id
        assert session2.run_id == run_id
        assert session1.run_id == session2.run_id

    def test_tenant_isolation(self):
        """Test that sessions are properly isolated by tenant."""
        session1 = ChatSession(
            id=1,
            title="Session 1",
            agent_role="planner",
            tenant_id="tenant-1",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        session2 = ChatSession(
            id=2,
            title="Session 2",
            agent_role="planner",
            tenant_id="tenant-2",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        assert session1.tenant_id != session2.tenant_id
        assert session1.tenant_id == "tenant-1"
        assert session2.tenant_id == "tenant-2"

    def test_message_content_preservation(self):
        """Test that message content is preserved correctly."""
        content = "This is a test message with special characters: !@#$%^&*()"
        message = ChatMessage(
            id=1,
            session_id=1,
            role=MessageRole.USER,
            content=content,
            created_at=datetime.utcnow(),
        )

        assert message.content == content
        assert len(message.content) > 0

    def test_long_message_content(self):
        """Test handling of long message content."""
        long_content = "A" * 5000  # 5000 character message
        message = ChatMessage(
            id=1,
            session_id=1,
            role=MessageRole.ASSISTANT,
            content=long_content,
            created_at=datetime.utcnow(),
        )

        assert len(message.content) == 5000
        assert message.content == long_content
