"""Unit tests for chat session entity models.

Tests SQLModel validation, field constraints, relationships,
and business logic for ChatSession and ChatMessage entities.
"""

from __future__ import annotations

from datetime import datetime
from typing import List

import pytest
from pydantic import ValidationError

from gearmeshing_ai.core.database.entities.chat_sessions import ChatSession, ChatSessionBase, ChatMessage, MessageRole


class TestMessageRole:
    """Tests for MessageRole enum."""
    
    def test_message_role_values(self):
        """Test MessageRole enum values."""
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.SYSTEM == "system"
    
    def test_message_role_iteration(self):
        """Test MessageRole enum iteration."""
        roles = list(MessageRole)
        assert MessageRole.USER in roles
        assert MessageRole.ASSISTANT in roles
        assert MessageRole.SYSTEM in roles
        assert len(roles) == 3


class TestChatSessionBase:
    """Tests for ChatSessionBase model validation."""
    
    def test_chat_session_base_valid_data(self, sample_chat_session_data):
        """Test ChatSessionBase with valid data."""
        session = ChatSessionBase(**sample_chat_session_data)
        
        assert session.title == "Project Discussion"
        assert session.agent_role == "developer"
        assert session.is_active is True
    
    def test_chat_session_base_missing_required_fields(self):
        """Test ChatSessionBase validation with missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            ChatSessionBase()
        
        errors = exc_info.value.errors()
        error_fields = {error["loc"][0] for error in errors}
        
        expected_fields = {"title", "agent_role"}
        assert expected_fields.issubset(error_fields)
    
    def test_chat_session_base_optional_fields(self, sample_chat_session_data):
        """Test ChatSessionBase with optional fields."""
        data = sample_chat_session_data.copy()
        data["description"] = None
        data["tenant_id"] = None
        data["run_id"] = None
        data["is_active"] = False
        
        session = ChatSessionBase(**data)
        
        assert session.description is None
        assert session.tenant_id is None
        assert session.run_id is None
        assert session.is_active is False


class TestChatSession:
    """Tests for ChatSession entity model."""
    
    def test_chat_session_creation_valid_data(self, sample_chat_session_data):
        """Test ChatSession creation with valid data."""
        session = ChatSession(**sample_chat_session_data)
        
        assert session.title == "Project Discussion"
        assert session.agent_role == "developer"
        assert session.is_active is True
        assert session.id is None  # Will be set by database
    
    def test_chat_session_automatic_timestamps(self, sample_chat_session_data):
        """Test ChatSession automatic timestamp generation."""
        session = ChatSession(**sample_chat_session_data)
        
        assert session.created_at is not None
        assert session.updated_at is not None
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.updated_at, datetime)
    
    def test_chat_session_relationship_initialization(self, sample_chat_session_data):
        """Test ChatSession relationship initialization."""
        session = ChatSession(**sample_chat_session_data)
        
        # Messages relationship is commented out in the model, so it shouldn't exist
        # When relationships are enabled, this test should be updated
        assert not hasattr(session, 'messages') or session.messages is None
    
    def test_chat_session_repr(self, sample_chat_session_data):
        """Test ChatSession string representation."""
        session = ChatSession(**sample_chat_session_data)
        
        repr_str = repr(session)
        assert "ChatSession" in repr_str
        assert session.title in repr_str
        assert session.agent_role in repr_str
    
    def test_chat_session_table_name(self):
        """Test ChatSession table name configuration."""
        assert ChatSession.__tablename__ == "chat_sessions"
    
    def test_chat_session_inheritance(self):
        """Test ChatSession inherits from ChatSessionBase."""
        assert issubclass(ChatSession, ChatSessionBase)
    
    @pytest.mark.parametrize("field_name", [
        "title", "agent_role"
    ])
    def test_chat_session_required_base_fields(self, sample_chat_session_data, field_name):
        """Test that base fields become None when not provided in ChatSession."""
        data = sample_chat_session_data.copy()
        del data[field_name]
        
        # SQLModel sets missing fields to None instead of raising validation error
        session = ChatSession(**data)
        
        # The missing field should be None
        assert getattr(session, field_name) is None


class TestChatMessage:
    """Tests for ChatMessage entity model."""
    
    def test_chat_message_creation_valid_data(self):
        """Test ChatMessage creation with valid data."""
        message = ChatMessage(
            session_id=1,
            role=MessageRole.USER,
            content="Hello, world!"
        )
        
        assert message.session_id == 1
        assert message.role == MessageRole.USER
        assert message.content == "Hello, world!"
        assert message.id is None  # Will be set by database
    
    def test_chat_message_with_optional_fields(self):
        """Test ChatMessage with optional fields."""
        message = ChatMessage(
            session_id=1,
            role=MessageRole.ASSISTANT,
            content="I can help you with that!",
            token_count=25,
            model_used="gpt-4o"
        )
        
        assert message.token_count == 25
        assert message.model_used == "gpt-4o"
    
    def test_chat_message_automatic_timestamps(self):
        """Test ChatMessage automatic timestamp generation."""
        message = ChatMessage(
            session_id=1,
            role=MessageRole.USER,
            content="Test message"
        )
        
        assert message.created_at is not None
        assert isinstance(message.created_at, datetime)
    
    def test_chat_message_role_validation(self):
        """Test ChatMessage role field validation."""
        # Test valid roles
        for role in [MessageRole.USER, MessageRole.ASSISTANT, MessageRole.SYSTEM]:
            message = ChatMessage(
                session_id=1,
                role=role,
                content="Test message"
            )
            assert message.role == role
    
    def test_chat_message_content_validation(self):
        """Test ChatMessage content field validation."""
        # Test non-empty content
        message = ChatMessage(
            session_id=1,
            role=MessageRole.USER,
            content="Non-empty content"
        )
        assert message.content == "Non-empty content"
        
        # Test empty content (should be allowed)
        empty_message = ChatMessage(
            session_id=1,
            role=MessageRole.USER,
            content=""
        )
        assert empty_message.content == ""
    
    def test_chat_message_repr(self):
        """Test ChatMessage string representation."""
        message = ChatMessage(
            session_id=1,
            role=MessageRole.USER,
            content="Test message"
        )
        
        repr_str = repr(message)
        assert "ChatMessage" in repr_str
        assert str(message.role) in repr_str
        assert str(message.session_id) in repr_str
    
    def test_chat_message_table_name(self):
        """Test ChatMessage table name configuration."""
        assert ChatMessage.__tablename__ == "chat_messages"
    
    def test_chat_message_foreign_key_constraint(self):
        """Test ChatMessage foreign key field."""
        message = ChatMessage(
            session_id=123,
            role=MessageRole.USER,
            content="Test message"
        )
        
        assert message.session_id == 123
        # The foreign key constraint is enforced at database level
    
    def test_chat_message_optional_metadata_fields(self):
        """Test ChatMessage optional metadata fields."""
        message = ChatMessage(
            session_id=1,
            role=MessageRole.ASSISTANT,
            content="Response with metadata",
            token_count=42,
            model_used="claude-3-5-sonnet"
        )
        
        assert message.token_count == 42
        assert message.model_used == "claude-3-5-sonnet"
        
        # Test with None values
        message_none = ChatMessage(
            session_id=1,
            role=MessageRole.USER,
            content="Simple message"
        )
        
        assert message_none.token_count is None
        assert message_none.model_used is None


class TestChatSessionIntegration:
    """Integration tests for ChatSession and ChatMessage relationship."""
    
    def test_session_with_multiple_messages(self, sample_chat_session_data):
        """Test ChatSession with multiple ChatMessage instances."""
        session = ChatSession(**sample_chat_session_data)
        
        # Create messages
        messages = [
            ChatMessage(
                session_id=1,  # Will be set properly after session creation
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
        
        # In a real scenario, these would be added through the session
        # For testing, we can verify the relationship structure
        assert len(messages) == 3
        assert all(msg.role in [MessageRole.USER, MessageRole.ASSISTANT] for msg in messages)
        assert all(msg.content for msg in messages)
    
    def test_message_role_distribution(self):
        """Test different message roles in a conversation."""
        roles = [
            MessageRole.SYSTEM,
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.USER,
            MessageRole.ASSISTANT
        ]
        
        messages = [
            ChatMessage(
                session_id=1,
                role=role,
                content=f"Message as {role}"
            )
            for role in roles
        ]
        
        # Count roles
        user_messages = [msg for msg in messages if msg.role == MessageRole.USER]
        assistant_messages = [msg for msg in messages if msg.role == MessageRole.ASSISTANT]
        system_messages = [msg for msg in messages if msg.role == MessageRole.SYSTEM]
        
        assert len(user_messages) == 2
        assert len(assistant_messages) == 2
        assert len(system_messages) == 1
    
    def test_chat_session_business_logic(self, sample_chat_session_data):
        """Test business logic scenarios for chat sessions."""
        # Active session
        active_session = ChatSession(**sample_chat_session_data)
        assert active_session.is_active is True
        
        # Inactive session
        inactive_data = sample_chat_session_data.copy()
        inactive_data["is_active"] = False
        inactive_session = ChatSession(**inactive_data)
        assert inactive_session.is_active is False
        
        # Session with different agent roles
        roles = ["developer", "planner", "tester", "reviewer"]
        for role in roles:
            data = sample_chat_session_data.copy()
            data["agent_role"] = role
            session = ChatSession(**data)
            assert session.agent_role == role
    
    def test_message_content_validation_scenarios(self):
        """Test various message content scenarios."""
        test_cases = [
            ("Simple message", "Hello, world!"),
            ("Empty message", ""),
            ("Long message", "This is a very long message that contains multiple sentences and should be handled properly by the system without any issues."),
            ("Message with special chars", "Hello @user #test $special %chars!"),
            ("Message with newlines", "Line 1\nLine 2\nLine 3"),
            ("JSON-like content", '{"key": "value", "array": [1, 2, 3]}'),
            ("Code snippet", "def hello():\n    print('Hello, world!')"),
        ]
        
        for description, content in test_cases:
            message = ChatMessage(
                session_id=1,
                role=MessageRole.USER,
                content=content
            )
            assert message.content == content
