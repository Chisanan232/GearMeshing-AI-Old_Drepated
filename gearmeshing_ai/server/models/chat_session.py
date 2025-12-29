"""
Database models for chat session and message persistence.

This module defines SQLModel-based models for storing chat sessions
(chat rooms/zoom) and conversation history with AI agents.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlmodel import Column, DateTime, Field, SQLModel


class MessageRole(str, Enum):
    """Role of message sender in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatSessionBase(SQLModel):
    """Base fields for chat session."""

    title: str = Field(description="Chat session title")
    description: Optional[str] = Field(default=None, description="Chat session description")
    agent_role: str = Field(description="Agent role for this session (e.g., 'dev', 'planner')")
    tenant_id: Optional[str] = Field(default=None, index=True, description="Tenant identifier")
    run_id: Optional[str] = Field(default=None, index=True, description="Associated agent run ID")
    is_active: bool = Field(default=True, description="Whether session is active")


class ChatSession(ChatSessionBase, table=True):
    """Persistent chat session in database."""

    __tablename__ = "chat_sessions"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(
        default_factory=datetime.utcnow, sa_column=Column(DateTime, nullable=False), description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, sa_column=Column(DateTime, nullable=False), description="Last update timestamp"
    )


class ChatSessionRead(ChatSessionBase):
    """Schema for reading chat session."""

    id: int
    created_at: datetime
    updated_at: datetime


class ChatSessionCreate(ChatSessionBase):
    """Schema for creating chat session."""


class ChatSessionUpdate(SQLModel):
    """Schema for updating chat session."""

    title: Optional[str] = None
    description: Optional[str] = None
    agent_role: Optional[str] = None
    is_active: Optional[bool] = None


class ChatMessageBase(SQLModel):
    """Base fields for chat message."""

    session_id: int = Field(foreign_key="chat_sessions.id", description="Chat session ID")
    role: MessageRole = Field(description="Message sender role (user/assistant/system)")
    content: str = Field(description="Message content")
    message_metadata: Optional[str] = Field(default=None, description="JSON metadata (tokens, latency, etc.)")


class ChatMessage(ChatMessageBase, table=True):
    """Persistent chat message in database."""

    __tablename__ = "chat_messages"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(
        default_factory=datetime.utcnow, sa_column=Column(DateTime, nullable=False), description="Message timestamp"
    )


class ChatMessageRead(ChatMessageBase):
    """Schema for reading chat message."""

    id: int
    created_at: datetime


class ChatMessageCreate(ChatMessageBase):
    """Schema for creating chat message."""


class ChatHistoryRead(SQLModel):
    """Schema for reading full chat history."""

    session: ChatSessionRead
    messages: list[ChatMessageRead]
