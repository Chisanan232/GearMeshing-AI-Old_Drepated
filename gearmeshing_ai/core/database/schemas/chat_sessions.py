"""
Schema models for chat session and message API requests and responses.

These schemas are used for API serialization/deserialization and are separate
from the entity models to allow independent evolution of API contracts.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role of message sender in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatSessionBase(BaseModel):
    """Base fields for chat session schema."""

    title: str = Field(description="Chat session title")
    description: Optional[str] = Field(default=None, description="Chat session description")
    agent_role: str = Field(description="Agent role for this session (e.g., 'dev', 'planner')")
    tenant_id: Optional[str] = Field(default=None, description="Tenant identifier")
    run_id: Optional[str] = Field(default=None, description="Associated agent run ID")
    is_active: bool = Field(default=True, description="Whether session is active")


class ChatSessionRead(ChatSessionBase):
    """Schema for reading chat session."""

    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ChatSessionCreate(ChatSessionBase):
    """Schema for creating chat session."""

    pass


class ChatSessionUpdate(BaseModel):
    """Schema for updating chat session."""

    title: Optional[str] = None
    description: Optional[str] = None
    agent_role: Optional[str] = None
    is_active: Optional[bool] = None


class ChatMessageBase(BaseModel):
    """Base fields for chat message schema."""

    session_id: int = Field(description="Chat session ID")
    role: MessageRole = Field(description="Message sender role (user/assistant/system)")
    content: str = Field(description="Message content")
    message_metadata: Optional[str] = Field(default=None, description="JSON metadata (tokens, latency, etc.)")


class ChatMessageRead(ChatMessageBase):
    """Schema for reading chat message."""

    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class ChatMessageCreate(ChatMessageBase):
    """Schema for creating chat message."""

    pass


class ChatHistoryRead(BaseModel):
    """Schema for reading full chat history."""

    session: ChatSessionRead
    messages: list[ChatMessageRead]
