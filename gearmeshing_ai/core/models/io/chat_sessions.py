"""
Chat session I/O models for API requests and responses.

This module contains Pydantic-based I/O schemas for chat session and message
API endpoints. These models define the contract between the API and clients
for managing chat sessions and messages.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class MessageRole(str, Enum):
    """Role of message sender in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatSessionRead(BaseModel):
    """Schema for reading chat session from API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str = Field(description="Chat session title")
    description: Optional[str] = Field(default=None, description="Chat session description")
    agent_role: str = Field(description="Agent role for this session")
    tenant_id: Optional[str] = Field(default=None, description="Tenant identifier")
    run_id: Optional[str] = Field(default=None, description="Associated agent run ID")
    is_active: bool = Field(description="Whether session is active")
    created_at: datetime
    updated_at: datetime


class ChatSessionCreate(BaseModel):
    """Schema for creating chat session via API."""

    title: str = Field(description="Chat session title")
    description: Optional[str] = Field(default=None, description="Chat session description")
    agent_role: str = Field(description="Agent role for this session")
    tenant_id: Optional[str] = Field(default=None, description="Tenant identifier")
    run_id: Optional[str] = Field(default=None, description="Associated agent run ID")
    is_active: bool = Field(default=True, description="Whether session is active")


class ChatSessionUpdate(BaseModel):
    """Schema for updating chat session via API."""

    title: Optional[str] = None
    description: Optional[str] = None
    agent_role: Optional[str] = None
    is_active: Optional[bool] = None


class ChatMessageRead(BaseModel):
    """Schema for reading chat message from API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: int = Field(description="Chat session ID")
    role: MessageRole = Field(description="Message sender role")
    content: str = Field(description="Message content")
    token_count: Optional[int] = Field(default=None, description="Token count for this message")
    model_used: Optional[str] = Field(default=None, description="Model used for assistant messages")
    created_at: datetime


class ChatMessageCreate(BaseModel):
    """Schema for creating chat message via API."""

    session_id: int = Field(description="Chat session ID")
    role: MessageRole = Field(description="Message sender role")
    content: str = Field(description="Message content")
    token_count: Optional[int] = Field(default=None, description="Token count for this message")
    model_used: Optional[str] = Field(default=None, description="Model used for assistant messages")


class ChatHistoryRead(BaseModel):
    """Schema for reading full chat history."""

    session: ChatSessionRead
    messages: List[ChatMessageRead]
