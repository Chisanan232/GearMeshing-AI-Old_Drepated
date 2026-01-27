"""
Chat session entity models.

This module contains the database entities for chat session and message
persistence. Chat sessions represent conversations with AI agents,
including message history and metadata.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from sqlmodel import Field, Relationship, SQLModel

from ..base import Base


class MessageRole(str, Enum):
    """Role of message sender in conversation."""
    
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatSessionBase(Base):
    """Base fields for chat session."""
    
    title: str = Field(description="Chat session title")
    description: Optional[str] = Field(default=None, description="Chat session description")
    agent_role: str = Field(description="Agent role for this session")
    tenant_id: Optional[str] = Field(default=None, description="Tenant identifier")
    run_id: Optional[str] = Field(default=None, description="Associated agent run ID")
    is_active: bool = Field(default=True, description="Whether session is active")


class ChatSession(ChatSessionBase, table=True):
    """Persistent chat session in database.
    
    Represents a conversation session with an AI agent,
    including metadata and associated message history.
    
    Table: chat_sessions
    """
    
    __tablename__ = "chat_sessions"
    __table_args__ = ({"extend_existing": True},)
    
    # Primary key
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"onupdate": datetime.utcnow})
    
    # Relationships
    # messages: List["ChatMessage"] = Relationship(
    #     back_populates="session",
    #     sa_relationship_kwargs={"cascade": "all, delete-orphan", "lazy": "selectin"}
    # )
    
    def __repr__(self) -> str:
        return f"ChatSession(id={self.id}, title={self.title}, role={self.agent_role})"


class ChatMessage(Base, table=True):
    """Individual message within a chat session.
    
    Stores individual messages in the conversation history,
    including content, role, and metadata.
    
    Table: chat_messages
    """
    
    __tablename__ = "chat_messages"
    __table_args__ = ({"extend_existing": True},)
    
    # Primary key
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Foreign key to session
    session_id: int = Field(foreign_key="chat_sessions.id", index=True)
    
    # Message content
    role: MessageRole = Field(description="Message sender role")
    content: str = Field(description="Message content")
    
    # Metadata
    token_count: Optional[int] = Field(default=None, description="Token count for this message")
    model_used: Optional[str] = Field(default=None, description="Model used for assistant messages")
    
    # Timestamp
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    
    # Relationships
    # session: ChatSession = Relationship(back_populates="messages")
    
    def __repr__(self) -> str:
        return f"ChatMessage(id={self.id}, role={self.role}, session_id={self.session_id})"
