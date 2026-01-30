"""
Chat session repository interface and implementation.

This module provides data access operations for chat session and message
management, including conversation history and session metadata.
Built exclusively on SQLModel for type-safe ORM operations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from ..entities.chat_sessions import ChatMessage, ChatSession
from .base import BaseRepository, QueryBuilder, _utc_now_naive


class ChatSessionRepository(BaseRepository[ChatSession]):
    """Repository for chat session data access operations using SQLModel."""

    def __init__(self, session: Session) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLModel Session for database operations
        """
        super().__init__(session, ChatSession)

    async def create(self, session: ChatSession) -> ChatSession:
        """Create a new chat session.

        Args:
            session: ChatSession SQLModel instance

        Returns:
            Persisted ChatSession with generated fields
        """
        self.session.add(session)
        self.session.commit()
        self.session.refresh(session)
        return session

    async def get_by_id(self, session_id: str | int) -> Optional[ChatSession]:
        """Get chat session by its ID.

        Args:
            session_id: Session ID

        Returns:
            ChatSession instance or None
        """
        stmt = select(ChatSession).where(ChatSession.id == session_id)
        result = self.session.exec(stmt)
        return result.one_or_none()

    async def update(self, session: ChatSession) -> ChatSession:
        """Update an existing chat session.

        Args:
            session: ChatSession instance with updated fields

        Returns:
            Updated ChatSession instance
        """
        session.updated_at = _utc_now_naive()
        self.session.add(session)
        self.session.commit()
        self.session.refresh(session)
        return session

    async def delete(self, session_id: str | int) -> bool:
        """Delete chat session by its ID (cascades to messages).

        Args:
            session_id: Session ID to delete

        Returns:
            True if deleted, False if not found
        """
        session = await self.get_by_id(session_id)
        if session:
            self.session.delete(session)
            self.session.commit()
            return True
        return False

    async def list(
        self, limit: Optional[int] = None, offset: Optional[int] = None, filters: Optional[Dict[str, Any]] = None
    ) -> List[ChatSession]:
        """List chat sessions with optional pagination and filtering.

        Args:
            limit: Maximum records to return
            offset: Records to skip
            filters: Field filters (tenant_id, agent_role, is_active, run_id)

        Returns:
            List of ChatSession instances
        """
        stmt = select(ChatSession).order_by(ChatSession.updated_at.desc())  # type: ignore

        if filters:
            stmt = QueryBuilder.apply_filters(stmt, ChatSession, filters)

        stmt = QueryBuilder.apply_pagination(stmt, limit, offset)

        result = self.session.exec(stmt)
        return list(result)

    async def get_sessions_for_tenant(self, tenant_id: str) -> List[ChatSession]:
        """Get all sessions for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of ChatSession instances for tenant
        """
        stmt = (
            select(ChatSession)
            .where(ChatSession.tenant_id == tenant_id)
            .order_by(ChatSession.updated_at.desc())  # type: ignore
        )
        result = self.session.exec(stmt)
        return list(result)

    async def get_active_sessions_for_role(self, agent_role: str) -> List[ChatSession]:
        """Get all active sessions for a specific agent role.

        Args:
            agent_role: Agent role identifier

        Returns:
            List of active ChatSession instances
        """
        stmt = (
            select(ChatSession)
            .where((ChatSession.agent_role == agent_role) & (ChatSession.is_active == True))
            .order_by(ChatSession.updated_at.desc())  # type: ignore
        )
        result = self.session.exec(stmt)
        return list(result)

    async def add_message(self, session_id: int, message: ChatMessage) -> ChatMessage:
        """Add a message to a chat session.

        Args:
            session_id: Chat session ID
            message: ChatMessage instance to add

        Returns:
            Persisted ChatMessage instance
        """
        message.session_id = session_id
        self.session.add(message)

        # Update session timestamp
        session = await self.get_by_id(session_id)
        if session:
            session.updated_at = datetime.utcnow()

        self.session.commit()
        self.session.refresh(message)
        return message

    async def get_messages_for_session(self, session_id: int, limit: Optional[int] = None) -> List[ChatMessage]:
        """Get messages for a chat session.

        Args:
            session_id: Chat session ID
            limit: Maximum messages to return

        Returns:
            List of ChatMessage instances in chronological order
        """
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())  # type: ignore
        )

        if limit:
            stmt = stmt.limit(limit)

        result = self.session.exec(stmt)
        return list(result)

    # Chat persistence service methods
    async def get_or_create_session(
        self,
        run_id: str,
        tenant_id: str,
        agent_role: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ChatSession:
        """Get existing chat session for a run or create a new one."""
        # Check if session already exists
        stmt = select(ChatSession).where(ChatSession.run_id == run_id)
        result = await self.session.exec(stmt)
        existing_session = result.one_or_none()

        if existing_session:
            return existing_session

        # Create new session
        new_session = ChatSession(
            run_id=run_id,
            tenant_id=tenant_id,
            agent_role=agent_role,
            title=title or f"Chat - {agent_role}",
            description=description or f"Chat history for run {run_id}",
            is_active=True,
        )
        return await self.create(new_session)

    async def add_user_message(
        self,
        session_id: int,
        content: str,
        metadata: Optional[dict] = None,
    ) -> ChatMessage:
        """Add a user message to the chat session."""
        import json

        message = ChatMessage(
            session_id=session_id,
            role="user",
            content=content,
            message_metadata=json.dumps(metadata) if metadata else None,
        )

        # Add message and update session timestamp
        await self.session.add(message)

        session = await self.get_by_id(session_id)
        if session:
            session.updated_at = _utc_now_naive()

        await self.session.commit()
        await self.session.refresh(message)
        return message

    async def add_agent_message(
        self,
        session_id: int,
        content: str,
        event_type: Optional[str] = None,
        event_category: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ChatMessage:
        """Add an agent message to the chat session."""
        import json

        # Combine metadata with event information
        full_metadata = metadata or {}
        full_metadata["event_type"] = event_type
        full_metadata["event_category"] = event_category

        message = ChatMessage(
            session_id=session_id,
            role="assistant",
            content=content,
            message_metadata=json.dumps(full_metadata) if full_metadata else None,
        )

        # Add message and update session timestamp
        await self.session.add(message)

        session = await self.get_by_id(session_id)
        if session:
            session.updated_at = _utc_now_naive()

        await self.session.commit()
        await self.session.refresh(message)
        return message
