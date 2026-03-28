"""
Service for persisting chat history from agent runs.

This service handles automatic persistence of chat messages derived from SSE events
and user interactions, storing them in the database for later retrieval.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from gearmeshing_ai.core.database.entities.chat_sessions import (
    ChatMessage,
    ChatSession,
    MessageRole,
)

logger = logging.getLogger(__name__)


class ChatPersistenceService:
    """Service for persisting chat messages from agent runs."""

    def __init__(self, session: AsyncSession):
        """Initialize chat persistence service with database session."""
        self.session = session

    async def get_or_create_session(
        self,
        run_id: str,
        tenant_id: str,
        agent_role: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ChatSession:
        """
        Get existing chat session for a run or create a new one.

        Args:
            run_id: The agent run ID
            tenant_id: The tenant identifier
            agent_role: The agent role (e.g., 'planner', 'dev')
            title: Optional session title
            description: Optional session description

        Returns:
            ChatSession object (existing or newly created)
        """
        # Check if session already exists
        statement = select(ChatSession).where(ChatSession.run_id == run_id)
        result = await self.session.execute(statement)
        existing_session = result.scalars().first()

        if existing_session:
            logger.debug(f"Found existing chat session {existing_session.id} for run {run_id}")
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
        self.session.add(new_session)
        await self.session.commit()
        await self.session.refresh(new_session)
        logger.info(f"Created new chat session {new_session.id} for run {run_id}")
        return new_session

    async def add_user_message(
        self,
        session_id: int,
        content: str,
        metadata: Optional[dict] = None,
    ) -> ChatMessage:
        """
        Add a user message to the chat session.

        Args:
            session_id: The chat session ID
            content: The message content
            metadata: Optional metadata (e.g., role tags, objective)

        Returns:
            ChatMessage object
        """
        message = ChatMessage(
            session_id=session_id,
            role=MessageRole.USER,
            content=content,
            message_metadata=json.dumps(metadata) if metadata else None,
        )
        self.session.add(message)
        await self.session.commit()
        await self.session.refresh(message)
        logger.debug(f"Added user message {message.id} to session {session_id}")
        return message

    async def add_agent_message(
        self,
        session_id: int,
        content: str,
        event_type: Optional[str] = None,
        event_category: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ChatMessage:
        """
        Add an agent message to the chat session.

        Args:
            session_id: The chat session ID
            content: The message content (can be HTML formatted)
            event_type: The SSE event type (e.g., 'thought_executed', 'capability_executed')
            event_category: The event category (e.g., 'thinking', 'operation')
            metadata: Optional metadata from the event

        Returns:
            ChatMessage object
        """
        # Combine metadata with event information
        full_metadata = metadata or {}
        full_metadata["event_type"] = event_type
        full_metadata["event_category"] = event_category

        message = ChatMessage(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=content,
            message_metadata=json.dumps(full_metadata) if full_metadata else None,
        )
        self.session.add(message)
        await self.session.commit()
        await self.session.refresh(message)
        logger.debug(f"Added agent message {message.id} to session {session_id}")
        return message
