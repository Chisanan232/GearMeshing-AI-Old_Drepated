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

from ..entities.chat_sessions import ChatSession, ChatMessage
from .base import BaseRepository, QueryBuilder


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
        await self.session.commit()
        await self.session.refresh(session)
        return session
    
    async def get_by_id(self, session_id: str | int) -> Optional[ChatSession]:
        """Get chat session by its ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            ChatSession instance or None
        """
        stmt = select(ChatSession).where(ChatSession.id == session_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def update(self, session: ChatSession) -> ChatSession:
        """Update an existing chat session.
        
        Args:
            session: ChatSession instance with updated fields
            
        Returns:
            Updated ChatSession instance
        """
        session.updated_at = datetime.utcnow()
        self.session.add(session)
        await self.session.commit()
        await self.session.refresh(session)
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
            await self.session.delete(session)
            await self.session.commit()
            return True
        return False
    
    async def list(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ChatSession]:
        """List chat sessions with optional pagination and filtering.
        
        Args:
            limit: Maximum records to return
            offset: Records to skip
            filters: Field filters (tenant_id, agent_role, is_active, run_id)
            
        Returns:
            List of ChatSession instances
        """
        stmt = select(ChatSession).order_by(ChatSession.updated_at.desc())
        
        if filters:
            stmt = QueryBuilder.apply_filters(stmt, ChatSession, filters)
        
        stmt = QueryBuilder.apply_pagination(stmt, limit, offset)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
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
            .order_by(ChatSession.updated_at.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_active_sessions_for_role(self, agent_role: str) -> List[ChatSession]:
        """Get all active sessions for a specific agent role.
        
        Args:
            agent_role: Agent role identifier
            
        Returns:
            List of active ChatSession instances
        """
        stmt = (
            select(ChatSession)
            .where(
                (ChatSession.agent_role == agent_role) & (ChatSession.is_active == True)
            )
            .order_by(ChatSession.updated_at.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
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
        
        await self.session.commit()
        await self.session.refresh(message)
        return message
    
    async def get_messages_for_session(
        self, 
        session_id: int, 
        limit: Optional[int] = None
    ) -> List[ChatMessage]:
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
            .order_by(ChatMessage.created_at.asc())
        )
        
        if limit:
            stmt = stmt.limit(limit)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
