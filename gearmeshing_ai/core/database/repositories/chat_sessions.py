"""
Chat session repository interface and implementation.

This module provides data access operations for chat session and message
management, including conversation history and session metadata.
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import Select, and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..entities.chat_sessions import ChatSession, ChatMessage
from .base import BaseRepository


class ChatSessionRepository(BaseRepository[ChatSession]):
    """Repository for chat session data access operations."""
    
    async def create(self, session: ChatSession) -> ChatSession:
        """Create a new chat session."""
        self.session.add(session)
        await self.session.commit()
        await self.session.refresh(session)
        return session
    
    async def get_by_id(self, session_id: int) -> Optional[ChatSession]:
        """Get chat session by its ID."""
        stmt = select(ChatSession).where(ChatSession.id == session_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def update(self, session: ChatSession) -> ChatSession:
        """Update an existing chat session."""
        from datetime import datetime
        session.updated_at = datetime.utcnow()
        self.session.add(session)
        await self.session.commit()
        await self.session.refresh(session)
        return session
    
    async def delete(self, session_id: int) -> bool:
        """Delete chat session by its ID (cascades to messages)."""
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
        filters: Optional[dict] = None
    ) -> List[ChatSession]:
        """List chat sessions with optional pagination and filtering."""
        stmt = select(ChatSession).order_by(ChatSession.updated_at.desc())
        
        if filters:
            if filters.get("tenant_id"):
                stmt = stmt.where(ChatSession.tenant_id == filters["tenant_id"])
            if filters.get("agent_role"):
                stmt = stmt.where(ChatSession.agent_role == filters["agent_role"])
            if filters.get("is_active"):
                stmt = stmt.where(ChatSession.is_active == filters["is_active"])
            if filters.get("run_id"):
                stmt = stmt.where(ChatSession.run_id == filters["run_id"])
        
        if limit:
            stmt = stmt.limit(limit)
        if offset:
            stmt = stmt.offset(offset)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_sessions_for_tenant(self, tenant_id: str) -> List[ChatSession]:
        """Get all sessions for a tenant."""
        stmt = (
            select(ChatSession)
            .where(ChatSession.tenant_id == tenant_id)
            .order_by(ChatSession.updated_at.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_active_sessions_for_role(self, agent_role: str) -> List[ChatSession]:
        """Get all active sessions for a specific agent role."""
        stmt = (
            select(ChatSession)
            .where(
                and_(
                    ChatSession.agent_role == agent_role,
                    ChatSession.is_active == True
                )
            )
            .order_by(ChatSession.updated_at.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def add_message(self, session_id: int, message: ChatMessage) -> ChatMessage:
        """Add a message to a chat session."""
        message.session_id = session_id
        self.session.add(message)
        
        # Update session timestamp
        session = await self.get_by_id(session_id)
        if session:
            from datetime import datetime
            session.updated_at = datetime.utcnow()
        
        await self.session.commit()
        await self.session.refresh(message)
        return message
    
    async def get_messages_for_session(
        self, 
        session_id: int, 
        limit: Optional[int] = None
    ) -> List[ChatMessage]:
        """Get messages for a chat session."""
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
        )
        
        if limit:
            stmt = stmt.limit(limit)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
