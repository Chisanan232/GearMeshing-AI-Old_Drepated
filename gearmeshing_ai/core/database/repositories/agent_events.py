"""
Agent event repository interface and implementation.

This module provides data access operations for agent event streaming,
including append-only event creation and timeline queries.
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import Select, and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..entities.agent_events import AgentEvent
from .base import BaseRepository


class AgentEventRepository(BaseRepository[AgentEvent]):
    """Repository for agent event data access operations."""
    
    async def create(self, event: AgentEvent) -> AgentEvent:
        """Create a new agent event (append-only operation)."""
        self.session.add(event)
        await self.session.commit()
        await self.session.refresh(event)
        return event
    
    async def get_by_id(self, event_id: str) -> Optional[AgentEvent]:
        """Get agent event by its ID."""
        stmt = select(AgentEvent).where(AgentEvent.id == event_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def update(self, event: AgentEvent) -> AgentEvent:
        """Update operation not supported for events (append-only)."""
        raise NotImplementedError("Events are append-only and cannot be updated")
    
    async def delete(self, event_id: str) -> bool:
        """Delete operation not supported for events (append-only)."""
        raise NotImplementedError("Events are append-only and cannot be deleted")
    
    async def list(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[dict] = None
    ) -> List[AgentEvent]:
        """List agent events with optional pagination and filtering."""
        stmt = select(AgentEvent).order_by(AgentEvent.created_at.desc())
        
        if filters:
            if filters.get("run_id"):
                stmt = stmt.where(AgentEvent.run_id == filters["run_id"])
            if filters.get("type"):
                stmt = stmt.where(AgentEvent.type == filters["type"])
            if filters.get("correlation_id"):
                stmt = stmt.where(AgentEvent.correlation_id == filters["correlation_id"])
        
        if limit:
            stmt = stmt.limit(limit)
        if offset:
            stmt = stmt.offset(offset)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_events_for_run(
        self, 
        run_id: str, 
        limit: Optional[int] = None
    ) -> List[AgentEvent]:
        """Get all events for a specific agent run."""
        stmt = (
            select(AgentEvent)
            .where(AgentEvent.run_id == run_id)
            .order_by(AgentEvent.created_at.asc())
        )
        
        if limit:
            stmt = stmt.limit(limit)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_events_by_type(
        self, 
        run_id: str, 
        event_type: str
    ) -> List[AgentEvent]:
        """Get events of a specific type for a run."""
        stmt = (
            select(AgentEvent)
            .where(
                and_(
                    AgentEvent.run_id == run_id,
                    AgentEvent.type == event_type
                )
            )
            .order_by(AgentEvent.created_at.asc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_events_by_correlation(
        self, 
        correlation_id: str
    ) -> List[AgentEvent]:
        """Get all events with a specific correlation ID."""
        stmt = (
            select(AgentEvent)
            .where(AgentEvent.correlation_id == correlation_id)
            .order_by(AgentEvent.created_at.asc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
