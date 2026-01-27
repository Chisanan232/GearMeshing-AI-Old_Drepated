"""
Agent event repository interface and implementation.

This module provides data access operations for agent event streaming,
including append-only event creation and timeline queries.
Built exclusively on SQLModel for type-safe ORM operations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from ..entities.agent_events import AgentEvent
from .base import BaseRepository, QueryBuilder


class AgentEventRepository(BaseRepository[AgentEvent]):
    """Repository for agent event data access operations using SQLModel.
    
    Events are append-only and support creation and querying but not updates or deletes.
    """
    
    def __init__(self, session: Session) -> None:
        """Initialize repository with database session.
        
        Args:
            session: SQLModel Session for database operations
        """
        super().__init__(session, AgentEvent)
    
    async def create(self, event: AgentEvent) -> AgentEvent:
        """Create a new agent event (append-only operation).
        
        Args:
            event: AgentEvent SQLModel instance
            
        Returns:
            Persisted AgentEvent with generated fields
        """
        self.session.add(event)
        await self.session.commit()
        await self.session.refresh(event)
        return event
    
    async def get_by_id(self, event_id: str | int) -> Optional[AgentEvent]:
        """Get agent event by its ID.
        
        Args:
            event_id: Event ID
            
        Returns:
            AgentEvent instance or None
        """
        stmt = select(AgentEvent).where(AgentEvent.id == event_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def update(self, event: AgentEvent) -> AgentEvent:
        """Update operation not supported for events (append-only).
        
        Raises:
            NotImplementedError: Events cannot be updated
        """
        raise NotImplementedError("Events are append-only and cannot be updated")
    
    async def delete(self, event_id: str | int) -> bool:
        """Delete operation not supported for events (append-only).
        
        Raises:
            NotImplementedError: Events cannot be deleted
        """
        raise NotImplementedError("Events are append-only and cannot be deleted")
    
    async def list(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[AgentEvent]:
        """List agent events with optional pagination and filtering.
        
        Args:
            limit: Maximum records to return
            offset: Records to skip
            filters: Field filters (run_id, type, correlation_id)
            
        Returns:
            List of AgentEvent instances
        """
        stmt = select(AgentEvent).order_by(AgentEvent.created_at.desc())
        
        if filters:
            stmt = QueryBuilder.apply_filters(stmt, AgentEvent, filters)
        
        stmt = QueryBuilder.apply_pagination(stmt, limit, offset)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_events_for_run(
        self, 
        run_id: str, 
        limit: Optional[int] = None
    ) -> List[AgentEvent]:
        """Get all events for a specific agent run.
        
        Args:
            run_id: Agent run ID
            limit: Maximum records to return
            
        Returns:
            List of AgentEvent instances in chronological order
        """
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
        """Get events of a specific type for a run.
        
        Args:
            run_id: Agent run ID
            event_type: Event type filter
            
        Returns:
            List of matching AgentEvent instances
        """
        stmt = (
            select(AgentEvent)
            .where(
                (AgentEvent.run_id == run_id) & (AgentEvent.type == event_type)
            )
            .order_by(AgentEvent.created_at.asc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_events_by_correlation(
        self, 
        correlation_id: str
    ) -> List[AgentEvent]:
        """Get all events with a specific correlation ID.
        
        Args:
            correlation_id: Correlation ID filter
            
        Returns:
            List of AgentEvent instances with matching correlation ID
        """
        stmt = (
            select(AgentEvent)
            .where(AgentEvent.correlation_id == correlation_id)
            .order_by(AgentEvent.created_at.asc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
