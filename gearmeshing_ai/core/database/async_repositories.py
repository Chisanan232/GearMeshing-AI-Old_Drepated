"""
Async agent run repository interface and implementation.

This module provides async data access operations for agent run lifecycle
management, including creation, status updates, and querying.
Built with async SQLAlchemy to match the performance requirements of the agent runtime.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from .async_base import AsyncBaseRepository, AsyncQueryBuilder, _utc_now_naive
from .entities.agent_runs import AgentRun


class AsyncAgentRunRepository(AsyncBaseRepository[AgentRun]):
    """Async repository for agent run data access operations using SQLModel."""
    
    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with async database session.
        
        Args:
            session: Async SQLModel Session for database operations
        """
        super().__init__(session, AgentRun)
    
    async def create(self, run: AgentRun) -> AgentRun:
        """Create a new agent run record.
        
        Args:
            run: AgentRun SQLModel instance
            
        Returns:
            Persisted AgentRun with generated fields
        """
        self.session.add(run)
        await self.session.commit()
        await self.session.refresh(run)
        return run
    
    async def get_by_id(self, run_id: str | int) -> Optional[AgentRun]:
        """Get agent run by its ID.
        
        Args:
            run_id: Run ID (converted to string)
            
        Returns:
            AgentRun instance or None
        """
        run_id_str = str(run_id)
        stmt = select(AgentRun).where(AgentRun.id == run_id_str)
        result = await self.session.exec(stmt)
        return result.one_or_none()
    
    async def update(self, run: AgentRun) -> AgentRun:
        """Update an existing agent run record.
        
        Args:
            run: AgentRun instance with updated fields
            
        Returns:
            Updated AgentRun instance
        """
        run.updated_at = _utc_now_naive()
        self.session.add(run)
        await self.session.commit()
        await self.session.refresh(run)
        return run
    
    async def delete(self, run_id: str | int) -> bool:
        """Delete agent run by its ID.
        
        Args:
            run_id: Run ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        run_id_str = str(run_id)
        run = await self.get_by_id(run_id_str)
        if run:
            await self.session.delete(run)
            await self.session.commit()
            return True
        return False
    
    async def list(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[AgentRun]:
        """List agent runs with optional pagination and filtering.
        
        Args:
            limit: Maximum records to return
            offset: Records to skip
            filters: Field filters (tenant_id, status, role, workspace_id)
            
        Returns:
            List of AgentRun instances
        """
        stmt = select(AgentRun).order_by(AgentRun.created_at.desc())
        
        if filters:
            stmt = AsyncQueryBuilder.apply_filters(stmt, AgentRun, filters)
        
        stmt = AsyncQueryBuilder.apply_pagination(stmt, limit, offset)
        
        result = await self.session.exec(stmt)
        return list(result)
    
    # Additional methods that match the old interface
    async def get(self, run_id: str) -> Optional[AgentRun]:
        """Get run by ID (alias for get_by_id to match old interface).
        
        Args:
            run_id: Run identifier
            
        Returns:
            AgentRun instance or None
        """
        return await self.get_by_id(run_id)
    
    async def update_status(self, run_id: str, *, status: str) -> None:
        """Update the status of an existing run.
        
        Args:
            run_id: The ID of the run to update
            status: The new status value
        """
        run = await self.get_by_id(run_id)
        if run:
            run.status = status
            run.updated_at = _utc_now_naive()
            await self.session.commit()
    
    async def list_by_tenant(
        self, 
        tenant_id: Optional[str] = None, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[AgentRun]:
        """List runs, optionally filtered by tenant.
        
        Args:
            tenant_id: Optional tenant identifier to filter by
            limit: Max number of records to return
            offset: Pagination offset
            
        Returns:
            A list of AgentRun objects
        """
        filters = {"tenant_id": tenant_id} if tenant_id else None
        return await self.list(limit=limit, offset=offset, filters=filters)
