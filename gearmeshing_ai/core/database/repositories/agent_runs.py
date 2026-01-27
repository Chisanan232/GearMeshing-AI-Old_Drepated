"""
Agent run repository interface and implementation.

This module provides data access operations for agent run lifecycle
management, including creation, status updates, and querying.
Built exclusively on SQLModel for type-safe ORM operations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from ..entities.agent_runs import AgentRun
from .base import BaseRepository, QueryBuilder


class AgentRunRepository(BaseRepository[AgentRun]):
    """Repository for agent run data access operations using SQLModel."""
    
    def __init__(self, session: Session) -> None:
        """Initialize repository with database session.
        
        Args:
            session: SQLModel Session for database operations
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
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def update(self, run: AgentRun) -> AgentRun:
        """Update an existing agent run record.
        
        Args:
            run: AgentRun instance with updated fields
            
        Returns:
            Updated AgentRun instance
        """
        run.updated_at = datetime.utcnow()
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
            stmt = QueryBuilder.apply_filters(stmt, AgentRun, filters)
        
        stmt = QueryBuilder.apply_pagination(stmt, limit, offset)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_by_tenant_and_status(
        self, 
        tenant_id: str, 
        status: str
    ) -> List[AgentRun]:
        """Get runs by tenant and status.
        
        Args:
            tenant_id: Tenant identifier
            status: Run status filter
            
        Returns:
            List of matching AgentRun instances
        """
        stmt = (
            select(AgentRun)
            .where((AgentRun.tenant_id == tenant_id) & (AgentRun.status == status))
            .order_by(AgentRun.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def update_status(self, run_id: str, status: str) -> Optional[AgentRun]:
        """Update the status of an agent run.
        
        Args:
            run_id: Run ID to update
            status: New status value
            
        Returns:
            Updated AgentRun or None if not found
        """
        run = await self.get_by_id(run_id)
        if run:
            run.status = status
            run.updated_at = datetime.utcnow()
            await self.session.commit()
            await self.session.refresh(run)
        return run
    
    async def get_active_runs_for_tenant(self, tenant_id: str) -> List[AgentRun]:
        """Get all active runs for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            List of active AgentRun instances
        """
        active_statuses = ["running", "paused"]
        stmt = (
            select(AgentRun)
            .where(
                (AgentRun.tenant_id == tenant_id) & (AgentRun.status.in_(active_statuses))
            )
            .order_by(AgentRun.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
