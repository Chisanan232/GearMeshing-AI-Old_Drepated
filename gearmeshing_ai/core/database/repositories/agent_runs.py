"""
Agent run repository interface and implementation.

This module provides data access operations for agent run lifecycle
management, including creation, status updates, and querying.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import Select, and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..entities.agent_runs import AgentRun
from .base import BaseRepository


class AgentRunRepository(BaseRepository[AgentRun]):
    """Repository for agent run data access operations."""
    
    async def create(self, run: AgentRun) -> AgentRun:
        """Create a new agent run record."""
        self.session.add(run)
        await self.session.commit()
        await self.session.refresh(run)
        return run
    
    async def get_by_id(self, run_id: str) -> Optional[AgentRun]:
        """Get agent run by its ID."""
        stmt = select(AgentRun).where(AgentRun.id == run_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def update(self, run: AgentRun) -> AgentRun:
        """Update an existing agent run record."""
        run.updated_at = datetime.utcnow()
        self.session.add(run)
        await self.session.commit()
        await self.session.refresh(run)
        return run
    
    async def delete(self, run_id: str) -> bool:
        """Delete agent run by its ID."""
        run = await self.get_by_id(run_id)
        if run:
            await self.session.delete(run)
            await self.session.commit()
            return True
        return False
    
    async def list(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[dict] = None
    ) -> List[AgentRun]:
        """List agent runs with optional pagination and filtering."""
        stmt = select(AgentRun).order_by(AgentRun.created_at.desc())
        
        if filters:
            if filters.get("tenant_id"):
                stmt = stmt.where(AgentRun.tenant_id == filters["tenant_id"])
            if filters.get("status"):
                stmt = stmt.where(AgentRun.status == filters["status"])
            if filters.get("role"):
                stmt = stmt.where(AgentRun.role == filters["role"])
            if filters.get("workspace_id"):
                stmt = stmt.where(AgentRun.workspace_id == filters["workspace_id"])
        
        if limit:
            stmt = stmt.limit(limit)
        if offset:
            stmt = stmt.offset(offset)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_by_tenant_and_status(
        self, 
        tenant_id: str, 
        status: str
    ) -> List[AgentRun]:
        """Get runs by tenant and status."""
        stmt = (
            select(AgentRun)
            .where(and_(AgentRun.tenant_id == tenant_id, AgentRun.status == status))
            .order_by(AgentRun.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def update_status(self, run_id: str, status: str) -> Optional[AgentRun]:
        """Update the status of an agent run."""
        run = await self.get_by_id(run_id)
        if run:
            run.status = status
            run.updated_at = datetime.utcnow()
            await self.session.commit()
            await self.session.refresh(run)
        return run
    
    async def get_active_runs_for_tenant(self, tenant_id: str) -> List[AgentRun]:
        """Get all active runs for a tenant."""
        active_statuses = ["running", "paused"]
        stmt = (
            select(AgentRun)
            .where(
                and_(
                    AgentRun.tenant_id == tenant_id,
                    AgentRun.status.in_(active_statuses)
                )
            )
            .order_by(AgentRun.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
