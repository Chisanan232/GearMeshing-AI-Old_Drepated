"""
Tool invocation repository interface and implementation.

This module provides data access operations for tool execution records,
including audit trails and risk assessment tracking.
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import Select, and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..entities.tool_invocations import ToolInvocation
from .base import BaseRepository


class ToolInvocationRepository(BaseRepository[ToolInvocation]):
    """Repository for tool invocation data access operations."""
    
    async def create(self, invocation: ToolInvocation) -> ToolInvocation:
        """Create a new tool invocation record."""
        self.session.add(invocation)
        await self.session.commit()
        await self.session.refresh(invocation)
        return invocation
    
    async def get_by_id(self, invocation_id: str) -> Optional[ToolInvocation]:
        """Get tool invocation by its ID."""
        stmt = select(ToolInvocation).where(ToolInvocation.id == invocation_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def update(self, invocation: ToolInvocation) -> ToolInvocation:
        """Update an existing tool invocation record."""
        self.session.add(invocation)
        await self.session.commit()
        await self.session.refresh(invocation)
        return invocation
    
    async def delete(self, invocation_id: str) -> bool:
        """Delete tool invocation by its ID."""
        invocation = await self.get_by_id(invocation_id)
        if invocation:
            await self.session.delete(invocation)
            await self.session.commit()
            return True
        return False
    
    async def list(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[dict] = None
    ) -> List[ToolInvocation]:
        """List tool invocations with optional pagination and filtering."""
        stmt = select(ToolInvocation).order_by(ToolInvocation.created_at.desc())
        
        if filters:
            if filters.get("run_id"):
                stmt = stmt.where(ToolInvocation.run_id == filters["run_id"])
            if filters.get("tool_name"):
                stmt = stmt.where(ToolInvocation.tool_name == filters["tool_name"])
            if filters.get("server_id"):
                stmt = stmt.where(ToolInvocation.server_id == filters["server_id"])
            if filters.get("risk"):
                stmt = stmt.where(ToolInvocation.risk == filters["risk"])
            if filters.get("ok") is not None:
                stmt = stmt.where(ToolInvocation.ok == filters["ok"])
        
        if limit:
            stmt = stmt.limit(limit)
        if offset:
            stmt = stmt.offset(offset)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_invocations_for_run(self, run_id: str) -> List[ToolInvocation]:
        """Get all tool invocations for a specific agent run."""
        stmt = (
            select(ToolInvocation)
            .where(ToolInvocation.run_id == run_id)
            .order_by(ToolInvocation.created_at.asc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_high_risk_invocations(self, risk_level: str = "high") -> List[ToolInvocation]:
        """Get all high-risk tool invocations."""
        stmt = (
            select(ToolInvocation)
            .where(ToolInvocation.risk == risk_level)
            .order_by(ToolInvocation.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
