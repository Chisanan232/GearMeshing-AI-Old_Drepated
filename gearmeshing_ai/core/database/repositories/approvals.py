"""
Approvals repository interface and implementation.

This module provides data access operations for approval workflow management,
including approval requests and decision tracking.
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import Select, and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..entities.approvals import Approval
from .base import BaseRepository


class ApprovalRepository(BaseRepository[Approval]):
    """Repository for approval data access operations."""
    
    async def create(self, approval: Approval) -> Approval:
        """Create a new approval request."""
        self.session.add(approval)
        await self.session.commit()
        await self.session.refresh(approval)
        return approval
    
    async def get_by_id(self, approval_id: str) -> Optional[Approval]:
        """Get approval by its ID."""
        stmt = select(Approval).where(Approval.id == approval_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def update(self, approval: Approval) -> Approval:
        """Update an existing approval record."""
        self.session.add(approval)
        await self.session.commit()
        await self.session.refresh(approval)
        return approval
    
    async def delete(self, approval_id: str) -> bool:
        """Delete approval by its ID."""
        approval = await self.get_by_id(approval_id)
        if approval:
            await self.session.delete(approval)
            await self.session.commit()
            return True
        return False
    
    async def list(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[dict] = None
    ) -> List[Approval]:
        """List approvals with optional pagination and filtering."""
        stmt = select(Approval).order_by(Approval.requested_at.desc())
        
        if filters:
            if filters.get("run_id"):
                stmt = stmt.where(Approval.run_id == filters["run_id"])
            if filters.get("decision"):
                stmt = stmt.where(Approval.decision == filters["decision"])
            if filters.get("risk"):
                stmt = stmt.where(Approval.risk == filters["risk"])
        
        if limit:
            stmt = stmt.limit(limit)
        if offset:
            stmt = stmt.offset(offset)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_pending_approvals(self) -> List[Approval]:
        """Get all pending approval requests."""
        stmt = (
            select(Approval)
            .where(Approval.decision.is_(None))
            .order_by(Approval.requested_at.asc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
