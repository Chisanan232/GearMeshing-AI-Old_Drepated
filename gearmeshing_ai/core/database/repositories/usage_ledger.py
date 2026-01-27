"""
Usage ledger repository interface and implementation.

This module provides data access operations for token and cost accounting,
including usage tracking and billing analytics.
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import Select, and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..entities.usage_ledger import UsageLedger
from .base import BaseRepository


class UsageLedgerRepository(BaseRepository[UsageLedger]):
    """Repository for usage ledger data access operations."""
    
    async def create(self, usage: UsageLedger) -> UsageLedger:
        """Create a new usage ledger entry."""
        self.session.add(usage)
        await self.session.commit()
        await self.session.refresh(usage)
        return usage
    
    async def get_by_id(self, usage_id: str) -> Optional[UsageLedger]:
        """Get usage ledger entry by its ID."""
        stmt = select(UsageLedger).where(UsageLedger.id == usage_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def update(self, usage: UsageLedger) -> UsageLedger:
        """Update an existing usage ledger entry."""
        self.session.add(usage)
        await self.session.commit()
        await self.session.refresh(usage)
        return usage
    
    async def delete(self, usage_id: str) -> bool:
        """Delete usage ledger entry by its ID."""
        usage = await self.get_by_id(usage_id)
        if usage:
            await self.session.delete(usage)
            await self.session.commit()
            return True
        return False
    
    async def list(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[dict] = None
    ) -> List[UsageLedger]:
        """List usage ledger entries with optional pagination and filtering."""
        stmt = select(UsageLedger).order_by(UsageLedger.created_at.desc())
        
        if filters:
            if filters.get("run_id"):
                stmt = stmt.where(UsageLedger.run_id == filters["run_id"])
            if filters.get("tenant_id"):
                stmt = stmt.where(UsageLedger.tenant_id == filters["tenant_id"])
            if filters.get("provider"):
                stmt = stmt.where(UsageLedger.provider == filters["provider"])
            if filters.get("model"):
                stmt = stmt.where(UsageLedger.model == filters["model"])
        
        if limit:
            stmt = stmt.limit(limit)
        if offset:
            stmt = stmt.offset(offset)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_usage_for_run(self, run_id: str) -> List[UsageLedger]:
        """Get all usage entries for a specific agent run."""
        stmt = (
            select(UsageLedger)
            .where(UsageLedger.run_id == run_id)
            .order_by(UsageLedger.created_at.asc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_usage_for_tenant(self, tenant_id: str) -> List[UsageLedger]:
        """Get all usage entries for a specific tenant."""
        stmt = (
            select(UsageLedger)
            .where(UsageLedger.tenant_id == tenant_id)
            .order_by(UsageLedger.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_total_usage_for_run(self, run_id: str) -> Optional[dict]:
        """Get total token usage and cost for a specific run."""
        stmt = (
            select(
                func.sum(UsageLedger.total_tokens).label('total_tokens'),
                func.sum(UsageLedger.prompt_tokens).label('prompt_tokens'),
                func.sum(UsageLedger.completion_tokens).label('completion_tokens'),
                func.sum(UsageLedger.cost_usd).label('total_cost')
            )
            .where(UsageLedger.run_id == run_id)
        )
        result = await self.session.execute(stmt)
        row = result.first()
        if row and row.total_tokens:
            return {
                'total_tokens': row.total_tokens,
                'prompt_tokens': row.prompt_tokens,
                'completion_tokens': row.completion_tokens,
                'total_cost': row.total_cost
            }
        return None
