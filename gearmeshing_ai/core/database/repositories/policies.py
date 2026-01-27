"""
Policy repository interface and implementation.

This module provides data access operations for tenant policy
configurations, including policy management and queries.
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..entities.policies import Policy
from .base import BaseRepository


class PolicyRepository(BaseRepository[Policy]):
    """Repository for policy data access operations."""
    
    async def create(self, policy: Policy) -> Policy:
        """Create a new policy configuration."""
        self.session.add(policy)
        await self.session.commit()
        await self.session.refresh(policy)
        return policy
    
    async def get_by_id(self, policy_id: str) -> Optional[Policy]:
        """Get policy by its ID."""
        stmt = select(Policy).where(Policy.id == policy_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_by_tenant(self, tenant_id: str) -> Optional[Policy]:
        """Get policy configuration for a specific tenant."""
        stmt = select(Policy).where(Policy.tenant_id == tenant_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def update(self, policy: Policy) -> Policy:
        """Update an existing policy configuration."""
        from datetime import datetime
        policy.updated_at = datetime.utcnow()
        self.session.add(policy)
        await self.session.commit()
        await self.session.refresh(policy)
        return policy
    
    async def delete(self, policy_id: str) -> bool:
        """Delete policy by its ID."""
        policy = await self.get_by_id(policy_id)
        if policy:
            await self.session.delete(policy)
            await self.session.commit()
            return True
        return False
    
    async def list(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[dict] = None
    ) -> List[Policy]:
        """List policies with optional pagination and filtering."""
        stmt = select(Policy).order_by(Policy.tenant_id)
        
        if filters:
            if filters.get("tenant_id"):
                stmt = stmt.where(Policy.tenant_id == filters["tenant_id"])
        
        if limit:
            stmt = stmt.limit(limit)
        if offset:
            stmt = stmt.offset(offset)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
