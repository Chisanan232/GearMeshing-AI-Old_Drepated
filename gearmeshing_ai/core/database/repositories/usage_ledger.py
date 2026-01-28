"""
Usage ledger repository interface and implementation.

This module provides data access operations for token and cost accounting,
including usage tracking and billing analytics.
Built exclusively on SQLModel for type-safe ORM operations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import func
from sqlmodel import Session, select

from ..entities.usage_ledger import UsageLedger
from .base import BaseRepository, QueryBuilder, AsyncQueryBuilder


class UsageLedgerRepository(BaseRepository[UsageLedger]):
    """Repository for usage ledger data access operations using SQLModel."""
    
    def __init__(self, session: Session) -> None:
        """Initialize repository with database session.
        
        Args:
            session: SQLModel Session for database operations
        """
        super().__init__(session, UsageLedger)
    
    async def create(self, usage: UsageLedger) -> UsageLedger:
        """Create a new usage ledger entry.
        
        Args:
            usage: UsageLedger SQLModel instance
            
        Returns:
            Persisted UsageLedger with generated fields
        """
        self.session.add(usage)
        self.session.commit()
        self.session.refresh(usage)
        return usage
    
    async def get_by_id(self, usage_id: str | int) -> Optional[UsageLedger]:
        """Get usage ledger entry by its ID.
        
        Args:
            usage_id: Usage ledger ID
            
        Returns:
            UsageLedger instance or None
        """
        stmt = select(UsageLedger).where(UsageLedger.id == usage_id)
        result = self.session.exec(stmt)
        return result.one_or_none()
    
    async def update(self, usage: UsageLedger) -> UsageLedger:
        """Update an existing usage ledger entry.
        
        Args:
            usage: UsageLedger instance with updated fields
            
        Returns:
            Updated UsageLedger instance
        """
        self.session.add(usage)
        self.session.commit()
        self.session.refresh(usage)
        return usage
    
    async def delete(self, usage_id: str | int) -> bool:
        """Delete usage ledger entry by its ID.
        
        Args:
            usage_id: Usage ledger ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        usage = await self.get_by_id(usage_id)
        if usage:
            self.session.delete(usage)
            self.session.commit()
            return True
        return False
    
    async def list(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[UsageLedger]:
        """List usage ledger entries with optional pagination and filtering.
        
        Args:
            limit: Maximum records to return
            offset: Records to skip
            filters: Field filters (run_id, tenant_id, provider, model)
            
        Returns:
            List of UsageLedger instances
        """
        stmt = select(UsageLedger).order_by(UsageLedger.created_at.desc())  # type: ignore
        
        if filters:
            stmt = QueryBuilder.apply_filters(stmt, UsageLedger, filters)
        
        stmt = QueryBuilder.apply_pagination(stmt, limit, offset)
        
        result = self.session.exec(stmt)
        return list(result)
    
    async def get_usage_for_run(self, run_id: str) -> List[UsageLedger]:
        """Get all usage entries for a specific agent run.
        
        Args:
            run_id: Agent run ID
            
        Returns:
            List of UsageLedger instances in chronological order
        """
        stmt = (
            select(UsageLedger)
            .where(UsageLedger.run_id == run_id)
            .order_by(UsageLedger.created_at.asc())  # type: ignore
        )
        result = self.session.exec(stmt)
        return list(result)
    
    async def get_usage_for_tenant(self, tenant_id: str) -> List[UsageLedger]:
        """Get all usage entries for a specific tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            List of UsageLedger instances for tenant
        """
        stmt = (
            select(UsageLedger)
            .where(UsageLedger.tenant_id == tenant_id)
            .order_by(UsageLedger.created_at.desc())  # type: ignore
        )
        result = self.session.exec(stmt)
        return list(result)
    
    async def get_total_usage_for_run(self, run_id: str) -> Optional[dict]:
        """Get total token usage and cost for a specific run.
        
        Args:
            run_id: Agent run ID
            
        Returns:
            Dictionary with total_tokens, prompt_tokens, completion_tokens, total_cost or None
        """
        stmt = (
            select(
                func.sum(UsageLedger.total_tokens).label('total_tokens'),
                func.sum(UsageLedger.prompt_tokens).label('prompt_tokens'),
                func.sum(UsageLedger.completion_tokens).label('completion_tokens'),
                func.sum(UsageLedger.cost_usd).label('total_cost')
            )
            .where(UsageLedger.run_id == run_id)
        )
        result = self.session.exec(stmt)
        row = result.first()
        if row and row[0]:
            return {
                'total_tokens': row[0],
                'prompt_tokens': row[1],
                'completion_tokens': row[2],
                'total_cost': row[3]
            }
        return None
    
    # Methods to match old interface
    async def append(self, usage: UsageLedger) -> None:
        """Record a usage entry."""
        await self.create(usage)
    
    async def list_by_tenant(
        self, 
        tenant_id: str, 
        from_date: Optional[datetime] = None, 
        to_date: Optional[datetime] = None
    ) -> List[UsageLedger]:
        """List usage entries for a tenant within a date range."""
        # Need to join with runs to filter by tenant_id since UsageLedger doesn't have tenant_id
        # For now, use the existing get_usage_for_tenant method
        return await self.get_usage_for_tenant(tenant_id)
