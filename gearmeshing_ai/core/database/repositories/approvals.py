"""
Approvals repository interface and implementation.

This module provides data access operations for approval workflow management,
including approval requests and decision tracking.
Built exclusively on SQLModel for type-safe ORM operations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from ..entities.approvals import Approval
from .base import BaseRepository, QueryBuilder


class ApprovalRepository(BaseRepository[Approval]):
    """Repository for approval data access operations using SQLModel."""
    
    def __init__(self, session: Session) -> None:
        """Initialize repository with database session.
        
        Args:
            session: SQLModel Session for database operations
        """
        super().__init__(session, Approval)
    
    async def create(self, approval: Approval) -> Approval:
        """Create a new approval request.
        
        Args:
            approval: Approval SQLModel instance
            
        Returns:
            Persisted Approval with generated fields
        """
        self.session.add(approval)
        self.session.commit()
        self.session.refresh(approval)
        return approval
    
    async def get_by_id(self, approval_id: str | int) -> Optional[Approval]:
        """Get approval by its ID.
        
        Args:
            approval_id: Approval ID
            
        Returns:
            Approval instance or None
        """
        stmt = select(Approval).where(Approval.id == approval_id)
        result = self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def update(self, approval: Approval) -> Approval:
        """Update an existing approval record.
        
        Args:
            approval: Approval instance with updated fields
            
        Returns:
            Updated Approval instance
        """
        self.session.add(approval)
        self.session.commit()
        self.session.refresh(approval)
        return approval
    
    async def delete(self, approval_id: str | int) -> bool:
        """Delete approval by its ID.
        
        Args:
            approval_id: Approval ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        approval = await self.get_by_id(approval_id)
        if approval:
            self.session.delete(approval)
            self.session.commit()
            return True
        return False
    
    async def list(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Approval]:
        """List approvals with optional pagination and filtering.
        
        Args:
            limit: Maximum records to return
            offset: Records to skip
            filters: Field filters (run_id, decision, risk)
            
        Returns:
            List of Approval instances
        """
        stmt = select(Approval).order_by(Approval.requested_at.desc())  # type: ignore
        
        if filters:
            stmt = QueryBuilder.apply_filters(stmt, Approval, filters)
        
        stmt = QueryBuilder.apply_pagination(stmt, limit, offset)
        
        result = self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_pending_approvals(self) -> List[Approval]:
        """Get all pending approval requests.
        
        Returns:
            List of Approval instances with no decision
        """
        stmt = (
            select(Approval)
            .where(Approval.decision.is_(None))  # type: ignore
            .order_by(Approval.requested_at.asc())  # type: ignore
        )
        result = self.session.execute(stmt)
        return list(result.scalars().all())
