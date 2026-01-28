"""
Policy repository interface and implementation.

This module provides data access operations for tenant policy
configurations, including policy management and queries.
Built exclusively on SQLModel for type-safe ORM operations.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from ..entities.policies import Policy
from gearmeshing_ai.core.models.domain.policy import PolicyConfig
from .base import BaseRepository, QueryBuilder, AsyncQueryBuilder, _utc_now_naive


class PolicyRepository(BaseRepository[Policy]):
    """Repository for policy data access operations using SQLModel."""
    
    def __init__(self, session: Session) -> None:
        """Initialize repository with database session.
        
        Args:
            session: SQLModel Session for database operations
        """
        super().__init__(session, Policy)
    
    async def create(self, policy: Policy) -> Policy:
        """Create a new policy configuration.
        
        Args:
            policy: Policy SQLModel instance
            
        Returns:
            Persisted Policy with generated fields
        """
        self.session.add(policy)
        self.session.commit()
        self.session.refresh(policy)
        return policy
    
    async def get_by_id(self, policy_id: str | int) -> Optional[Policy]:
        """Get policy by its ID.
        
        Args:
            policy_id: Policy ID
            
        Returns:
            Policy instance or None
        """
        stmt = select(Policy).where(Policy.id == policy_id)
        result = self.session.exec(stmt)
        return result.one_or_none()
    
    async def get_by_tenant(self, tenant_id: str) -> Optional[Policy]:
        """Get policy configuration for a specific tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Policy instance or None
        """
        stmt = select(Policy).where(Policy.tenant_id == tenant_id)
        result = self.session.exec(stmt)
        return result.one_or_none()
    
    async def update(self, policy: Policy) -> Policy:
        """Update an existing policy configuration.
        
        Args:
            policy: Policy instance with updated fields
            
        Returns:
            Updated Policy instance
        """
        policy.updated_at = _utc_now_naive()
        self.session.add(policy)
        self.session.commit()
        self.session.refresh(policy)
        return policy
    
    async def delete(self, policy_id: str | int) -> bool:
        """Delete policy by its ID.
        
        Args:
            policy_id: Policy ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        policy = await self.get_by_id(policy_id)
        if policy:
            self.session.delete(policy)
            self.session.commit()
            return True
        return False
    
    async def list(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Policy]:
        """List policies with optional pagination and filtering.
        
        Args:
            limit: Maximum records to return
            offset: Records to skip
            filters: Field filters (tenant_id)
            
        Returns:
            List of Policy instances
        """
        stmt = select(Policy).order_by(Policy.tenant_id)
        
        if filters:
            stmt = QueryBuilder.apply_filters(stmt, Policy, filters)
        
        stmt = QueryBuilder.apply_pagination(stmt, limit, offset)
        
        result = self.session.exec(stmt)
        return list(result)
    
    # Match the old interface first
    async def get(self, tenant_id: str) -> Optional[PolicyConfig]:
        """Get policy config by tenant ID (backward compatibility wrapper).
        
        This method wraps get_by_tenant() to maintain compatibility with
        the old agent_core/repos PolicyRepository interface.
        
        Args:
            tenant_id: The tenant identifier
            
        Returns:
            PolicyConfig object if found, None otherwise
        """
        policy = await self.get_by_tenant(tenant_id)
        if policy is None:
            return None
        return PolicyConfig.model_validate(policy.config)
    
    # Match the old interface first
    async def update(self, tenant_id: str, config: PolicyConfig) -> None:
        """Update or create policy config (backward compatibility wrapper).
        
        This method wraps the standard update() to maintain compatibility with
        the old agent_core/repos PolicyRepository interface.
        
        Args:
            tenant_id: The tenant identifier
            config: The new PolicyConfig object
        """
        # Get existing policy or create new one
        policy = await self.get_by_tenant(tenant_id)
        
        if policy is None:
            policy = Policy(
                id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                config=config.model_dump(mode="json"),
                created_at=_utc_now_naive(),
                updated_at=_utc_now_naive()
            )
            self.session.add(policy)
        else:
            policy.config = config.model_dump(mode="json")
            policy.updated_at = _utc_now_naive()
            self.session.add(policy)
        
        self.session.commit()
        self.session.refresh(policy)
