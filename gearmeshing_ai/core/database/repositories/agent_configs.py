"""
Agent configuration repository interface and implementation.

This module provides data access operations for agent configuration
management, including CRUD operations and tenant-specific queries.
Built exclusively on SQLModel for type-safe ORM operations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from ..entities.agent_configs import AgentConfig
from .base import BaseRepository, QueryBuilder


class AgentConfigRepository(BaseRepository[AgentConfig]):
    """Repository for agent configuration data access operations using SQLModel."""
    
    def __init__(self, session: Session) -> None:
        """Initialize repository with database session.
        
        Args:
            session: SQLModel Session for database operations
        """
        super().__init__(session, AgentConfig)
    
    async def create(self, config: AgentConfig) -> AgentConfig:
        """Create a new agent configuration.
        
        Args:
            config: AgentConfig SQLModel instance
            
        Returns:
            Persisted AgentConfig with generated fields
        """
        self.session.add(config)
        await self.session.commit()
        await self.session.refresh(config)
        return config
    
    async def get_by_id(self, config_id: str | int) -> Optional[AgentConfig]:
        """Get agent configuration by its ID.
        
        Args:
            config_id: Configuration ID
            
        Returns:
            AgentConfig instance or None
        """
        stmt = select(AgentConfig).where(AgentConfig.id == config_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_by_role(
        self,
        role_name: str,
        tenant_id: Optional[str] = None
    ) -> Optional[AgentConfig]:
        """Get agent configuration by role name and optional tenant.
        
        Args:
            role_name: Role identifier
            tenant_id: Optional tenant ID for tenant-specific configs
            
        Returns:
            AgentConfig instance or None
        """
        stmt = select(AgentConfig).where(AgentConfig.role_name == role_name)
        
        if tenant_id:
            stmt = stmt.where(AgentConfig.tenant_id == tenant_id)
        else:
            stmt = stmt.where(AgentConfig.tenant_id.is_(None))
        
        stmt = stmt.where(AgentConfig.is_active == True)
        
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def update(self, config: AgentConfig) -> AgentConfig:
        """Update an existing agent configuration.
        
        Args:
            config: AgentConfig instance with updated fields
            
        Returns:
            Updated AgentConfig instance
        """
        config.updated_at = datetime.utcnow()
        self.session.add(config)
        await self.session.commit()
        await self.session.refresh(config)
        return config
    
    async def delete(self, config_id: str | int) -> bool:
        """Delete agent configuration by its ID.
        
        Args:
            config_id: Configuration ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        config = await self.get_by_id(config_id)
        if config:
            await self.session.delete(config)
            await self.session.commit()
            return True
        return False
    
    async def list(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[AgentConfig]:
        """List agent configurations with optional pagination and filtering.
        
        Args:
            limit: Maximum records to return
            offset: Records to skip
            filters: Field filters (tenant_id, model_provider, role_name)
            
        Returns:
            List of AgentConfig instances
        """
        stmt = select(AgentConfig).where(AgentConfig.is_active == True).order_by(AgentConfig.role_name)
        
        if filters:
            # Apply standard filters
            if filters.get("tenant_id"):
                stmt = stmt.where(AgentConfig.tenant_id == filters["tenant_id"])
            if filters.get("model_provider"):
                stmt = stmt.where(AgentConfig.model_provider == filters["model_provider"])
            # Handle role_name with ILIKE for partial matching
            if filters.get("role_name"):
                stmt = stmt.where(AgentConfig.role_name.ilike(f"%{filters['role_name']}%"))
        
        stmt = QueryBuilder.apply_pagination(stmt, limit, offset)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_active_configs_for_tenant(self, tenant_id: str) -> List[AgentConfig]:
        """Get all active configurations for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            List of active AgentConfig instances for tenant
        """
        stmt = (
            select(AgentConfig)
            .where(
                (AgentConfig.tenant_id == tenant_id) & (AgentConfig.is_active == True)
            )
            .order_by(AgentConfig.role_name)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_global_configs(self) -> List[AgentConfig]:
        """Get all global (non-tenant) configurations.
        
        Returns:
            List of global AgentConfig instances
        """
        stmt = (
            select(AgentConfig)
            .where(
                (AgentConfig.tenant_id.is_(None)) & (AgentConfig.is_active == True)
            )
            .order_by(AgentConfig.role_name)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def deactivate_config(self, config_id: int) -> Optional[AgentConfig]:
        """Deactivate an agent configuration.
        
        Args:
            config_id: Configuration ID to deactivate
            
        Returns:
            Updated AgentConfig or None if not found
        """
        config = await self.get_by_id(config_id)
        if config:
            config.is_active = False
            await self.update(config)
        return config
