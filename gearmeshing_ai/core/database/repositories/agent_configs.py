"""
Agent configuration repository interface and implementation.

This module provides data access operations for agent configuration
management, including CRUD operations and tenant-specific queries.
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import Select, and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..entities.agent_configs import AgentConfig
from .base import BaseRepository


class AgentConfigRepository(BaseRepository[AgentConfig]):
    """Repository for agent configuration data access operations."""
    
    async def create(self, config: AgentConfig) -> AgentConfig:
        """Create a new agent configuration."""
        self.session.add(config)
        await self.session.commit()
        await self.session.refresh(config)
        return config
    
    async def get_by_id(self, config_id: int) -> Optional[AgentConfig]:
        """Get agent configuration by its ID."""
        stmt = select(AgentConfig).where(AgentConfig.id == config_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_by_role(self, role_name: str, tenant_id: Optional[str] = None) -> Optional[AgentConfig]:
        """Get agent configuration by role name and optional tenant."""
        stmt = select(AgentConfig).where(AgentConfig.role_name == role_name)
        
        if tenant_id:
            stmt = stmt.where(AgentConfig.tenant_id == tenant_id)
        else:
            stmt = stmt.where(AgentConfig.tenant_id.is_(None))
        
        stmt = stmt.where(AgentConfig.is_active == True)
        
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def update(self, config: AgentConfig) -> AgentConfig:
        """Update an existing agent configuration."""
        from datetime import datetime
        config.updated_at = datetime.utcnow()
        self.session.add(config)
        await self.session.commit()
        await self.session.refresh(config)
        return config
    
    async def delete(self, config_id: int) -> bool:
        """Delete agent configuration by its ID."""
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
        filters: Optional[dict] = None
    ) -> List[AgentConfig]:
        """List agent configurations with optional pagination and filtering."""
        stmt = select(AgentConfig).where(AgentConfig.is_active == True).order_by(AgentConfig.role_name)
        
        if filters:
            if filters.get("tenant_id"):
                stmt = stmt.where(AgentConfig.tenant_id == filters["tenant_id"])
            if filters.get("model_provider"):
                stmt = stmt.where(AgentConfig.model_provider == filters["model_provider"])
            if filters.get("role_name"):
                stmt = stmt.where(AgentConfig.role_name.ilike(f"%{filters['role_name']}%"))
        
        if limit:
            stmt = stmt.limit(limit)
        if offset:
            stmt = stmt.offset(offset)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_active_configs_for_tenant(self, tenant_id: str) -> List[AgentConfig]:
        """Get all active configurations for a tenant."""
        stmt = (
            select(AgentConfig)
            .where(
                and_(
                    AgentConfig.tenant_id == tenant_id,
                    AgentConfig.is_active == True
                )
            )
            .order_by(AgentConfig.role_name)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_global_configs(self) -> List[AgentConfig]:
        """Get all global (non-tenant) configurations."""
        stmt = (
            select(AgentConfig)
            .where(
                and_(
                    AgentConfig.tenant_id.is_(None),
                    AgentConfig.is_active == True
                )
            )
            .order_by(AgentConfig.role_name)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def deactivate_config(self, config_id: int) -> Optional[AgentConfig]:
        """Deactivate an agent configuration."""
        config = await self.get_by_id(config_id)
        if config:
            config.is_active = False
            await self.update(config)
        return config
