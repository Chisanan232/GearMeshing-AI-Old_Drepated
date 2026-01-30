"""
Agent run repository interface and implementation.

This module provides data access operations for agent run lifecycle
management, including creation, status updates, and querying.
Built exclusively on SQLModel for type-safe ORM operations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from ..entities.agent_runs import AgentRun
from .base import BaseRepository, QueryBuilder, _utc_now_naive


class AgentRunRepository(BaseRepository[AgentRun]):
    """Repository for agent run data access operations using SQLModel."""

    def __init__(self, session: Session) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLModel Session for database operations
        """
        super().__init__(session, AgentRun)

    async def create(self, run: AgentRun) -> AgentRun:
        """Create a new agent run record.

        Args:
            run: AgentRun SQLModel instance

        Returns:
            Persisted AgentRun with generated fields
        """
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)
        return run

    async def get_by_id(self, run_id: str | int) -> Optional[AgentRun]:
        """Get agent run by its ID.

        Args:
            run_id: Run ID (converted to string)

        Returns:
            AgentRun instance or None
        """
        run_id_str = str(run_id)
        stmt = select(AgentRun).where(AgentRun.id == run_id_str)
        result = self.session.exec(stmt)
        return result.one_or_none()

    async def update(self, run: AgentRun) -> AgentRun:
        """Update an existing agent run record.

        Args:
            run: AgentRun instance with updated fields

        Returns:
            Updated AgentRun instance
        """
        run.updated_at = datetime.utcnow()
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)
        return run

    async def delete(self, run_id: str | int) -> bool:
        """Delete agent run by its ID.

        Args:
            run_id: Run ID to delete

        Returns:
            True if deleted, False if not found
        """
        run_id_str = str(run_id)
        run = await self.get_by_id(run_id_str)
        if run:
            self.session.delete(run)
            self.session.commit()
            return True
        return False

    async def list(
        self, limit: Optional[int] = None, offset: Optional[int] = None, filters: Optional[Dict[str, Any]] = None
    ) -> List[AgentRun]:
        """List agent runs with optional pagination and filtering.

        Args:
            limit: Maximum records to return
            offset: Records to skip
            filters: Field filters (tenant_id, status, role, workspace_id)

        Returns:
            List of AgentRun instances
        """
        stmt = select(AgentRun).order_by(AgentRun.created_at.desc())  # type: ignore

        if filters:
            stmt = QueryBuilder.apply_filters(stmt, AgentRun, filters)

        stmt = QueryBuilder.apply_pagination(stmt, limit, offset)

        result = self.session.exec(stmt)
        return list(result)

    async def get_by_tenant_and_status(self, tenant_id: str, status: str) -> List[AgentRun]:
        """Get runs by tenant and status.

        Args:
            tenant_id: Tenant identifier
            status: Run status filter

        Returns:
            List of matching AgentRun instances
        """
        stmt = (
            select(AgentRun)
            .where((AgentRun.tenant_id == tenant_id) & (AgentRun.status == status))
            .order_by(AgentRun.created_at.desc())  # type: ignore
        )
        result = self.session.exec(stmt)
        return list(result)

    # Methods to match old interface
    async def get(self, run_id: str) -> Optional[AgentRun]:
        """Get run by ID (alias for get_by_id to match old interface)."""
        return await self.get_by_id(run_id)

    async def update_status(self, run_id: str, *, status: str) -> None:
        """Update the status of an existing run."""
        run = await self.get_by_id(run_id)
        if run:
            run.status = status
            run.updated_at = _utc_now_naive()
            self.session.add(run)
            self.session.commit()

    async def list_by_tenant(
        self, tenant_id: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[AgentRun]:
        """List runs, optionally filtered by tenant."""
        filters = {"tenant_id": tenant_id} if tenant_id else None
        return await self.list(limit=limit, offset=offset, filters=filters)
