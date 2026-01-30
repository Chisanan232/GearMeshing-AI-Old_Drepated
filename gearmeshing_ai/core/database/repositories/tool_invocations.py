"""
Tool invocation repository interface and implementation.

This module provides data access operations for tool execution records,
including audit trails and risk assessment tracking.
Built exclusively on SQLModel for type-safe ORM operations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from ..entities.tool_invocations import ToolInvocation
from .base import BaseRepository, QueryBuilder


class ToolInvocationRepository(BaseRepository[ToolInvocation]):
    """Repository for tool invocation data access operations using SQLModel."""

    def __init__(self, session: Session) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLModel Session for database operations
        """
        super().__init__(session, ToolInvocation)

    async def create(self, invocation: ToolInvocation) -> ToolInvocation:
        """Create a new tool invocation record.

        Args:
            invocation: ToolInvocation SQLModel instance

        Returns:
            Persisted ToolInvocation with generated fields
        """
        self.session.add(invocation)
        self.session.commit()
        self.session.refresh(invocation)
        return invocation

    async def get_by_id(self, invocation_id: str | int) -> Optional[ToolInvocation]:
        """Get tool invocation by its ID.

        Args:
            invocation_id: Invocation ID

        Returns:
            ToolInvocation instance or None
        """
        stmt = select(ToolInvocation).where(ToolInvocation.id == invocation_id)
        result = self.session.exec(stmt)
        return result.one_or_none()

    async def update(self, invocation: ToolInvocation) -> ToolInvocation:
        """Update an existing tool invocation record.

        Args:
            invocation: ToolInvocation instance with updated fields

        Returns:
            Updated ToolInvocation instance
        """
        self.session.add(invocation)
        self.session.commit()
        self.session.refresh(invocation)
        return invocation

    async def delete(self, invocation_id: str | int) -> bool:
        """Delete tool invocation by its ID.

        Args:
            invocation_id: Invocation ID to delete

        Returns:
            True if deleted, False if not found
        """
        invocation = await self.get_by_id(invocation_id)
        if invocation:
            self.session.delete(invocation)
            self.session.commit()
            return True
        return False

    async def list(
        self, limit: Optional[int] = None, offset: Optional[int] = None, filters: Optional[Dict[str, Any]] = None
    ) -> List[ToolInvocation]:
        """List tool invocations with optional pagination and filtering.

        Args:
            limit: Maximum records to return
            offset: Records to skip
            filters: Field filters (run_id, tool_name, server_id, risk, ok)

        Returns:
            List of ToolInvocation instances
        """
        stmt = select(ToolInvocation).order_by(ToolInvocation.created_at.desc())  # type: ignore

        if filters:
            stmt = QueryBuilder.apply_filters(stmt, ToolInvocation, filters)

        stmt = QueryBuilder.apply_pagination(stmt, limit, offset)

        result = self.session.exec(stmt)
        return list(result)

    async def get_invocations_for_run(self, run_id: str) -> List[ToolInvocation]:
        """Get all tool invocations for a specific agent run.

        Args:
            run_id: Agent run ID

        Returns:
            List of ToolInvocation instances in chronological order
        """
        stmt = (
            select(ToolInvocation)
            .where(ToolInvocation.run_id == run_id)
            .order_by(ToolInvocation.created_at.asc())  # type: ignore
        )
        result = self.session.exec(stmt)
        return list(result)

    async def get_high_risk_invocations(self, risk_level: str = "high") -> List[ToolInvocation]:
        """Get all high-risk tool invocations.

        Args:
            risk_level: Risk level filter (default: "high")

        Returns:
            List of high-risk ToolInvocation instances
        """
        stmt = (
            select(ToolInvocation)
            .where(ToolInvocation.risk == risk_level)
            .order_by(ToolInvocation.created_at.desc())  # type: ignore
        )
        result = self.session.exec(stmt)
        return list(result)

    # Methods to match old interface
    async def append(self, invocation: ToolInvocation) -> None:
        """Log a tool invocation."""
        await self.create(invocation)
