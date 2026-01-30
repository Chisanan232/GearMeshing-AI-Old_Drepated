"""
Checkpoints repository interface and implementation.

This module provides data access operations for LangGraph state checkpoints,
enabling pause/resume functionality for agent workflows.
Built exclusively on SQLModel for type-safe ORM operations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from ..entities.checkpoints import Checkpoint
from .base import BaseRepository, QueryBuilder


class CheckpointRepository(BaseRepository[Checkpoint]):
    """Repository for checkpoint data access operations using SQLModel."""

    def __init__(self, session: Session) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLModel Session for database operations
        """
        super().__init__(session, Checkpoint)

    async def create(self, checkpoint: Checkpoint) -> Checkpoint:
        """Create a new checkpoint.

        Args:
            checkpoint: Checkpoint SQLModel instance

        Returns:
            Persisted Checkpoint with generated fields
        """
        self.session.add(checkpoint)
        self.session.commit()
        self.session.refresh(checkpoint)
        return checkpoint

    async def get_by_id(self, checkpoint_id: str | int) -> Optional[Checkpoint]:
        """Get checkpoint by its ID.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            Checkpoint instance or None
        """
        stmt = select(Checkpoint).where(Checkpoint.id == checkpoint_id)
        result = self.session.exec(stmt)
        return result.one_or_none()

    async def update(self, checkpoint: Checkpoint) -> Checkpoint:
        """Update an existing checkpoint record.

        Args:
            checkpoint: Checkpoint instance with updated fields

        Returns:
            Updated Checkpoint instance
        """
        self.session.add(checkpoint)
        self.session.commit()
        self.session.refresh(checkpoint)
        return checkpoint

    async def delete(self, checkpoint_id: str | int) -> bool:
        """Delete checkpoint by its ID.

        Args:
            checkpoint_id: Checkpoint ID to delete

        Returns:
            True if deleted, False if not found
        """
        checkpoint = await self.get_by_id(checkpoint_id)
        if checkpoint:
            self.session.delete(checkpoint)
            self.session.commit()
            return True
        return False

    async def list(
        self, limit: Optional[int] = None, offset: Optional[int] = None, filters: Optional[Dict[str, Any]] = None
    ) -> List[Checkpoint]:
        """List checkpoints with optional pagination and filtering.

        Args:
            limit: Maximum records to return
            offset: Records to skip
            filters: Field filters (run_id, node)

        Returns:
            List of Checkpoint instances
        """
        stmt = select(Checkpoint).order_by(Checkpoint.created_at.desc())

        if filters:
            stmt = QueryBuilder.apply_filters(stmt, Checkpoint, filters)

        stmt = QueryBuilder.apply_pagination(stmt, limit, offset)

        result = self.session.exec(stmt)
        return list(result)

    async def get_latest_checkpoint_for_run(self, run_id: str) -> Optional[Checkpoint]:
        """Get the latest checkpoint for a specific run.

        Args:
            run_id: Agent run ID

        Returns:
            Latest Checkpoint instance or None
        """
        stmt = select(Checkpoint).where(Checkpoint.run_id == run_id).order_by(Checkpoint.created_at.desc()).limit(1)
        result = self.session.exec(stmt)
        return result.one_or_none()

    # Methods to match old interface
    async def save(self, checkpoint: Checkpoint) -> None:
        """Persist a checkpoint state."""
        await self.create(checkpoint)

    async def latest(self, run_id: str) -> Optional[Checkpoint]:
        """Fetch the most recent checkpoint for a run."""
        return await self.get_latest_checkpoint_for_run(run_id)
