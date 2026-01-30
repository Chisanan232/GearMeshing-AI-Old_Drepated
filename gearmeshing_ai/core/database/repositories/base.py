"""
Base repository interfaces and utilities.

This module provides the foundational repository patterns and interfaces
used across all repository implementations in the centralized database layer.
Built with async SQLAlchemy to match the performance requirements of the agent runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel

# Generic type for SQLModel entities
EntityType = TypeVar("EntityType", bound=SQLModel)


class AsyncBaseRepository(ABC, Generic[EntityType]):
    """Base async repository interface with common CRUD operations using SQLModel."""

    def __init__(self, session: AsyncSession, model: Type[EntityType]) -> None:
        """Initialize repository with async database session and SQLModel entity class.

        Args:
            session: Async SQLModel Session for database operations
            model: SQLModel entity class for this repository
        """
        self.session = session
        self.model = model

    @abstractmethod
    async def create(self, entity: EntityType) -> EntityType:
        """Create a new entity record.

        Args:
            entity: SQLModel instance to persist

        Returns:
            Persisted entity with generated fields populated
        """

    @abstractmethod
    async def get_by_id(self, entity_id: str | int) -> Optional[EntityType]:
        """Get entity by its primary identifier.

        Args:
            entity_id: Primary key value

        Returns:
            Entity instance or None if not found
        """

    @abstractmethod
    async def update(self, entity: EntityType) -> EntityType:
        """Update an existing entity record.

        Args:
            entity: SQLModel instance with updated fields

        Returns:
            Updated entity instance
        """

    @abstractmethod
    async def delete(self, entity_id: str | int) -> bool:
        """Delete entity by its primary identifier.

        Args:
            entity_id: Primary key value

        Returns:
            True if deleted, False if not found
        """

    @abstractmethod
    async def list(
        self, limit: Optional[int] = None, offset: Optional[int] = None, filters: Optional[Dict[str, Any]] = None
    ) -> List[EntityType]:
        """List entities with optional pagination and filtering.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            filters: Dictionary of field filters

        Returns:
            List of entity instances
        """


# Legacy sync interface for backward compatibility
class BaseRepository(ABC, Generic[EntityType]):
    """Base repository interface with common CRUD operations using SQLModel."""

    def __init__(self, session: AsyncSession, model: Type[EntityType]) -> None:
        """Initialize repository with database session and SQLModel entity class.

        Args:
            session: Async SQLModel Session for database operations
            model: SQLModel entity class for this repository
        """
        self.session = session
        self.model = model

    @abstractmethod
    async def create(self, entity: EntityType) -> EntityType:
        """Create a new entity record.

        Args:
            entity: SQLModel instance to persist

        Returns:
            Persisted entity with generated fields populated
        """

    @abstractmethod
    async def get_by_id(self, entity_id: str | int) -> Optional[EntityType]:
        """Get entity by its primary identifier.

        Args:
            entity_id: Primary key value

        Returns:
            Entity instance or None if not found
        """

    @abstractmethod
    async def update(self, entity: EntityType) -> EntityType:
        """Update an existing entity record.

        Args:
            entity: SQLModel instance with updated fields

        Returns:
            Updated entity instance
        """

    @abstractmethod
    async def delete(self, entity_id: str | int) -> bool:
        """Delete entity by its primary identifier.

        Args:
            entity_id: Primary key value

        Returns:
            True if deleted, False if not found
        """

    @abstractmethod
    async def list(
        self, limit: Optional[int] = None, offset: Optional[int] = None, filters: Optional[Dict[str, Any]] = None
    ) -> List[EntityType]:
        """List entities with optional pagination and filtering.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            filters: Dictionary of field filters

        Returns:
            List of entity instances
        """


class QueryBuilder:
    """Utility class for building SQLModel-based database queries."""

    @staticmethod
    def apply_filters(stmt, model: Type[EntityType], filters: Dict[str, Any]):
        """Apply filters to a SQLModel select statement.

        Args:
            stmt: SQLModel select statement
            model: SQLModel entity class
            filters: Dictionary of field filters

        Returns:
            Modified select statement with filters applied
        """
        for key, value in filters.items():
            if value is not None and hasattr(model, key):
                stmt = stmt.where(getattr(model, key) == value)
        return stmt

    @staticmethod
    def apply_pagination(stmt, limit: Optional[int], offset: Optional[int]):
        """Apply pagination to a SQLModel select statement.

        Args:
            stmt: SQLModel select statement
            limit: Maximum number of records
            offset: Number of records to skip

        Returns:
            Modified select statement with pagination applied
        """
        if limit is not None:
            stmt = stmt.limit(limit)
        if offset is not None:
            stmt = stmt.offset(offset)
        return stmt


class AsyncQueryBuilder:
    """Utility class for building async SQLModel-based database queries."""

    @staticmethod
    def apply_filters(stmt, model: Type[EntityType], filters: Dict[str, Any]):
        """Apply filters to a SQLModel select statement.

        Args:
            stmt: SQLModel select statement
            model: SQLModel entity class
            filters: Dictionary of field filters

        Returns:
            Modified select statement with filters applied
        """
        for key, value in filters.items():
            if value is not None and hasattr(model, key):
                stmt = stmt.where(getattr(model, key) == value)
        return stmt

    @staticmethod
    def apply_pagination(stmt, limit: Optional[int], offset: Optional[int]):
        """Apply pagination to a SQLModel select statement.

        Args:
            stmt: SQLModel select statement
            limit: Maximum number of records
            offset: Number of records to skip

        Returns:
            Modified select statement with pagination applied
        """
        if limit is not None:
            stmt = stmt.limit(limit)
        if offset is not None:
            stmt = stmt.offset(offset)
        return stmt


def _utc_now_naive() -> datetime:
    """Get current UTC datetime as naive datetime.

    Returns:
        Current UTC datetime without timezone info
    """
    return datetime.now(timezone.utc)
