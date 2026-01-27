"""
Base repository interfaces and utilities.

This module provides the foundational repository patterns and interfaces
used across all repository implementations in the centralized database layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

# Generic type for entity models
EntityType = TypeVar("EntityType")


class BaseRepository(ABC, Generic[EntityType]):
    """Base repository interface with common CRUD operations."""
    
    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session."""
        self.session = session
    
    @abstractmethod
    async def create(self, entity: EntityType) -> EntityType:
        """Create a new entity record."""
        pass
    
    @abstractmethod
    async def get_by_id(self, entity_id: str | int) -> Optional[EntityType]:
        """Get entity by its primary identifier."""
        pass
    
    @abstractmethod
    async def update(self, entity: EntityType) -> EntityType:
        """Update an existing entity record."""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: str | int) -> bool:
        """Delete entity by its primary identifier."""
        pass
    
    @abstractmethod
    async def list(
        self, 
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[EntityType]:
        """List entities with optional pagination and filtering."""
        pass


class QueryBuilder:
    """Utility class for building common database queries."""
    
    @staticmethod
    def apply_filters(query, filters: Dict[str, Any]):
        """Apply filters to a SQLAlchemy query."""
        for key, value in filters.items():
            if value is not None:
                query = query.filter(getattr(query.column_descriptions[0]['type'], key) == value)
        return query
    
    @staticmethod
    def apply_pagination(query, limit: Optional[int], offset: Optional[int]):
        """Apply pagination to a SQLAlchemy query."""
        if limit is not None:
            query = query.limit(limit)
        if offset is not None:
            query = query.offset(offset)
        return query
