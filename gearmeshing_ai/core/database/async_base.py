"""
Async base repository interfaces and utilities.

This module provides the foundational async repository patterns and interfaces
used across all repository implementations in the centralized database layer.
Built with async SQLAlchemy to match the performance requirements of the agent runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, Generic, List, Optional, Protocol, Type, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel, select

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
        pass
    
    @abstractmethod
    async def get_by_id(self, entity_id: str | int) -> Optional[EntityType]:
        """Get entity by its primary identifier.
        
        Args:
            entity_id: Primary key value
            
        Returns:
            Entity instance or None if not found
        """
        pass
    
    @abstractmethod
    async def update(self, entity: EntityType) -> EntityType:
        """Update an existing entity record.
        
        Args:
            entity: SQLModel instance with updated fields
            
        Returns:
            Updated entity instance
        """
        pass
    
    @abstractmethod
    async def delete(self, entity_id: str | int) -> bool:
        """Delete entity by its primary identifier.
        
        Args:
            entity_id: Primary key value
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    async def list(
        self, 
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[EntityType]:
        """List entities with optional pagination and filtering.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            filters: Dictionary of field filters
            
        Returns:
            List of entity instances
        """
        pass


# Import the domain models for the interfaces
from .entities.agent_events import AgentEvent
from .entities.agent_runs import AgentRun
from .entities.approvals import Approval
from .entities.checkpoints import Checkpoint
from .entities.policies import Policy
from .entities.tool_invocations import ToolInvocation
from .entities.usage_ledger import UsageLedger


class EventRepository(Protocol):
    """Append-only store for runtime events."""
    
    async def append(self, event: AgentEvent) -> None:
        """Append a new event to the run's event stream."""
        ...
    
    async def list(self, run_id: str, limit: int = 100) -> List[AgentEvent]:
        """List events for a specific run."""
        ...


class RunRepository(Protocol):
    """Persist and query the lifecycle of an agent run."""
    
    async def create(self, run: AgentRun) -> None:
        """Create a new run record."""
        ...
    
    async def update_status(self, run_id: str, *, status: str) -> None:
        """Update the status of an existing run."""
        ...
    
    async def get(self, run_id: str) -> Optional[AgentRun]:
        """Retrieve a run by its ID."""
        ...
    
    async def list(self, tenant_id: Optional[str] = None, limit: int = 100, offset: int = 0) -> List[AgentRun]:
        """List runs, optionally filtered by tenant."""
        ...


class ApprovalRepository(Protocol):
    """Store approvals and their resolution outcomes."""
    
    async def create(self, approval: Approval) -> None:
        """Create a new approval request."""
        ...
    
    async def get(self, approval_id: str) -> Optional[Approval]:
        """Retrieve an approval by ID."""
        ...
    
    async def resolve(self, approval_id: str, *, decision: str, decided_by: str | None) -> None:
        """Resolve a pending approval."""
        ...
    
    async def list(self, run_id: str, pending_only: bool = True) -> List[Approval]:
        """List approvals for a run."""
        ...


class CheckpointRepository(Protocol):
    """Persist serialized engine state for pause/resume."""
    
    async def save(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint state."""
        ...
    
    async def latest(self, run_id: str) -> Optional[Checkpoint]:
        """Retrieve the most recent checkpoint for a run."""
        ...


class ToolInvocationRepository(Protocol):
    """Audit log of side-effecting tool/capability invocations."""
    
    async def append(self, invocation: ToolInvocation) -> None:
        """Log a tool invocation."""
        ...


class UsageRepository(Protocol):
    """Append-only store of token/cost usage ledger entries."""
    
    async def append(self, usage: UsageLedger) -> None:
        """Record a usage entry."""
        ...
    
    async def list(
        self, tenant_id: str, from_date: Optional[datetime] = None, to_date: Optional[datetime] = None
    ) -> List[UsageLedger]:
        """List usage entries for a tenant within a date range."""
        ...


class PolicyRepository(Protocol):
    """Store and retrieve tenant policy configurations."""
    
    async def get(self, tenant_id: str) -> Optional[Policy]:
        """Retrieve policy config for a tenant."""
        ...
    
    async def update(self, tenant_id: str, config: Any) -> None:
        """Update or create policy config for a tenant."""
        ...


class AsyncQueryBuilder:
    """Utility class for building async SQLModel-based database queries."""
    
    @staticmethod
    def apply_filters(
        stmt,
        model: Type[EntityType],
        filters: Dict[str, Any]
    ):
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
    def apply_pagination(
        stmt,
        limit: Optional[int],
        offset: Optional[int]
    ):
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
