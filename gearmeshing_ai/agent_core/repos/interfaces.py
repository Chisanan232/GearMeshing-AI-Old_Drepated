from __future__ import annotations

"""Repository interface contracts.

The runtime depends on these Protocols instead of concrete persistence
implementations.

Contract guidelines
-------------------

- All methods are async.
- Repository implementations should be safe to call from the runtime engine
  without leaking SQLAlchemy sessions/transactions.
- Methods should be idempotent where reasonable (e.g. updating status for an
  unknown run may be a no-op).
- The event repository is append-only.

These interfaces intentionally mirror the engine’s audit needs:

- Runs track coarse status and metadata.
- Events form a timeline of what happened.
- Approvals/checkpoints support pause/resume.
- Tool invocations capture side-effecting actions.
- Usage records track token/cost accounting.
"""

from datetime import datetime
from typing import Optional, Protocol

from ..schemas.domain import (
    AgentEvent,
    AgentRun,
    Approval,
    Checkpoint,
    ToolInvocation,
    UsageLedgerEntry,
)


class RunRepository(Protocol):
    """Persist and query the lifecycle of an agent run."""

    async def create(self, run: AgentRun) -> None:
        """
        Create a new run record.

        Args:
            run: The initial run state to persist.
        """
        ...

    async def update_status(self, run_id: str, *, status: str) -> None:
        """
        Update the status of an existing run.

        Args:
            run_id: The ID of the run to update.
            status: The new status string (e.g., 'running', 'paused').
        """
        ...

    async def get(self, run_id: str) -> Optional[AgentRun]:
        """
        Retrieve a run by its ID.

        Args:
            run_id: The run identifier.

        Returns:
            The AgentRun object if found, else None.
        """
        ...

    async def list(self, tenant_id: Optional[str] = None, limit: int = 100, offset: int = 0) -> list[AgentRun]:
        """
        List runs, optionally filtered by tenant.

        Args:
            tenant_id: Optional tenant identifier to filter by.
            limit: Max number of records to return.
            offset: Pagination offset.

        Returns:
            A list of AgentRun objects.
        """
        ...


class EventRepository(Protocol):
    """Append-only store for runtime events."""

    async def append(self, event: AgentEvent) -> None:
        """
        Append a new event to the run's event stream.

        Args:
            event: The event to persist.
        """
        ...

    async def list(self, run_id: str, limit: int = 100) -> list[AgentEvent]:
        """
        List events for a specific run.

        Args:
            run_id: The run identifier.
            limit: Max number of events to return.

        Returns:
            A list of AgentEvent objects.
        """
        ...


class ApprovalRepository(Protocol):
    """Store approvals and their resolution outcomes."""

    async def create(self, approval: Approval) -> None:
        """
        Create a new approval request.

        Args:
            approval: The approval object containing request details.
        """
        ...

    async def get(self, approval_id: str) -> Optional[Approval]:
        """
        Retrieve an approval by its ID.

        Args:
            approval_id: The approval identifier.

        Returns:
            The Approval object if found, else None.
        """
        ...

    async def resolve(self, approval_id: str, *, decision: str, decided_by: str | None) -> None:
        """
        Resolve a pending approval.

        Args:
            approval_id: The ID of the approval to resolve.
            decision: The decision value (e.g., 'approved', 'rejected').
            decided_by: The identifier of the user/system making the decision.
        """
        ...

    async def list(self, run_id: str, pending_only: bool = True) -> list[Approval]:
        """
        List approvals for a run.

        Args:
            run_id: The run identifier.
            pending_only: If True, return only approvals with decision=None.

        Returns:
            A list of Approval objects.
        """
        ...


class CheckpointRepository(Protocol):
    """Persist serialized engine state for pause/resume."""

    async def save(self, checkpoint: Checkpoint) -> None:
        """
        Save a checkpoint state.

        Args:
            checkpoint: The checkpoint containing serialized graph state.
        """
        ...

    async def latest(self, run_id: str) -> Optional[Checkpoint]:
        """
        Retrieve the most recent checkpoint for a run.

        Args:
            run_id: The run identifier.

        Returns:
            The latest Checkpoint if any exist, else None.
        """
        ...


class ToolInvocationRepository(Protocol):
    """Audit log of side-effecting tool/capability invocations."""

    async def append(self, invocation: ToolInvocation) -> None:
        """
        Log a tool invocation.

        Args:
            invocation: The invocation record including args and result.
        """
        ...


class UsageRepository(Protocol):
    """Append-only store of token/cost usage ledger entries."""

    async def append(self, usage: UsageLedgerEntry) -> None:
        """
        Record a usage entry.

        Args:
            usage: The usage ledger entry to persist.
        """
        ...

    async def list(self, tenant_id: str, from_date: Optional[datetime] = None, to_date: Optional[datetime] = None) -> list[UsageLedgerEntry]:
        """
        List usage entries for a tenant within a date range.

        Args:
            tenant_id: The tenant identifier.
            from_date: Optional start datetime (inclusive).
            to_date: Optional end datetime (inclusive).

        Returns:
            A list of UsageLedgerEntry objects.
        """
        ...


class PolicyRepository(Protocol):
    """Store and retrieve tenant policy configurations."""

    async def get(self, tenant_id: str) -> Optional[dict]:
        """
        Retrieve policy config for a tenant.

        Args:
            tenant_id: The tenant identifier.

        Returns:
            The policy configuration dict if found, else None.
        """
        ...

    async def update(self, tenant_id: str, config: dict) -> None:
        """
        Update or create policy config for a tenant.

        Args:
            tenant_id: The tenant identifier.
            config: The new configuration dict.
        """
        ...
