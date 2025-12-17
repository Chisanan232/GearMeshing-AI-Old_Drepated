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

    async def create(self, run: AgentRun) -> None: ...

    async def update_status(self, run_id: str, *, status: str) -> None: ...

    async def get(self, run_id: str) -> Optional[AgentRun]: ...


class EventRepository(Protocol):
    """Append-only store for runtime events."""

    async def append(self, event: AgentEvent) -> None: ...


class ApprovalRepository(Protocol):
    """Store approvals and their resolution outcomes."""

    async def create(self, approval: Approval) -> None: ...

    async def get(self, approval_id: str) -> Optional[Approval]: ...

    async def resolve(self, approval_id: str, *, decision: str, decided_by: str | None) -> None: ...


class CheckpointRepository(Protocol):
    """Persist serialized engine state for pause/resume."""

    async def save(self, checkpoint: Checkpoint) -> None: ...

    async def latest(self, run_id: str) -> Optional[Checkpoint]: ...


class ToolInvocationRepository(Protocol):
    """Audit log of side-effecting tool/capability invocations."""

    async def append(self, invocation: ToolInvocation) -> None: ...


class UsageRepository(Protocol):
    """Append-only store of token/cost usage ledger entries."""

    async def append(self, usage: UsageLedgerEntry) -> None: ...
