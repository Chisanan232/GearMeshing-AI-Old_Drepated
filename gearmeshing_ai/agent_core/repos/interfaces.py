from __future__ import annotations

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
    async def create(self, run: AgentRun) -> None: ...

    async def update_status(self, run_id: str, *, status: str) -> None: ...

    async def get(self, run_id: str) -> Optional[AgentRun]: ...


class EventRepository(Protocol):
    async def append(self, event: AgentEvent) -> None: ...


class ApprovalRepository(Protocol):
    async def create(self, approval: Approval) -> None: ...

    async def get(self, approval_id: str) -> Optional[Approval]: ...

    async def resolve(self, approval_id: str, *, decision: str, decided_by: str | None) -> None: ...


class CheckpointRepository(Protocol):
    async def save(self, checkpoint: Checkpoint) -> None: ...

    async def latest(self, run_id: str) -> Optional[Checkpoint]: ...


class ToolInvocationRepository(Protocol):
    async def append(self, invocation: ToolInvocation) -> None: ...


class UsageRepository(Protocol):
    async def append(self, usage: UsageLedgerEntry) -> None: ...
