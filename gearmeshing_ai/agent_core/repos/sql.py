from __future__ import annotations

"""SQLAlchemy async repository implementations.

This module provides a Postgres-backed persistence implementation for the
repository interfaces defined in ``gearmeshing_ai.agent_core.repos.interfaces``.

Usage
-----

Typical wiring (tests or application setup):

- Create an async engine with ``create_engine``.
- Create tables with ``create_all`` (for tests/dev; production typically uses
  migrations).
- Create a session factory with ``create_sessionmaker``.
- Build repository instances with ``build_sql_repos``.

Transaction model
-----------------

Each repository method opens an ``AsyncSession``, performs its operation, and
commits. This keeps persistence boundaries simple for the runtime engine and
guarantees that each persisted artifact (event, checkpoint, etc.) is durable
when the method returns.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ..schemas.domain import (
    AgentEvent,
    AgentRun,
    Approval,
    Checkpoint,
    ToolInvocation,
    UsageLedgerEntry,
)
from .interfaces import (
    ApprovalRepository,
    CheckpointRepository,
    EventRepository,
    RunRepository,
    ToolInvocationRepository,
    UsageRepository,
)
from .models import (
    ApprovalRow,
    Base,
    CheckpointRow,
    EventRow,
    RunRow,
    ToolInvocationRow,
    UsageRow,
)


def create_engine(db_url: str) -> AsyncEngine:
    """Create an async SQLAlchemy engine.

    The helper normalizes Postgres URLs to ensure the async driver is used.
    For example, it rewrites ``postgresql://`` and other variants to
    ``postgresql+asyncpg://``.
    """
    url = re.sub(r"^postgres(?:ql)?(?:\+[a-z0-9_]+)?://", "postgresql+asyncpg://", db_url, count=1)
    return create_async_engine(url, pool_pre_ping=True)


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create an ``async_sessionmaker`` with safe defaults for this project."""
    return async_sessionmaker(engine, expire_on_commit=False)


async def create_all(engine: AsyncEngine) -> None:
    """Create all tables for the current ORM metadata.

    This is mainly intended for tests and local development.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def _utc_now_naive() -> datetime:
    # Store timestamps in UTC; DB columns are timezone-aware.
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class SqlRunRepository(RunRepository):
    """SQL implementation of ``RunRepository``."""
    session_factory: async_sessionmaker[AsyncSession]

    async def create(self, run: AgentRun) -> None:
        async with self.session_factory() as s:
            s.add(
                RunRow(
                    id=run.id,
                    tenant_id=run.tenant_id,
                    workspace_id=run.workspace_id,
                    role=run.role,
                    autonomy_profile=str(getattr(run.autonomy_profile, "value", run.autonomy_profile)),
                    objective=run.objective,
                    done_when=run.done_when,
                    prompt_provider_version=run.prompt_provider_version,
                    status=str(getattr(run.status, "value", run.status)),
                    created_at=run.created_at,
                    updated_at=run.updated_at,
                )
            )
            await s.commit()

    async def update_status(self, run_id: str, *, status: str) -> None:
        async with self.session_factory() as s:
            row = await s.get(RunRow, run_id)
            if row is None:
                return
            row.status = status
            row.updated_at = _utc_now_naive()
            await s.commit()

    async def get(self, run_id: str) -> Optional[AgentRun]:
        async with self.session_factory() as s:
            row = await s.get(RunRow, run_id)
            if row is None:
                return None
            return AgentRun(
                id=row.id,
                tenant_id=row.tenant_id,
                workspace_id=row.workspace_id,
                role=row.role,
                autonomy_profile=row.autonomy_profile,
                objective=row.objective,
                done_when=row.done_when,
                prompt_provider_version=row.prompt_provider_version,
                status=row.status,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )


@dataclass(frozen=True)
class SqlEventRepository(EventRepository):
    """SQL implementation of ``EventRepository`` (append-only)."""
    session_factory: async_sessionmaker[AsyncSession]

    async def append(self, event: AgentEvent) -> None:
        async with self.session_factory() as s:
            s.add(
                EventRow(
                    id=event.id,
                    run_id=event.run_id,
                    type=str(getattr(event.type, "value", event.type)),
                    created_at=event.created_at,
                    correlation_id=event.correlation_id,
                    payload=event.payload,
                )
            )
            await s.commit()


@dataclass(frozen=True)
class SqlApprovalRepository(ApprovalRepository):
    """SQL implementation of ``ApprovalRepository``."""
    session_factory: async_sessionmaker[AsyncSession]

    async def create(self, approval: Approval) -> None:
        async with self.session_factory() as s:
            s.add(
                ApprovalRow(
                    id=approval.id,
                    run_id=approval.run_id,
                    risk=str(getattr(approval.risk, "value", approval.risk)),
                    capability=str(getattr(approval.capability, "value", approval.capability)),
                    reason=approval.reason,
                    requested_at=approval.requested_at,
                    expires_at=approval.expires_at,
                    decision=(
                        str(getattr(approval.decision, "value", approval.decision)) if approval.decision else None
                    ),
                    decided_at=approval.decided_at,
                    decided_by=approval.decided_by,
                )
            )
            await s.commit()

    async def get(self, approval_id: str) -> Optional[Approval]:
        async with self.session_factory() as s:
            row = await s.get(ApprovalRow, approval_id)
            if row is None:
                return None
            return Approval(
                id=row.id,
                run_id=row.run_id,
                risk=row.risk,
                capability=row.capability,
                reason=row.reason,
                requested_at=row.requested_at,
                expires_at=row.expires_at,
                decision=row.decision,
                decided_at=row.decided_at,
                decided_by=row.decided_by,
            )

    async def resolve(self, approval_id: str, *, decision: str, decided_by: str | None) -> None:
        async with self.session_factory() as s:
            row = await s.get(ApprovalRow, approval_id)
            if row is None:
                return
            row.decision = decision
            row.decided_by = decided_by
            row.decided_at = _utc_now_naive()
            await s.commit()


@dataclass(frozen=True)
class SqlCheckpointRepository(CheckpointRepository):
    """SQL implementation of ``CheckpointRepository``.

    Checkpoints store serialized graph state used by the engine to resume a
    paused run.
    """
    session_factory: async_sessionmaker[AsyncSession]

    async def save(self, checkpoint: Checkpoint) -> None:
        async with self.session_factory() as s:
            s.add(
                CheckpointRow(
                    id=checkpoint.id,
                    run_id=checkpoint.run_id,
                    node=checkpoint.node,
                    state=checkpoint.state,
                    created_at=checkpoint.created_at,
                )
            )
            await s.commit()

    async def latest(self, run_id: str) -> Optional[Checkpoint]:
        async with self.session_factory() as s:
            stmt = (
                select(CheckpointRow)
                .where(CheckpointRow.run_id == run_id)
                .order_by(CheckpointRow.created_at.desc())
                .limit(1)
            )
            res = await s.execute(stmt)
            row = res.scalar_one_or_none()
            if row is None:
                return None
            return Checkpoint(id=row.id, run_id=row.run_id, node=row.node, state=row.state, created_at=row.created_at)


@dataclass(frozen=True)
class SqlToolInvocationRepository(ToolInvocationRepository):
    """SQL implementation of ``ToolInvocationRepository``."""
    session_factory: async_sessionmaker[AsyncSession]

    async def append(self, invocation: ToolInvocation) -> None:
        async with self.session_factory() as s:
            s.add(
                ToolInvocationRow(
                    id=invocation.id,
                    run_id=invocation.run_id,
                    server_id=invocation.server_id,
                    tool_name=invocation.tool_name,
                    args=invocation.args,
                    ok=invocation.ok,
                    result=invocation.result,
                    risk=str(getattr(invocation.risk, "value", invocation.risk)),
                    created_at=invocation.created_at,
                )
            )
            await s.commit()


@dataclass(frozen=True)
class SqlUsageRepository(UsageRepository):
    """SQL implementation of ``UsageRepository`` (append-only ledger)."""
    session_factory: async_sessionmaker[AsyncSession]

    async def append(self, usage: UsageLedgerEntry) -> None:
        async with self.session_factory() as s:
            s.add(
                UsageRow(
                    id=usage.id,
                    run_id=usage.run_id,
                    provider=usage.provider,
                    model=usage.model,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    cost_usd=usage.cost_usd,
                    created_at=usage.created_at,
                )
            )
            await s.commit()


@dataclass(frozen=True)
class SqlRepoBundle:
    """Convenience bundle of all SQL repositories for dependency injection."""
    runs: SqlRunRepository
    events: SqlEventRepository
    approvals: SqlApprovalRepository
    checkpoints: SqlCheckpointRepository
    tool_invocations: SqlToolInvocationRepository
    usage: SqlUsageRepository


def build_sql_repos(*, session_factory: async_sessionmaker[AsyncSession]) -> SqlRepoBundle:
    """Build a ``SqlRepoBundle`` from a session factory."""
    return SqlRepoBundle(
        runs=SqlRunRepository(session_factory=session_factory),
        events=SqlEventRepository(session_factory=session_factory),
        approvals=SqlApprovalRepository(session_factory=session_factory),
        checkpoints=SqlCheckpointRepository(session_factory=session_factory),
        tool_invocations=SqlToolInvocationRepository(session_factory=session_factory),
        usage=SqlUsageRepository(session_factory=session_factory),
    )
