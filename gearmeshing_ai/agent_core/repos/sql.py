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
    PolicyRepository,
    RunRepository,
    ToolInvocationRepository,
    UsageRepository,
)
from .models import (
    ApprovalRow,
    Base,
    CheckpointRow,
    EventRow,
    PolicyRow,
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
        """
        Persist a new run record.

        Args:
            run: The run domain object to insert.
        """
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
        """
        Update the status of an existing run.

        Args:
            run_id: The ID of the run to update.
            status: The new status value.
        """
        async with self.session_factory() as s:
            row = await s.get(RunRow, run_id)
            if row is None:
                return
            row.status = status
            row.updated_at = _utc_now_naive()
            await s.commit()

    async def get(self, run_id: str) -> Optional[AgentRun]:
        """
        Retrieve a run by its ID.

        Args:
            run_id: The run identifier.

        Returns:
            The AgentRun domain object if found, otherwise None.
        """
        async with self.session_factory() as s:
            row = await s.get(RunRow, run_id)
            if row is None:
                return None
            from gearmeshing_ai.agent_core.schemas.domain import (
                AgentRunStatus,
                AutonomyProfile,
            )

            return AgentRun(
                id=row.id,
                tenant_id=row.tenant_id,
                workspace_id=row.workspace_id,
                role=row.role,
                autonomy_profile=AutonomyProfile(row.autonomy_profile),
                objective=row.objective,
                done_when=row.done_when,
                prompt_provider_version=row.prompt_provider_version,
                status=AgentRunStatus(row.status),
                created_at=row.created_at,
                updated_at=row.updated_at,
            )

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
        async with self.session_factory() as s:
            stmt = select(RunRow)
            if tenant_id:
                stmt = stmt.where(RunRow.tenant_id == tenant_id)
            stmt = stmt.order_by(RunRow.created_at.desc()).offset(offset).limit(limit)

            result = await s.execute(stmt)
            rows = result.scalars().all()

            from gearmeshing_ai.agent_core.schemas.domain import (
                AgentRunStatus,
                AutonomyProfile,
            )

            return [
                AgentRun(
                    id=row.id,
                    tenant_id=row.tenant_id,
                    workspace_id=row.workspace_id,
                    role=row.role,
                    autonomy_profile=AutonomyProfile(row.autonomy_profile),
                    objective=row.objective,
                    done_when=row.done_when,
                    prompt_provider_version=row.prompt_provider_version,
                    status=AgentRunStatus(row.status),
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
                for row in rows
            ]


@dataclass(frozen=True)
class SqlEventRepository(EventRepository):
    """SQL implementation of ``EventRepository`` (append-only)."""

    session_factory: async_sessionmaker[AsyncSession]

    async def append(self, event: AgentEvent) -> None:
        """
        Append a new event to the store.

        Args:
            event: The event domain object.
        """
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

    async def list(self, run_id: str, limit: int = 100) -> list[AgentEvent]:
        """
        List events for a specific run.

        Args:
            run_id: The run identifier.
            limit: Max number of events to return.

        Returns:
            A list of AgentEvent objects.
        """
        async with self.session_factory() as s:
            stmt = select(EventRow).where(EventRow.run_id == run_id).order_by(EventRow.created_at.asc()).limit(limit)
            result = await s.execute(stmt)
            rows = result.scalars().all()

            from gearmeshing_ai.agent_core.schemas.domain import AgentEventType

            return [
                AgentEvent(
                    id=row.id,
                    run_id=row.run_id,
                    type=AgentEventType(row.type),
                    created_at=row.created_at,
                    correlation_id=row.correlation_id,
                    payload=row.payload,
                )
                for row in rows
            ]


@dataclass(frozen=True)
class SqlApprovalRepository(ApprovalRepository):
    """SQL implementation of ``ApprovalRepository``."""

    session_factory: async_sessionmaker[AsyncSession]

    async def create(self, approval: Approval) -> None:
        """
        Create a new approval request record.

        Args:
            approval: The approval domain object.
        """
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
        """
        Retrieve an approval by ID.

        Args:
            approval_id: The approval identifier.

        Returns:
            The Approval domain object or None.
        """
        async with self.session_factory() as s:
            row = await s.get(ApprovalRow, approval_id)
            if row is None:
                return None
            from gearmeshing_ai.agent_core.schemas.domain import (
                ApprovalDecision,
                CapabilityName,
                RiskLevel,
            )

            return Approval(
                id=row.id,
                run_id=row.run_id,
                risk=RiskLevel(row.risk),
                capability=CapabilityName(row.capability),
                reason=row.reason,
                requested_at=row.requested_at,
                expires_at=row.expires_at,
                decision=ApprovalDecision(row.decision) if row.decision else None,
                decided_at=row.decided_at,
                decided_by=row.decided_by,
            )

    async def resolve(self, approval_id: str, *, decision: str, decided_by: str | None) -> None:
        """
        Update an approval with a decision.

        Args:
            approval_id: The approval identifier.
            decision: The decision string (e.g. 'approved').
            decided_by: The actor who made the decision.
        """
        async with self.session_factory() as s:
            row = await s.get(ApprovalRow, approval_id)
            if row is None:
                return
            row.decision = decision
            row.decided_by = decided_by
            row.decided_at = _utc_now_naive()
            await s.commit()

    async def list(self, run_id: str, pending_only: bool = True) -> list[Approval]:
        """
        List approvals for a run.

        Args:
            run_id: The run identifier.
            pending_only: If True, return only approvals with decision=None.

        Returns:
            A list of Approval objects.
        """
        async with self.session_factory() as s:
            stmt = select(ApprovalRow).where(ApprovalRow.run_id == run_id)
            if pending_only:
                stmt = stmt.where(ApprovalRow.decision.is_(None))
            stmt = stmt.order_by(ApprovalRow.requested_at.asc())

            result = await s.execute(stmt)
            rows = result.scalars().all()

            from gearmeshing_ai.agent_core.schemas.domain import (
                ApprovalDecision,
                CapabilityName,
                RiskLevel,
            )

            return [
                Approval(
                    id=row.id,
                    run_id=row.run_id,
                    risk=RiskLevel(row.risk),
                    capability=CapabilityName(row.capability),
                    reason=row.reason,
                    requested_at=row.requested_at,
                    expires_at=row.expires_at,
                    decision=ApprovalDecision(row.decision) if row.decision else None,
                    decided_at=row.decided_at,
                    decided_by=row.decided_by,
                )
                for row in rows
            ]


@dataclass(frozen=True)
class SqlCheckpointRepository(CheckpointRepository):
    """SQL implementation of ``CheckpointRepository``.

    Checkpoints store serialized graph state used by the engine to resume a
    paused run.
    """

    session_factory: async_sessionmaker[AsyncSession]

    async def save(self, checkpoint: Checkpoint) -> None:
        """
        Persist a checkpoint state.

        Args:
            checkpoint: The checkpoint domain object.
        """
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
        """
        Fetch the most recent checkpoint for a run.

        Args:
            run_id: The run identifier.

        Returns:
            The latest Checkpoint object or None.
        """
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
        """
        Append a tool invocation record.

        Args:
            invocation: The tool invocation domain object.
        """
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
        """
        Append a usage ledger entry.

        Args:
            usage: The usage domain object.
        """
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

    async def list(
        self, tenant_id: str, from_date: Optional[datetime] = None, to_date: Optional[datetime] = None
    ) -> list[UsageLedgerEntry]:
        """
        List usage entries for a tenant within a date range.

        Args:
            tenant_id: The tenant identifier.
            from_date: Optional start datetime.
            to_date: Optional end datetime.

        Returns:
            A list of UsageLedgerEntry objects.
        """
        async with self.session_factory() as s:
            # We need to join with runs to filter by tenant_id, OR if UsageRow has tenant_id.
            # Checking sql.py models... UsageRow does NOT have tenant_id. RunRow has it.
            # So we need to join UsageRow and RunRow.

            stmt = select(UsageRow).join(RunRow, UsageRow.run_id == RunRow.id).where(RunRow.tenant_id == tenant_id)

            if from_date:
                stmt = stmt.where(UsageRow.created_at >= from_date)
            if to_date:
                stmt = stmt.where(UsageRow.created_at <= to_date)

            stmt = stmt.order_by(UsageRow.created_at.desc())

            result = await s.execute(stmt)
            rows = result.scalars().all()

            return [
                UsageLedgerEntry(
                    id=row.id,
                    run_id=row.run_id,
                    provider=row.provider,
                    model=row.model,
                    prompt_tokens=row.prompt_tokens,
                    completion_tokens=row.completion_tokens,
                    total_tokens=row.total_tokens,
                    cost_usd=row.cost_usd,
                    created_at=row.created_at,
                )
                for row in rows
            ]


@dataclass(frozen=True)
class SqlPolicyRepository(PolicyRepository):
    """SQL implementation of ``PolicyRepository``."""

    session_factory: async_sessionmaker[AsyncSession]

    async def get(self, tenant_id: str) -> Optional[dict]:
        async with self.session_factory() as s:
            stmt = select(PolicyRow).where(PolicyRow.tenant_id == tenant_id)
            result = await s.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return row.config

    async def update(self, tenant_id: str, config: dict) -> None:
        async with self.session_factory() as s:
            stmt = select(PolicyRow).where(PolicyRow.tenant_id == tenant_id)
            result = await s.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                import uuid

                row = PolicyRow(
                    id=str(uuid.uuid4()),
                    tenant_id=tenant_id,
                    config=config,
                    created_at=_utc_now_naive(),
                    updated_at=_utc_now_naive(),
                )
                s.add(row)
            else:
                row.config = config
                row.updated_at = _utc_now_naive()
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
    policies: SqlPolicyRepository


def build_sql_repos(*, session_factory: async_sessionmaker[AsyncSession]) -> SqlRepoBundle:
    """Build a ``SqlRepoBundle`` from a session factory."""
    return SqlRepoBundle(
        runs=SqlRunRepository(session_factory=session_factory),
        events=SqlEventRepository(session_factory=session_factory),
        approvals=SqlApprovalRepository(session_factory=session_factory),
        checkpoints=SqlCheckpointRepository(session_factory=session_factory),
        tool_invocations=SqlToolInvocationRepository(session_factory=session_factory),
        usage=SqlUsageRepository(session_factory=session_factory),
        policies=SqlPolicyRepository(session_factory=session_factory),
    )
