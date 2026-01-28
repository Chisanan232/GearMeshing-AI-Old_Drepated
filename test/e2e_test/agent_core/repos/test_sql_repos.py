from __future__ import annotations

from datetime import datetime, timezone
from typing import AsyncIterator, Iterator

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
)
from testcontainers.postgres import PostgresContainer

from gearmeshing_ai.agent_core.repos.models import (
    ApprovalRow,
    CheckpointRow,
    EventRow,
    RunRow,
    ToolInvocationRow,
    UsageRow,
)
from gearmeshing_ai.agent_core.repos.sql import (
    build_sql_repos,
    create_all,
    create_engine,
    create_sessionmaker,
)
from gearmeshing_ai.core.models.domain import (
    AgentEvent,
    AgentEventType,
    AgentRun,
    Approval,
    ApprovalDecision,
    Checkpoint,
    RiskLevel,
    ToolInvocation,
    UsageLedgerEntry,
)
from gearmeshing_ai.info_provider import CapabilityName


@pytest.fixture
def pg_url() -> Iterator[str]:
    with PostgresContainer("postgres:16") as pg:
        yield pg.get_connection_url()


@pytest_asyncio.fixture
async def engine(pg_url: str) -> AsyncIterator[AsyncEngine]:
    eng = create_engine(pg_url)
    await create_all(eng)
    await create_all(eng)
    yield eng
    await eng.dispose()


@pytest.fixture
def session_factory(engine: AsyncEngine):
    return create_sessionmaker(engine)


@pytest.fixture
def repos(session_factory):
    return build_sql_repos(session_factory=session_factory)


@pytest_asyncio.fixture
async def run(repos) -> AgentRun:
    r = AgentRun(role="dev", objective="ship it")
    await repos.runs.create(r)
    return r


@pytest.mark.asyncio
async def test_runs_create_get_and_update_status(repos, run: AgentRun) -> None:
    loaded = await repos.runs.get(run.id)
    assert loaded is not None
    assert loaded.id == run.id
    assert loaded.role == "dev"
    assert loaded.objective == "ship it"

    await repos.runs.update_status(run.id, status="succeeded")
    updated = await repos.runs.get(run.id)
    assert updated is not None
    assert str(getattr(updated.status, "value", updated.status)) == "succeeded"


@pytest.mark.asyncio
async def test_runs_update_status_missing_is_noop(repos) -> None:
    await repos.runs.update_status("missing", status="failed")


@pytest.mark.asyncio
async def test_events_append_persists(session_factory, repos, run: AgentRun) -> None:
    event = AgentEvent(run_id=run.id, type=AgentEventType.run_started, payload={"k": "v"})
    await repos.events.append(event)

    async with session_factory() as s:
        res = await s.execute(select(EventRow).where(EventRow.run_id == run.id))
        assert res.scalar_one_or_none() is not None


@pytest.mark.asyncio
async def test_tool_invocations_append_persists(session_factory, repos, run: AgentRun) -> None:
    invocation = ToolInvocation(
        run_id=run.id,
        server_id="srv",
        tool_name="echo",
        args={"x": 1},
        ok=True,
        result={"y": 2},
        risk=RiskLevel.low,
    )
    await repos.tool_invocations.append(invocation)

    async with session_factory() as s:
        res = await s.execute(select(ToolInvocationRow).where(ToolInvocationRow.run_id == run.id))
        assert res.scalar_one_or_none() is not None


@pytest.mark.asyncio
async def test_usage_append_persists(session_factory, repos, run: AgentRun) -> None:
    usage = UsageLedgerEntry(
        run_id=run.id,
        provider="mock",
        model="m",
        prompt_tokens=1,
        completion_tokens=2,
        total_tokens=3,
    )
    await repos.usage.append(usage)

    async with session_factory() as s:
        res = await s.execute(select(UsageRow).where(UsageRow.run_id == run.id))
        assert res.scalar_one_or_none() is not None


@pytest.mark.asyncio
async def test_checkpoints_save_and_latest(session_factory, repos, run: AgentRun) -> None:
    now = datetime.now(timezone.utc)
    cp1 = Checkpoint(run_id=run.id, node="n1", state={"a": 1}, created_at=now)
    cp2 = Checkpoint(
        run_id=run.id,
        node="n2",
        state={"a": 2},
        created_at=now.replace(microsecond=now.microsecond + 1),
    )
    await repos.checkpoints.save(cp1)
    await repos.checkpoints.save(cp2)

    latest = await repos.checkpoints.latest(run.id)
    assert latest is not None
    assert latest.node == "n2"
    assert latest.state == {"a": 2}

    async with session_factory() as s:
        res = await s.execute(select(CheckpointRow).where(CheckpointRow.run_id == run.id))
        assert res.first() is not None


@pytest.mark.asyncio
async def test_checkpoints_latest_returns_none_when_missing(repos, run: AgentRun) -> None:
    missing = await repos.checkpoints.latest(run.id)
    assert missing is None


@pytest.mark.asyncio
async def test_approvals_create_get_and_resolve(session_factory, repos, run: AgentRun) -> None:
    approval = Approval(
        run_id=run.id,
        risk=RiskLevel.medium,
        capability=CapabilityName.mcp_call,
        reason="needs approval",
    )
    await repos.approvals.create(approval)

    approval_loaded = await repos.approvals.get(approval.id)
    assert approval_loaded is not None
    assert str(getattr(approval_loaded.risk, "value", approval_loaded.risk)) == RiskLevel.medium.value
    assert (
        str(getattr(approval_loaded.capability, "value", approval_loaded.capability)) == CapabilityName.mcp_call.value
    )
    assert approval_loaded.decision is None

    await repos.approvals.resolve(approval.id, decision=ApprovalDecision.approved.value, decided_by="tester")
    approval_resolved = await repos.approvals.get(approval.id)
    assert approval_resolved is not None
    assert (
        str(getattr(approval_resolved.decision, "value", approval_resolved.decision)) == ApprovalDecision.approved.value
    )
    assert approval_resolved.decided_by == "tester"
    assert approval_resolved.decided_at is not None

    async with session_factory() as s:
        res = await s.execute(select(ApprovalRow).where(ApprovalRow.run_id == run.id))
        assert res.scalar_one_or_none() is not None


@pytest.mark.asyncio
async def test_approvals_get_returns_none_when_missing(repos) -> None:
    missing = await repos.approvals.get("missing")
    assert missing is None


@pytest.mark.asyncio
async def test_approvals_resolve_missing_is_noop(repos) -> None:
    await repos.approvals.resolve("missing", decision=ApprovalDecision.rejected.value, decided_by=None)


@pytest.mark.asyncio
async def test_run_row_persists(session_factory, run: AgentRun) -> None:
    async with session_factory() as s:
        row_run = await s.get(RunRow, run.id)
        assert row_run is not None
