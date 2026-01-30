from __future__ import annotations

from typing import Any, Dict

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from sqlalchemy import select
from testcontainers.postgres import PostgresContainer

from gearmeshing_ai.agent_core.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityResult,
)
from gearmeshing_ai.agent_core.capabilities.registry import CapabilityRegistry
from gearmeshing_ai.agent_core.policy.global_policy import GlobalPolicy
from gearmeshing_ai.agent_core.runtime import EngineDeps
from gearmeshing_ai.agent_core.runtime.engine import AgentEngine
from gearmeshing_ai.core.database import (
    create_all,
    create_engine,
    create_sessionmaker,
)
from gearmeshing_ai.core.database.entities.agent_events import AgentEvent as EventRow
from gearmeshing_ai.core.database.entities.tool_invocations import (
    ToolInvocation as ToolInvocationRow,
)
from gearmeshing_ai.core.database.repositories.bundle import build_sql_repos
from gearmeshing_ai.core.models.domain import (
    AgentRun,
    AgentRunStatus,
)
from gearmeshing_ai.core.models.domain.policy import PolicyConfig
from gearmeshing_ai.info_provider import CapabilityName


class _OkCapability(Capability):
    name = CapabilityName.summarize

    async def execute(self, ctx: CapabilityContext, *, args: Dict[str, Any]) -> CapabilityResult:
        return CapabilityResult(ok=True, output={"echo": dict(args)})


@pytest.mark.asyncio
async def test_engine_multi_step_happy_path_executes_all_steps_and_finishes() -> None:
    with PostgresContainer("postgres:16") as pg:
        db_url = pg.get_connection_url().replace("postgresql://", "postgresql+asyncpg://")
        pool_url = pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")

        engine = create_engine(db_url)
        await create_all(engine)
        session_factory = create_sessionmaker(engine)
        repos = await build_sql_repos(session_factory=session_factory)

        async with AsyncConnectionPool(conninfo=pool_url, min_size=1, max_size=1, kwargs={"autocommit": True}) as pool:
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()

            reg = CapabilityRegistry()
            reg.register(_OkCapability())

            cfg = PolicyConfig()
            cfg.tool_policy.allowed_capabilities = {CapabilityName.summarize}
            runtime = AgentEngine(
                policy=GlobalPolicy(cfg),
                deps=EngineDeps(
                    runs=repos.runs,
                    events=repos.events,
                    approvals=repos.approvals,
                    checkpoints=repos.checkpoints,
                    tool_invocations=repos.tool_invocations,
                    usage=repos.usage,
                    capabilities=reg,
                    checkpointer=checkpointer,
                ),
            )

            run = AgentRun(role="dev", objective="multi step")
            plan = [
                {"capability": CapabilityName.summarize.value, "args": {"text": "a"}},
                {"capability": CapabilityName.summarize.value, "args": {"text": "b"}},
            ]
            await runtime.start_run(run=run, plan=plan)

            loaded = await repos.runs.get(run.id)
            assert loaded is not None
            assert str(getattr(loaded.status, "value", loaded.status)) == AgentRunStatus.succeeded.value

            async with session_factory() as s:
                inv = await s.execute(
                    select(ToolInvocationRow)
                    .where(ToolInvocationRow.run_id == run.id)
                    .order_by(ToolInvocationRow.created_at)
                )
                inv_rows = list(inv.scalars().all())
                assert len(inv_rows) == 2
                assert [r.args.get("text") for r in inv_rows] == ["a", "b"]

                ev = await s.execute(select(EventRow).where(EventRow.run_id == run.id))
                ev_rows = list(ev.scalars().all())
                # should include capability requested/executed and run completed
                types = [r.type for r in ev_rows]
                assert "capability.executed" in types
                assert "run.completed" in types

            await engine.dispose()


@pytest.mark.asyncio
async def test_engine_empty_plan_happy_path_finishes_immediately() -> None:
    with PostgresContainer("postgres:16") as pg:
        db_url = pg.get_connection_url().replace("postgresql://", "postgresql+asyncpg://")
        pool_url = pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")

        engine = create_engine(db_url)
        await create_all(engine)
        session_factory = create_sessionmaker(engine)
        repos = await build_sql_repos(session_factory=session_factory)

        async with AsyncConnectionPool(conninfo=pool_url, min_size=1, max_size=1, kwargs={"autocommit": True}) as pool:
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()

            reg = CapabilityRegistry()
            reg.register(_OkCapability())

            cfg = PolicyConfig()
            cfg.tool_policy.allowed_capabilities = {CapabilityName.summarize}
            runtime = AgentEngine(
                policy=GlobalPolicy(cfg),
                deps=EngineDeps(
                    runs=repos.runs,
                    events=repos.events,
                    approvals=repos.approvals,
                    checkpoints=repos.checkpoints,
                    tool_invocations=repos.tool_invocations,
                    capabilities=reg,
                    usage=repos.usage,
                    checkpointer=checkpointer,
                ),
            )

            run = AgentRun(role="dev", objective="empty")
            await runtime.start_run(run=run, plan=[])

            loaded = await repos.runs.get(run.id)
            assert loaded is not None
            assert str(getattr(loaded.status, "value", loaded.status)) == AgentRunStatus.succeeded.value

            async with session_factory() as s:
                inv = await s.execute(select(ToolInvocationRow).where(ToolInvocationRow.run_id == run.id))
                assert inv.first() is None

            await engine.dispose()


@pytest.mark.asyncio
async def test_engine_blocked_capability_results_in_failed_run() -> None:
    with PostgresContainer("postgres:16") as pg:
        db_url = pg.get_connection_url().replace("postgresql://", "postgresql+asyncpg://")
        pool_url = pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")

        engine = create_engine(db_url)
        await create_all(engine)
        session_factory = create_sessionmaker(engine)
        repos = await build_sql_repos(session_factory=session_factory)

        async with AsyncConnectionPool(conninfo=pool_url, min_size=1, max_size=1, kwargs={"autocommit": True}) as pool:
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()

            reg = CapabilityRegistry()
            reg.register(_OkCapability())

            cfg = PolicyConfig()
            cfg.tool_policy.allowed_capabilities = {CapabilityName.summarize}
            runtime = AgentEngine(
                policy=GlobalPolicy(cfg),
                deps=EngineDeps(
                    runs=repos.runs,
                    events=repos.events,
                    approvals=repos.approvals,
                    checkpoints=repos.checkpoints,
                    tool_invocations=repos.tool_invocations,
                    usage=repos.usage,
                    capabilities=reg,
                    checkpointer=checkpointer,
                ),
            )

            run = AgentRun(role="dev", objective="do blocked")
            plan = [{"capability": CapabilityName.shell_exec.value, "args": {"cmd": "echo hi"}}]
            await runtime.start_run(run=run, plan=plan)

            loaded = await repos.runs.get(run.id)
            assert loaded is not None
            assert str(getattr(loaded.status, "value", loaded.status)) == AgentRunStatus.failed.value

            await engine.dispose()


@pytest.mark.asyncio
async def test_engine_too_large_args_results_in_failed_run() -> None:
    with PostgresContainer("postgres:16") as pg:
        db_url = pg.get_connection_url().replace("postgresql://", "postgresql+asyncpg://")
        pool_url = pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")

        engine = create_engine(db_url)
        await create_all(engine)
        session_factory = create_sessionmaker(engine)
        repos = await build_sql_repos(session_factory=session_factory)

        async with AsyncConnectionPool(conninfo=pool_url, min_size=1, max_size=1, kwargs={"autocommit": True}) as pool:
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()

            reg = CapabilityRegistry()
            reg.register(_OkCapability())

            cfg = PolicyConfig()
            cfg.tool_policy.allowed_capabilities = {CapabilityName.summarize}
            cfg.safety_policy.max_tool_args_bytes = 1
            runtime = AgentEngine(
                policy=GlobalPolicy(cfg),
                deps=EngineDeps(
                    runs=repos.runs,
                    events=repos.events,
                    approvals=repos.approvals,
                    checkpoints=repos.checkpoints,
                    tool_invocations=repos.tool_invocations,
                    capabilities=reg,
                    checkpointer=checkpointer,
                ),
            )

            run = AgentRun(role="dev", objective="too big args")
            plan = [{"capability": CapabilityName.summarize.value, "args": {"text": "xx"}}]
            await runtime.start_run(run=run, plan=plan)

            loaded = await repos.runs.get(run.id)
            assert loaded is not None
            assert str(getattr(loaded.status, "value", loaded.status)) == AgentRunStatus.failed.value

            await engine.dispose()
