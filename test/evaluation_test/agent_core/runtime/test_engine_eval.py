from __future__ import annotations

from test.settings import test_settings
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
from gearmeshing_ai.agent_core.policy.models import PolicyConfig
from gearmeshing_ai.agent_core.repos.models import ToolInvocationRow
from gearmeshing_ai.agent_core.repos.sql import (
    build_sql_repos,
    create_all,
    create_engine,
    create_sessionmaker,
)
from gearmeshing_ai.agent_core.runtime import EngineDeps
from gearmeshing_ai.agent_core.runtime.engine import AgentEngine
from gearmeshing_ai.agent_core.schemas.domain import (
    AgentRun,
    AgentRunStatus,
    CapabilityName,
)


def _eval_enabled() -> bool:
    """Check if evaluation tests are enabled via GM_RUN_EVAL_TESTS setting."""
    return test_settings.run_eval_tests


class _DeterministicSummarize(Capability):
    name = CapabilityName.summarize

    async def execute(self, ctx: CapabilityContext, *, args: Dict[str, Any]) -> CapabilityResult:
        text = str(args.get("text") or "")
        return CapabilityResult(ok=True, output={"summary": text[:10]})


@pytest.mark.asyncio
@pytest.mark.slow
async def test_eval_end_to_end_graph_happy_path_selects_and_runs_tools() -> None:
    if not _eval_enabled():
        pytest.skip("set GM_RUN_EVAL_TESTS=1 to enable evaluation tests")

    with PostgresContainer("postgres:16") as pg:
        # testcontainers returns postgresql+psycopg2://user:pass@host:port/db
        # We need to convert it to postgresql+asyncpg:// for SQLAlchemy
        base_url = pg.get_connection_url()
        # Handle both postgresql:// and postgresql+psycopg2:// formats
        if "postgresql+psycopg2://" in base_url:
            db_url = base_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
            psycopg_url = base_url.replace("postgresql+psycopg2://", "postgresql://")
        else:
            db_url = base_url.replace("postgresql://", "postgresql+asyncpg://")
            psycopg_url = base_url

        engine = create_engine(db_url)
        await create_all(engine)
        session_factory = create_sessionmaker(engine)
        repos = build_sql_repos(session_factory=session_factory)

        reg = CapabilityRegistry()
        reg.register(_DeterministicSummarize())

        cfg = PolicyConfig()
        cfg.tool_policy.allowed_capabilities = {CapabilityName.summarize}

        # Create connection pool for AsyncPostgresSaver
        # psycopg_pool expects a plain postgresql:// URL
        pool = AsyncConnectionPool(conninfo=psycopg_url)
        await pool.open()

        try:
            # Setup checkpointer with autocommit mode to allow CREATE INDEX CONCURRENTLY
            async with pool.connection() as conn:
                await conn.set_autocommit(True)
                checkpointer = AsyncPostgresSaver(conn)
                await checkpointer.setup()

            # Create a new checkpointer instance for the engine that uses the pool
            checkpointer = AsyncPostgresSaver(pool)

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

            run = AgentRun(role="eval", objective="Summarize: hello world")
            plan = [
                {"capability": CapabilityName.summarize.value, "args": {"text": "hello world"}},
                {"capability": CapabilityName.summarize.value, "args": {"text": "second step"}},
            ]
            await runtime.start_run(run=run, plan=plan)

            loaded = await repos.runs.get(run.id)
            assert loaded is not None
            assert str(getattr(loaded.status, "value", loaded.status)) == AgentRunStatus.succeeded.value

            # Tool selection/execution: should have exactly two persisted invocations
            # (this is our proxy for "tools selected and run as expected")
            async with session_factory() as s:
                res = await s.execute(select(ToolInvocationRow).where(ToolInvocationRow.run_id == run.id))
                rows = list(res.scalars().all())
                assert len(rows) == 2
                assert all(r.ok for r in rows)

            await engine.dispose()
        finally:
            await pool.close()
