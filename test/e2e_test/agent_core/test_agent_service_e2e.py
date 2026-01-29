from __future__ import annotations

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from testcontainers.postgres import PostgresContainer

from gearmeshing_ai.agent_core.factory import build_default_registry
from gearmeshing_ai.agent_core.planning.planner import StructuredPlanner
from gearmeshing_ai.core.models.domain.policy import PolicyConfig
from gearmeshing_ai.core.database import (
    create_all,
    create_engine,
    create_sessionmaker,
)
from gearmeshing_ai.core.database.repositories.bundle import build_sql_repos
from gearmeshing_ai.agent_core.runtime import EngineDeps
from gearmeshing_ai.core.models.domain import AgentRun
from gearmeshing_ai.agent_core.service import AgentService, AgentServiceDeps


@pytest.mark.asyncio
async def test_full_agent_run_with_real_persistence() -> None:
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

            deps = EngineDeps(
                runs=repos.runs,
                events=repos.events,
                approvals=repos.approvals,
                checkpoints=repos.checkpoints,
                tool_invocations=repos.tool_invocations,
                usage=repos.usage,
                capabilities=build_default_registry(),
                checkpointer=checkpointer,
            )

            service = AgentService(
                policy_config=PolicyConfig(),
                deps=AgentServiceDeps(engine_deps=deps, planner=StructuredPlanner()),
            )

            run_id = await service.run(run=AgentRun(role="planner", objective="Summarize this objective"))
            assert run_id

            await engine.dispose()
