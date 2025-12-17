from __future__ import annotations

import pytest
from testcontainers.postgres import PostgresContainer

from gearmeshing_ai.agent_core.factory import build_default_registry
from gearmeshing_ai.agent_core.planning.planner import StructuredPlanner
from gearmeshing_ai.agent_core.policy.models import PolicyConfig
from gearmeshing_ai.agent_core.repos.sql import build_sql_repos, create_all, create_engine, create_sessionmaker
from gearmeshing_ai.agent_core.runtime import EngineDeps
from gearmeshing_ai.agent_core.schemas.domain import AgentRun
from gearmeshing_ai.agent_core.service import AgentService, AgentServiceDeps


@pytest.mark.asyncio
async def test_full_agent_run_with_real_persistence() -> None:
    with PostgresContainer("postgres:16") as pg:
        db_url = pg.get_connection_url().replace("postgresql://", "postgresql+asyncpg://")

        engine = create_engine(db_url)
        await create_all(engine)
        session_factory = create_sessionmaker(engine)
        repos = build_sql_repos(session_factory=session_factory)

        deps = EngineDeps(
            runs=repos.runs,
            events=repos.events,
            approvals=repos.approvals,
            checkpoints=repos.checkpoints,
            tool_invocations=repos.tool_invocations,
            capabilities=build_default_registry(),
        )

        service = AgentService(
            policy_config=PolicyConfig(),
            deps=AgentServiceDeps(engine_deps=deps, planner=StructuredPlanner()),
        )

        run_id = await service.run(run=AgentRun(role="planner", objective="Summarize this objective"))
        assert run_id

        await engine.dispose()
