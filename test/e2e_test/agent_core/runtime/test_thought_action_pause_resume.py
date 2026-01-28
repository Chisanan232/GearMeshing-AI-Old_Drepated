from __future__ import annotations

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from sqlalchemy import select
from testcontainers.postgres import PostgresContainer

from gearmeshing_ai.agent_core.factory import build_default_registry
from gearmeshing_ai.agent_core.policy.global_policy import GlobalPolicy
from gearmeshing_ai.core.models.domain.policy import PolicyConfig
from gearmeshing_ai.agent_core.repos.models import EventRow, ToolInvocationRow
from gearmeshing_ai.core.database import (
    create_all,
    create_engine,
    create_sessionmaker,
)
from gearmeshing_ai.agent_core.repos.sql import build_sql_repos
from gearmeshing_ai.agent_core.runtime import EngineDeps
from gearmeshing_ai.agent_core.runtime.engine import AgentEngine
from gearmeshing_ai.core.models.domain import (
    AgentEventType,
    AgentRun,
    AgentRunStatus,
    ApprovalDecision,
    RiskLevel,
)
from gearmeshing_ai.info_provider import CapabilityName


@pytest.mark.asyncio
async def test_mixed_thought_then_action_pause_and_resume_round_trip() -> None:
    with PostgresContainer("postgres:16") as pg:
        db_url = pg.get_connection_url().replace("postgresql://", "postgresql+asyncpg://")
        pool_url = pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")

        engine = create_engine(db_url)
        await create_all(engine)
        session_factory = create_sessionmaker(engine)

        repos = build_sql_repos(session_factory=session_factory)

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

            cfg = PolicyConfig()
            cfg.approval_policy.require_for_risk_at_or_above = RiskLevel.low
            cfg.tool_policy.allowed_capabilities = {CapabilityName.summarize}
            runtime = AgentEngine(policy=GlobalPolicy(cfg), deps=deps)

            run = AgentRun(role="planner", objective="mixed")
            plan = [
                {"kind": "thought", "thought": "design", "args": {"note": "n"}},
                {"kind": "action", "capability": CapabilityName.summarize.value, "args": {"text": "hello"}},
            ]

            await runtime.start_run(run=run, plan=plan)

            # Retrieve checkpoint from the actual saver, not the old repo
            cp = await checkpointer.aget_tuple(config={"configurable": {"thread_id": run.id}})
            assert cp is not None

            # AsyncPostgresSaver returns a CheckpointTuple, state is in cp.checkpoint
            state = cp.checkpoint

            # LangGraph stores state in channel_values
            if "channel_values" in state:
                state = state["channel_values"]

            approval_id = state.get("awaiting_approval_id")
            assert approval_id

            async with session_factory() as s:
                ev = await s.execute(select(EventRow).where(EventRow.run_id == run.id))
                ev_rows = list(ev.scalars().all())
                types = [r.type for r in ev_rows]
                assert AgentEventType.thought_executed.value in types
                assert AgentEventType.artifact_created.value in types
                assert AgentEventType.approval_requested.value in types

                inv = await s.execute(select(ToolInvocationRow).where(ToolInvocationRow.run_id == run.id))
                assert inv.first() is None

            await repos.approvals.resolve(approval_id, decision=ApprovalDecision.approved.value, decided_by="tester")
            await runtime.resume_run(run_id=run.id, approval_id=approval_id)

            updated = await repos.runs.get(run.id)
            assert updated is not None
            assert str(getattr(updated.status, "value", updated.status)) == AgentRunStatus.succeeded.value

            async with session_factory() as s:
                inv = await s.execute(
                    select(ToolInvocationRow)
                    .where(ToolInvocationRow.run_id == run.id)
                    .order_by(ToolInvocationRow.created_at)
                )
                inv_rows = list(inv.scalars().all())
                assert len(inv_rows) == 1
                assert inv_rows[0].tool_name == CapabilityName.summarize.value

            await engine.dispose()
