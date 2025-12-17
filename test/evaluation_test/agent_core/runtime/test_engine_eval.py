from __future__ import annotations

import os
from typing import Any, Dict

import pytest
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
    v = str(os.getenv("GM_RUN_EVAL_TESTS", "")).strip().lower()
    return v in {"1", "true", "yes", "on"}


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
        db_url = pg.get_connection_url().replace("postgresql://", "postgresql+asyncpg://")

        engine = create_engine(db_url)
        await create_all(engine)
        session_factory = create_sessionmaker(engine)
        repos = build_sql_repos(session_factory=session_factory)

        reg = CapabilityRegistry()
        reg.register(_DeterministicSummarize())

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

        # Path selection: should not pause for approval
        cp = await repos.checkpoints.latest(run.id)
        assert cp is not None
        assert cp.state.get("awaiting_approval_id") is None

        # Tool selection/execution: should have exactly two persisted invocations
        # (this is our proxy for "tools selected and run as expected")
        async with session_factory() as s:
            res = await s.execute(select(ToolInvocationRow).where(ToolInvocationRow.run_id == run.id))
            rows = list(res.scalars().all())
            assert len(rows) == 2
            assert all(r.ok for r in rows)

        await engine.dispose()
