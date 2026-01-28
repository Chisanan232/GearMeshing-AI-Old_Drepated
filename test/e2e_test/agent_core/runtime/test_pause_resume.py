# from __future__ import annotations
#
# import pytest
# from testcontainers.postgres import PostgresContainer
#
# from gearmeshing_ai.agent_core.capabilities.registry import CapabilityRegistry
# from gearmeshing_ai.agent_core.factory import build_default_registry
# from gearmeshing_ai.agent_core.policy.global_policy import GlobalPolicy
# from gearmeshing_ai.core.models.domain.policy import PolicyConfig
# from gearmeshing_ai.agent_core.repos.sql import (
#     build_sql_repos,
#     create_all,
#     create_engine,
#     create_sessionmaker,
# )
# from gearmeshing_ai.agent_core.runtime import EngineDeps
# from gearmeshing_ai.agent_core.runtime.engine import AgentEngine
# from gearmeshing_ai.core.models.domain import (
#     AgentRun,
#     ApprovalDecision,
#     CapabilityName,
#     RiskLevel,
# )
#
#
# @pytest.mark.asyncio
# async def test_langgraph_pause_and_resume_round_trip() -> None:
#     with PostgresContainer("postgres:16") as pg:
#         url = pg.get_connection_url()
#         db_url = url.replace("postgresql://", "postgresql+asyncpg://")
#
#         engine = create_engine(db_url)
#         await create_all(engine)
#         session_factory = create_sessionmaker(engine)
#
#         repos = build_sql_repos(session_factory=session_factory)
#         reg: CapabilityRegistry = build_default_registry()
#
#         deps = EngineDeps(
#             runs=repos.runs,
#             events=repos.events,
#             approvals=repos.approvals,
#             checkpoints=repos.checkpoints,
#             tool_invocations=repos.tool_invocations,
#             capabilities=reg,
#         )
#
#         cfg = PolicyConfig()
#         cfg.approval_policy.require_for_risk_at_or_above = RiskLevel.low
#         engine_runtime = AgentEngine(policy=GlobalPolicy(cfg), deps=deps)
#
#         run = AgentRun(role="sre", objective="do risky thing")
#         plan = [{"capability": CapabilityName.shell_exec, "args": {"cmd": "echo hi"}}]
#         await engine_runtime.start_run(run=run, plan=plan)
#
#         cp = await repos.checkpoints.latest(run.id)
#         assert cp is not None
#         approval_id = cp.state.get("awaiting_approval_id")
#         assert approval_id
#
#         await repos.approvals.resolve(approval_id, decision=ApprovalDecision.approved.value, decided_by="tester")
#         await engine_runtime.resume_run(run_id=run.id, approval_id=approval_id)
#
#         updated = await repos.runs.get(run.id)
#         assert updated is not None
#         assert str(getattr(updated.status, "value", updated.status)) == "succeeded"
#
#         await engine.dispose()
