from __future__ import annotations

from dataclasses import dataclass

from .planning.planner import StructuredPlanner
from .policy.models import PolicyConfig
from .runtime import EngineDeps
from .schemas.domain import AgentRun


@dataclass(frozen=True)
class AgentServiceDeps:
    engine_deps: EngineDeps
    planner: StructuredPlanner


class AgentService:
    def __init__(self, *, policy_config: PolicyConfig, deps: AgentServiceDeps) -> None:
        self._policy_config = policy_config
        self._deps = deps

    async def run(self, *, run: AgentRun) -> str:
        from .factory import build_engine

        plan = await self._deps.planner.plan(objective=run.objective, role=run.role)
        engine = build_engine(policy_config=self._policy_config, deps=self._deps.engine_deps)
        return await engine.start_run(run=run, plan=plan)

    async def resume(self, *, run_id: str, approval_id: str) -> None:
        from .factory import build_engine

        engine = build_engine(policy_config=self._policy_config, deps=self._deps.engine_deps)
        await engine.resume_run(run_id=run_id, approval_id=approval_id)
