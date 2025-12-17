from __future__ import annotations

"""High-level orchestration service for agent runs.

``AgentService`` provides an application-friendly API for executing agent runs
without needing to manually wire the planner and runtime engine.

Workflow
--------

- ``run``:

  1. Uses ``StructuredPlanner`` to generate a plan (mixed Thought/Action).
  2. Builds an ``AgentEngine`` using the provided policy configuration.
  3. Starts execution and returns the run id.

- ``resume``:

  1. Builds an ``AgentEngine``.
  2. Resumes execution from the latest checkpoint once an approval is
     resolved.

``AgentService`` is intentionally thin: it delegates execution semantics to
the engine and does not contain policy logic itself.
"""

from dataclasses import dataclass

from .planning.planner import StructuredPlanner
from .policy.models import PolicyConfig
from .runtime import EngineDeps
from .schemas.domain import AgentRun


@dataclass(frozen=True)
class AgentServiceDeps:
    """Dependency bundle for ``AgentService``.

    This allows applications and tests to inject:

    - persistence/capability dependencies for the runtime engine,
    - a planner implementation.
    """
    engine_deps: EngineDeps
    planner: StructuredPlanner


class AgentService:
    """Orchestrate planning + execution for a single agent run."""
    def __init__(self, *, policy_config: PolicyConfig, deps: AgentServiceDeps) -> None:
        self._policy_config = policy_config
        self._deps = deps

    async def run(self, *, run: AgentRun) -> str:
        """Plan and execute a run.

        Returns
        -------
        str
            The run id.
        """
        from .factory import build_engine

        plan = await self._deps.planner.plan(objective=run.objective, role=run.role)
        engine = build_engine(policy_config=self._policy_config, deps=self._deps.engine_deps)
        return await engine.start_run(run=run, plan=plan)

    async def resume(self, *, run_id: str, approval_id: str) -> None:
        """Resume a paused run after approval."""
        from .factory import build_engine

        engine = build_engine(policy_config=self._policy_config, deps=self._deps.engine_deps)
        await engine.resume_run(run_id=run_id, approval_id=approval_id)
