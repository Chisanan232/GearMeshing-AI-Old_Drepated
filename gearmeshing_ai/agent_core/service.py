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

import logging
from dataclasses import dataclass

from .planning.planner import StructuredPlanner
from gearmeshing_ai.core.models.domain.policy import PolicyConfig
from .policy.provider import PolicyProvider
from .runtime import EngineDeps
from gearmeshing_ai.core.models.domain import AgentRun

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        *,
        policy_config: PolicyConfig,
        deps: AgentServiceDeps,
        policy_provider: PolicyProvider | None = None,
    ) -> None:
        self._policy_config = policy_config
        self._policy_provider = policy_provider
        self._deps = deps

    async def _policy_for_run(self, run: AgentRun) -> PolicyConfig:
        """
        Resolve the effective policy configuration for a specific run.

        Combines the base policy (or tenant-specific policy if a provider is present)
        with run-specific overrides like the autonomy profile.

        Args:
            run: The agent run context.

        Returns:
            A PolicyConfig object tailored for this run.
        """
        cfg = await self._policy_provider.get(run) if self._policy_provider is not None else self._policy_config
        cfg = cfg.model_copy(deep=True)
        cfg.autonomy_profile = run.autonomy_profile
        return cfg

    async def run(self, *, run: AgentRun) -> str:
        """Plan and execute a run with LangSmith tracing.

        This method is traced by LangSmith to capture:
        - Agent run initialization
        - Planning phase execution
        - Engine execution and all nested operations

        Returns
        -------
        str
            The run id.
        """
        from gearmeshing_ai.core.monitoring import get_traceable_decorator

        from .factory import build_engine

        # Get the traceable decorator (no-op if LangSmith unavailable)
        traceable = get_traceable_decorator()

        @traceable(name="agent_run", tags=["agent", "planning", "execution"])
        async def _execute_run():
            plan = await self._deps.planner.plan(objective=run.objective, role=run.role)
            policy_config = await self._policy_for_run(run)
            engine = build_engine(policy_config=policy_config, deps=self._deps.engine_deps)
            return await engine.start_run(run=run, plan=plan)

        return await _execute_run()

    async def resume(self, *, run_id: str, approval_id: str) -> None:
        """Resume a paused run after approval with LangSmith tracing.

        This method is traced by LangSmith to capture:
        - Approval resolution
        - State restoration
        - Resumed execution

        Args:
            run_id: The run identifier
            approval_id: The approval identifier
        """
        from gearmeshing_ai.core.monitoring import get_traceable_decorator

        from .factory import build_engine

        # Get the traceable decorator (no-op if LangSmith unavailable)
        traceable = get_traceable_decorator()

        @traceable(name="agent_resume", tags=["agent", "approval", "resume"])
        async def _execute_resume():
            run = None
            runs_repo = self._deps.engine_deps.runs
            get_run = getattr(runs_repo, "get", None)
            if callable(get_run):
                run = await get_run(run_id)
            cfg = await self._policy_for_run(run) if run is not None else self._policy_config
            engine = build_engine(policy_config=cfg, deps=self._deps.engine_deps)
            await engine.resume_run(run_id=run_id, approval_id=approval_id)

        await _execute_resume()
