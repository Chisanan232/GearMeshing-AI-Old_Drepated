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
from .policy.provider import PolicyProvider
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

    def _policy_for_run(self, run: AgentRun) -> PolicyConfig:
        """
        Resolve the effective policy configuration for a specific run.

        Combines the base policy (or tenant-specific policy if a provider is present)
        with run-specific overrides like the autonomy profile.

        Args:
            run: The agent run context.

        Returns:
            A PolicyConfig object tailored for this run.
        """
        cfg = self._policy_provider.get(run) if self._policy_provider is not None else self._policy_config
        cfg = cfg.model_copy(deep=True)
        cfg.autonomy_profile = run.autonomy_profile
        return cfg

    async def run(self, *, run: AgentRun) -> str:
        """Plan and execute a run.

        Returns
        -------
        str
            The run id.
        """
        from .factory import build_engine

        plan = await self._deps.planner.plan(objective=run.objective, role=run.role)
        engine = build_engine(policy_config=self._policy_for_run(run), deps=self._deps.engine_deps)
        return await engine.start_run(run=run, plan=plan)

    async def resume(self, *, run_id: str, approval_id: str) -> None:
        """Resume a paused run after approval."""
        from .factory import build_engine

        run = None
        runs_repo = self._deps.engine_deps.runs
        get_run = getattr(runs_repo, "get", None)
        if callable(get_run):
            run = await get_run(run_id)
        cfg = self._policy_for_run(run) if run is not None else self._policy_config
        engine = build_engine(policy_config=cfg, deps=self._deps.engine_deps)
        await engine.resume_run(run_id=run_id, approval_id=approval_id)
