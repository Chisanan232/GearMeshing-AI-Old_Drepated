from __future__ import annotations

"""Structured planning for agent runs.

This module defines the default planner used by ``AgentService``.

Responsibilities
----------------

- Convert an input pair (``objective``, ``role``) into a *plan*.
- Emit plan steps as dictionaries that conform to the step schemas in
  ``gearmeshing_ai.agent_core.planning.steps``.

The planner is intentionally constrained:

- It does not execute tools or capabilities.
- It does not decide approvals.
- It only emits structured steps.

The runtime enforces the thought/action split:

- Thought steps are LLM-only and cannot contain tool fields.
- Action steps are side-effecting and are routed through policy/approval.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic_ai import Agent

from .steps import ActionStep, ThoughtStep

logger = logging.getLogger(__name__)


class StructuredPlanner:
    """Planner that produces structured step dictionaries.

    The planner supports two modes:

    - ``model=None``: deterministic fallback that emits a single ``ThoughtStep``.
      This is useful for tests or deployments that want to avoid LLM calls.
    - ``model!=None``: uses Pydantic AI to produce a list of ``ActionStep``
      objects, then returns them as dictionaries.

    The returned list is suitable for passing directly into
    ``AgentEngine.start_run(plan=...)``.
    """

    def __init__(self, *, model: Any | None = None, role: Optional[str] = None, tenant_id: Optional[str] = None) -> None:
        """
        Initialize the planner.

        Args:
            model: The language model instance to use for generation (e.g., from LangChain or PydanticAI).
                   If None, the planner operates in a deterministic "summary-only" mode.
            role: Optional role name for configuration-based model creation.
            tenant_id: Optional tenant identifier for tenant-specific model configuration.
        """
        self._model = model
        self._role = role
        self._tenant_id = tenant_id
        self._model_creation_deferred = False
        
        # Note: Model creation from role is deferred to async context (plan method)
        # This avoids blocking the constructor and async/sync session mismatches

    async def plan(self, *, objective: str, role: str) -> List[Dict[str, Any]]:
        """Generate a plan for a run.

        Parameters
        ----------
        objective:
            The user objective, problem statement, or task description.
        role:
            The role/persona for planning context.

        Returns
        -------
        list[dict[str, Any]]
            A JSON-serializable list of plan step dictionaries.
        """
        # Try to create model from role if not already provided
        model = self._model
        if model is None and self._role is not None and not self._model_creation_deferred:
            try:
                from ..model_provider import async_create_model_for_role
                model = await async_create_model_for_role(self._role, tenant_id=self._tenant_id)
                logger.debug(f"Created planner model for role '{self._role}' from configuration")
                self._model = model
            except Exception as e:
                logger.debug(f"Could not create model for role '{self._role}': {e}")
                self._model_creation_deferred = True
                # Fall back to deterministic mode
                model = None
        
        if model is None:
            return [ThoughtStep(thought="summarize", args={"text": objective, "role": role}).model_dump()]

        agent: Agent = Agent(
            model,
            output_type=List[ActionStep],
            system_prompt=(
                "You are an expert planner for an autonomous software engineering agent. "
                "Return a minimal, safe sequence of action steps as JSON."
            ),
        )

        result = await agent.run(
            (
                "Create a short plan for this objective. "
                "Use only the supported capabilities.\n\n"
                f"role={role}\n"
                f"objective={objective}\n"
            )
        )
        steps = result.output
        return [s.model_dump() for s in steps]
