from __future__ import annotations

from dataclasses import dataclass

from .agent_registry import AgentRegistry
from .schemas.domain import AgentRun
from .service import AgentService


@dataclass(frozen=True)
class Router:
    """
    Routes agent runs to the appropriate service factory based on role or intent.

    The router acts as the entry point for dispatching agent runs. It resolves
    the appropriate ``AgentService`` to handle a run by inspecting the requested
    role. If intent routing is enabled and no role is specified, it attempts to
    infer the best role from the run's objective.

    Attributes:
        registry: The ``AgentRegistry`` containing available agent factories.
        default_role: The role to use if no specific role is requested or inferred (default: 'planner').
        enable_intent_routing: Whether to attempt role inference from the objective (default: False).
    """

    registry: AgentRegistry
    default_role: str = "planner"
    enable_intent_routing: bool = False

    def _infer_role(self, *, run: AgentRun) -> str:
        """
        Infer the agent role based on keywords in the run's objective.

        Args:
            run: The agent run containing the objective.

        Returns:
            A string representing the inferred role (e.g., 'dev', 'qa', 'sre', 'market'),
            or the default role if no keywords match.
        """
        obj = (run.objective or "").lower()
        if any(k in obj for k in ("bug", "fix", "refactor", "implement", "code", "pr", "pull request")):
            return "dev"
        if any(k in obj for k in ("test", "pytest", "unit test", "integration test", "e2e", "qa")):
            return "qa"
        if any(k in obj for k in ("deploy", "kubernetes", "incident", "on-call", "sre", "monitor")):
            return "sre"
        if any(k in obj for k in ("market", "pricing", "competitor", "go-to-market")):
            return "market"
        return self.default_role

    def route(self, *, run: AgentRun) -> AgentService:
        """
        Determine and instantiate the appropriate AgentService for a run.

        This method resolves the target role using the following precedence:
        1. Explicitly requested role in ``run.role``.
        2. Inferred role from ``run.objective`` (if ``enable_intent_routing`` is True).
        3. ``default_role``.

        If the resolved role is not found in the registry, it falls back to the default role.

        Args:
            run: The agent run to be routed.

        Returns:
            An instantiated ``AgentService`` ready to execute the run.
        """
        role = (run.role or "").strip()
        if not role:
            role = self._infer_role(run=run) if self.enable_intent_routing else self.default_role
        if not self.registry.has(role):
            role = self.default_role
        factory = self.registry.get(role)
        return factory(run)
