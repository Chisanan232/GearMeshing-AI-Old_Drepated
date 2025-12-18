from __future__ import annotations

from dataclasses import dataclass

from .agent_registry import AgentRegistry
from .schemas.domain import AgentRun
from .service import AgentService


@dataclass(frozen=True)
class Router:
    registry: AgentRegistry
    default_role: str = "planner"
    enable_intent_routing: bool = False

    def _infer_role(self, *, run: AgentRun) -> str:
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
        role = (run.role or "").strip()
        if not role:
            role = self._infer_role(run=run) if self.enable_intent_routing else self.default_role
        if not self.registry.has(role):
            role = self.default_role
        factory = self.registry.get(role)
        return factory(run)
