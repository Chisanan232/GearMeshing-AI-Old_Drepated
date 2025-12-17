from __future__ import annotations

from dataclasses import dataclass

from .agent_registry import AgentRegistry
from .schemas.domain import AgentRun
from .service import AgentService


@dataclass(frozen=True)
class Router:
    registry: AgentRegistry
    default_role: str = "planner"

    def route(self, *, run: AgentRun) -> AgentService:
        role = (run.role or self.default_role).strip() or self.default_role
        factory = self.registry.get(role)
        return factory(run)
