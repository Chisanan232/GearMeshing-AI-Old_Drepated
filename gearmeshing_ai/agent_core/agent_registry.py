from __future__ import annotations

from typing import Callable, Dict

from .schemas.domain import AgentRun
from .service import AgentService

AgentServiceFactory = Callable[[AgentRun], AgentService]


class AgentRegistry:
    def __init__(self) -> None:
        self._factories: Dict[str, AgentServiceFactory] = {}

    def register(self, role: str, factory: AgentServiceFactory) -> None:
        self._factories[str(role)] = factory

    def get(self, role: str) -> AgentServiceFactory:
        try:
            return self._factories[str(role)]
        except KeyError as e:
            raise KeyError(f"unknown role: {role}") from e

    def has(self, role: str) -> bool:
        return str(role) in self._factories
