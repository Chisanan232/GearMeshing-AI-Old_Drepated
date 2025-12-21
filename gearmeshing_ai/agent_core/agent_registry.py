from __future__ import annotations

from typing import Callable, Dict

from .schemas.domain import AgentRun
from .service import AgentService

AgentServiceFactory = Callable[[AgentRun], AgentService]
"""
AgentServiceFactory:
    A callable type that takes an ``AgentRun`` context and returns an initialized
    ``AgentService`` ready for execution.
"""


class AgentRegistry:
    """
    Registry for agent service factories keyed by role.
    
    This registry allows dynamic lookup of agent implementations based on the
    role requested for a specific run. It enables different agent configurations
    or specialized implementations (e.g., 'planner', 'coder', 'researcher') to
    be plugged into the system.
    """
    def __init__(self) -> None:
        """Initialize an empty agent registry."""
        self._factories: Dict[str, AgentServiceFactory] = {}

    def register(self, role: str, factory: AgentServiceFactory) -> None:
        """
        Register a factory for a specific agent role.

        Args:
            role: The role identifier (e.g., 'planner', 'dev').
            factory: A callable that takes an AgentRun and returns an AgentService.
        """
        self._factories[str(role)] = factory

    def get(self, role: str) -> AgentServiceFactory:
        """
        Retrieve the factory for a specific role.

        Args:
            role: The role identifier to look up.

        Returns:
            The registered AgentServiceFactory for the role.

        Raises:
            KeyError: If no factory is registered for the specified role.
        """
        try:
            return self._factories[str(role)]
        except KeyError as e:
            raise KeyError(f"unknown role: {role}") from e

    def has(self, role: str) -> bool:
        """
        Check if a factory exists for the specified role.

        Args:
            role: The role identifier.

        Returns:
            True if the role is registered, False otherwise.
        """
        return str(role) in self._factories
