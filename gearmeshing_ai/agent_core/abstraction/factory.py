"""Factory for creating AI agent instances.

This module provides a factory pattern implementation for creating and managing
AI agent instances, supporting multiple frameworks through registered implementations.
"""

from typing import Callable, Dict, Optional, Type

from .base import AIAgentBase, AIAgentConfig
from .cache import AIAgentCache


class AIAgentFactory:
    """Factory for creating AI agent instances.

    This factory manages:
    - Registration of agent implementations for different frameworks
    - Agent instance creation
    - Agent caching and lifecycle management
    - Framework-specific initialization

    Usage:
        factory = AIAgentFactory()
        factory.register('pydantic_ai', PydanticAIAgent)
        agent = await factory.create(config)
    """

    def __init__(self, cache: Optional[AIAgentCache] = None) -> None:
        """Initialize the factory.

        Args:
            cache: Optional AIAgentCache instance (creates new if not provided)
        """
        self._implementations: Dict[str, Type[AIAgentBase]] = {}
        self._factories: Dict[str, Callable] = {}
        self._cache = cache or AIAgentCache()

    def register(
        self,
        framework: str,
        implementation: Type[AIAgentBase],
    ) -> None:
        """Register an agent implementation for a framework.

        Args:
            framework: Framework identifier (e.g., 'pydantic_ai', 'langchain')
            implementation: AIAgentBase subclass for this framework

        Raises:
            ValueError: If framework is already registered
        """
        if framework in self._implementations:
            raise ValueError(f"Framework '{framework}' is already registered")
        self._implementations[framework] = implementation

    def register_factory(
        self,
        framework: str,
        factory_func: Callable[[AIAgentConfig], AIAgentBase],
    ) -> None:
        """Register a factory function for creating agents.

        This allows custom creation logic beyond simple class instantiation.

        Args:
            framework: Framework identifier
            factory_func: Callable that takes AIAgentConfig and returns AIAgentBase

        Raises:
            ValueError: If framework is already registered
        """
        if framework in self._factories:
            raise ValueError(f"Factory for framework '{framework}' is already registered")
        self._factories[framework] = factory_func

    def unregister(self, framework: str) -> None:
        """Unregister an agent implementation.

        Args:
            framework: Framework identifier
        """
        self._implementations.pop(framework, None)
        self._factories.pop(framework, None)

    def is_registered(self, framework: str) -> bool:
        """Check if a framework is registered.

        Args:
            framework: Framework identifier

        Returns:
            True if framework is registered, False otherwise
        """
        return framework in self._implementations or framework in self._factories

    async def create(
        self,
        config: AIAgentConfig,
        use_cache: bool = True,
    ) -> AIAgentBase:
        """Create an AI agent instance.

        Args:
            config: AIAgentConfig with agent parameters
            use_cache: Whether to use cached instance if available

        Returns:
            Initialized AIAgentBase instance

        Raises:
            ValueError: If framework is not registered
            RuntimeError: If agent initialization fails
        """
        # Check cache first
        cache_key = f"{config.framework}:{config.name}"
        if use_cache:
            cached = await self._cache.get(cache_key)
            if cached is not None:
                return cached

        # Create new instance
        if config.framework in self._factories:
            agent = self._factories[config.framework](config)
        elif config.framework in self._implementations:
            agent = self._implementations[config.framework](config)
        else:
            raise ValueError(
                f"Framework '{config.framework}' is not registered. Available: {list(self._implementations.keys())}"
            )

        # Initialize agent
        await agent.initialize()

        # Cache the initialized agent
        if use_cache:
            await self._cache.set(cache_key, agent)

        return agent

    async def create_batch(
        self,
        configs: list[AIAgentConfig],
        use_cache: bool = True,
    ) -> list[AIAgentBase]:
        """Create multiple agent instances.

        Args:
            configs: List of AIAgentConfig instances
            use_cache: Whether to use cache

        Returns:
            List of initialized AIAgentBase instances
        """
        agents = []
        for config in configs:
            agent = await self.create(config, use_cache=use_cache)
            agents.append(agent)
        return agents

    async def get_cached(self, name: str, framework: str) -> Optional[AIAgentBase]:
        """Get a cached agent by name and framework.

        Args:
            name: Agent name
            framework: Framework identifier

        Returns:
            Cached agent or None if not found
        """
        cache_key = f"{framework}:{name}"
        return await self._cache.get(cache_key)

    async def clear_cache(self) -> None:
        """Clear all cached agents."""
        await self._cache.clear()

    def get_cache(self) -> AIAgentCache:
        """Get the cache instance."""
        return self._cache

    def get_registered_frameworks(self) -> list[str]:
        """Get list of registered frameworks."""
        frameworks = set(self._implementations.keys()) | set(self._factories.keys())
        return sorted(list(frameworks))

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cache.clear()
