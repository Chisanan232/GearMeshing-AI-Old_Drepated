"""Provider system for selecting and configuring AI agents.

This module provides a global provider pattern for managing the active AI agent
framework and factory, allowing the entire application to use a unified entry point
for agent creation and configuration.
"""

import os
from typing import Optional

from .base import AIAgentConfig
from .factory import AIAgentFactory


class AIAgentProvider:
    """Global provider for AI agent framework selection and configuration.

    This class manages:
    - The active AI framework for the application
    - The agent factory instance
    - Framework configuration and initialization
    - Entry point for agent creation throughout the application

    Usage:
        provider = AIAgentProvider()
        provider.set_framework('pydantic_ai')
        agent = await provider.create_agent(config)
    """

    def __init__(self) -> None:
        """Initialize the provider."""
        self._framework: Optional[str] = None
        self._factory: Optional[AIAgentFactory] = None

    def set_framework(self, framework: str) -> None:
        """Set the active AI framework.

        Args:
            framework: Framework identifier (e.g., 'pydantic_ai', 'langchain')

        Raises:
            ValueError: If framework is not registered
        """
        if self._factory is None:
            raise RuntimeError("Factory not initialized. Call initialize() first.")

        if not self._factory.is_registered(framework):
            raise ValueError(
                f"Framework '{framework}' is not registered. "
                f"Available: {self._factory.get_registered_frameworks()}"
            )

        self._framework = framework

    def get_framework(self) -> Optional[str]:
        """Get the active framework.

        Returns:
            Framework identifier or None if not set
        """
        return self._framework

    def set_factory(self, factory: AIAgentFactory) -> None:
        """Set the agent factory.

        Args:
            factory: AIAgentFactory instance
        """
        self._factory = factory

    def get_factory(self) -> Optional[AIAgentFactory]:
        """Get the agent factory.

        Returns:
            AIAgentFactory instance or None if not set
        """
        return self._factory

    async def create_agent(self, config: AIAgentConfig, use_cache: bool = True):
        """Create an agent using the active framework.

        Args:
            config: AIAgentConfig with agent parameters
            use_cache: Whether to use cache

        Returns:
            Initialized AIAgentBase instance

        Raises:
            RuntimeError: If framework or factory not set
            ValueError: If framework is not registered
        """
        if self._factory is None:
            raise RuntimeError("Factory not initialized")

        if self._framework is None:
            raise RuntimeError("Framework not set. Call set_framework() first.")

        # Override framework in config to use the active one
        config_copy = AIAgentConfig(
            name=config.name,
            framework=self._framework,
            model=config.model,
            system_prompt=config.system_prompt,
            tools=config.tools,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            metadata=config.metadata,
        )

        return await self._factory.create(config_copy, use_cache=use_cache)

    def get_registered_frameworks(self) -> list[str]:
        """Get list of registered frameworks.

        Returns:
            List of framework identifiers
        """
        if self._factory is None:
            return []
        return self._factory.get_registered_frameworks()

    async def clear_cache(self) -> None:
        """Clear the agent cache."""
        if self._factory is not None:
            await self._factory.clear_cache()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AIAgentProvider(framework={self._framework}, "
            f"registered={self.get_registered_frameworks()})"
        )


# Global provider instance
_global_provider: Optional[AIAgentProvider] = None


def get_agent_provider() -> AIAgentProvider:
    """Get the global AI agent provider.

    Returns:
        Global AIAgentProvider instance

    Raises:
        RuntimeError: If provider not initialized
    """
    global _global_provider
    if _global_provider is None:
        raise RuntimeError(
            "Agent provider not initialized. Call initialize_agent_provider() first."
        )
    return _global_provider


def set_agent_provider(provider: AIAgentProvider) -> None:
    """Set the global AI agent provider.

    Args:
        provider: AIAgentProvider instance
    """
    global _global_provider
    _global_provider = provider


def initialize_agent_provider(
    factory: Optional[AIAgentFactory] = None,
    framework: Optional[str] = None,
) -> AIAgentProvider:
    """Initialize the global AI agent provider.

    This function sets up the global provider with a factory and optionally
    sets the active framework. It should be called once during application startup.

    Args:
        factory: AIAgentFactory instance (creates new if not provided)
        framework: Optional framework to set as active

    Returns:
        Initialized AIAgentProvider instance

    Raises:
        ValueError: If framework is specified but not registered
    """
    global _global_provider

    if factory is None:
        factory = AIAgentFactory()

    provider = AIAgentProvider()
    provider.set_factory(factory)

    # Set framework from parameter or environment variable
    active_framework = framework or os.getenv("AI_AGENT_FRAMEWORK")
    if active_framework:
        provider.set_framework(active_framework)

    _global_provider = provider
    return provider


def reset_agent_provider() -> None:
    """Reset the global agent provider.

    This is mainly useful for testing.
    """
    global _global_provider
    if _global_provider is not None:
        _global_provider = None
