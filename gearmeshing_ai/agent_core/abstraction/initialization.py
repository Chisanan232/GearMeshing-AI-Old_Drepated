"""Initialization system for AI agent abstraction.

This module provides initialization and setup functions for the agent abstraction
layer, including framework registration and provider setup.
"""

from typing import Optional

from gearmeshing_ai.core.logging_config import get_logger

from .adapters import PydanticAIAgent
from .cache import AIAgentCache
from .config import AgentAbstractionConfig
from .factory import AIAgentFactory
from .provider import AIAgentProvider, initialize_agent_provider, set_agent_provider

logger = get_logger(__name__)


def setup_agent_abstraction(
    config: Optional[AgentAbstractionConfig] = None,
) -> AIAgentProvider:
    """Set up the AI agent abstraction layer.

    This function initializes the agent abstraction system with:
    - Agent cache
    - Factory with registered frameworks
    - Global provider
    - Active framework selection

    Args:
        config: Optional AgentAbstractionConfig (loads from env if not provided)

    Returns:
        Initialized AIAgentProvider instance

    Raises:
        ValueError: If configuration is invalid
    """
    if config is None:
        config = AgentAbstractionConfig.from_env()

    logger.info(
        f"Setting up AI agent abstraction layer: "
        f"framework={config.framework}, "
        f"cache_enabled={config.cache_enabled}"
    )

    # Create cache
    cache = None
    if config.cache_enabled:
        cache = AIAgentCache(
            max_size=config.cache_max_size,
            ttl=config.cache_ttl,
        )
        logger.debug(
            f"Agent cache initialized: max_size={config.cache_max_size}, "
            f"ttl={config.cache_ttl}"
        )

    # Create factory
    factory = AIAgentFactory(cache=cache)

    # Register built-in frameworks
    _register_frameworks(factory)

    # Initialize provider
    provider = initialize_agent_provider(
        factory=factory,
        framework=config.framework,
    )

    logger.info(
        f"AI agent abstraction layer initialized. "
        f"Registered frameworks: {provider.get_registered_frameworks()}"
    )

    if config.framework:
        logger.info(f"Active framework: {config.framework}")

    return provider


def _register_frameworks(factory: AIAgentFactory) -> None:
    """Register built-in framework adapters.

    Args:
        factory: AIAgentFactory instance
    """
    try:
        factory.register("pydantic_ai", PydanticAIAgent)
        logger.debug("Registered framework: pydantic_ai")
    except Exception as e:
        logger.warning(f"Failed to register pydantic_ai framework: {e}")

    # Additional frameworks can be registered here as they're implemented
    # factory.register("langchain", LangChainAgent)
    # factory.register("anthropic", AnthropicAgent)


def get_default_provider() -> AIAgentProvider:
    """Get or create the default agent provider.

    This function ensures a provider is available, creating one with
    default configuration if needed.

    Returns:
        AIAgentProvider instance
    """
    try:
        from .provider import get_agent_provider
        return get_agent_provider()
    except RuntimeError:
        logger.debug("Creating default agent provider")
        return setup_agent_abstraction()
