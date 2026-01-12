"""Initialization system for AI agent abstraction.

This module provides initialization and setup functions for the agent abstraction
layer, including framework registration, provider setup, and API key validation.
"""

from typing import Optional

from gearmeshing_ai.core.logging_config import get_logger

from .adapters import PydanticAIAgent
from .api_key_validator import AIModelProvider, APIKeyValidator
from .cache import AIAgentCache
from .config import AgentAbstractionConfig
from .factory import AIAgentFactory
from .provider import AIAgentProvider, initialize_agent_provider

logger = get_logger(__name__)


def setup_agent_abstraction(
    config: Optional[AgentAbstractionConfig] = None,
    validate_api_keys: bool = True,
) -> AIAgentProvider:
    """Set up the AI agent abstraction layer.

    This function initializes the agent abstraction system with:
    - Agent cache
    - Factory with registered frameworks
    - Global provider
    - Active framework selection
    - Optional API key validation

    Args:
        config: Optional AgentAbstractionConfig (loads from env if not provided)
        validate_api_keys: Whether to validate API keys for known providers (default: True)

    Returns:
        Initialized AIAgentProvider instance

    Raises:
        ValueError: If configuration is invalid or required API keys are missing
    """
    if config is None:
        config = AgentAbstractionConfig()

    logger.info(
        f"Setting up AI agent abstraction layer: "
        f"framework={config.framework}, "
        f"cache_enabled={config.cache_enabled}"
    )

    # Validate API keys if requested
    if validate_api_keys:
        _validate_api_keys_for_setup()

    # Create cache
    cache = None
    if config.cache_enabled:
        cache = AIAgentCache(
            max_size=config.cache_max_size,
            ttl=config.cache_ttl,
        )
        logger.debug(f"Agent cache initialized: max_size={config.cache_max_size}, ttl={config.cache_ttl}")

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
        f"AI agent abstraction layer initialized. Registered frameworks: {provider.get_registered_frameworks()}"
    )

    if config.framework:
        logger.info(f"Active framework: {config.framework}")

    return provider


def _validate_api_keys_for_setup() -> None:
    """Validate that at least one API key is present for known providers.

    This function checks for API keys for OpenAI, Anthropic, Google, and Grok.
    It logs the status of each provider and raises an error if no API keys are found.

    Raises:
        ValueError: If no API keys are found for any known provider
    """
    providers = list(AIModelProvider)

    # Log API key status for all providers
    APIKeyValidator.log_api_key_status(providers)

    # Check if at least one provider has an API key
    missing_providers = APIKeyValidator.get_missing_api_keys(providers)

    if len(missing_providers) == len(providers):
        # All providers are missing API keys
        raise ValueError(
            "No API keys found for any supported AI provider. "
            "Please set at least one of the following environment variables:\n"
            "  - OpenAI: OPENAI_API_KEY\n"
            "  - Anthropic: ANTHROPIC_API_KEY\n"
            "  - Google: GOOGLE_API_KEY or GOOGLE_GENERATIVE_AI_API_KEY\n"
            "  - Grok: GROK_API_KEY or XAI_API_KEY"
        )

    # Log which providers have API keys
    available_providers = [p.value for p in providers if p not in missing_providers]
    logger.info(f"Available AI providers: {', '.join(available_providers)}")


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
