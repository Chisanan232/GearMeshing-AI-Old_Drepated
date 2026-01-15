"""API key validation for AI model providers.

This module provides utilities to validate that required API keys are present
in the runtime environment before initializing AI agents.

Uses regex patterns for flexible model identification across different provider
naming conventions and model versions.

References:
- OpenAI Models: https://platform.openai.com/docs/pricing
- Anthropic Models: https://platform.claude.com/docs/en/about-claude/models/overview
- Google Models: https://ai.google.dev/gemini-api/docs/models
- xAI (Grok) Models: https://docs.x.ai/docs/models
"""

import os
import re
from enum import Enum
from typing import Dict, List, Optional, Pattern

from gearmeshing_ai.core.logging_config import get_logger

logger = get_logger(__name__)


class AIModelProvider(str, Enum):
    """Enumeration of supported AI model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROK = "grok"

    def __str__(self) -> str:
        """Return the string value of the provider."""
        return self.value


# Mapping of AI model providers to their environment variable names
PROVIDER_API_KEY_MAPPING: Dict[AIModelProvider, List[str]] = {
    AIModelProvider.OPENAI: ["OPENAI_API_KEY"],
    AIModelProvider.ANTHROPIC: ["ANTHROPIC_API_KEY"],
    AIModelProvider.GOOGLE: ["GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"],
    AIModelProvider.GROK: ["GROK_API_KEY", "XAI_API_KEY"],
}

# Gateway API key environment variable
GATEWAY_API_KEY_ENV = "PYDANTIC_AI_GATEWAY_API_KEY"

# Regex patterns for identifying AI models by provider
# Based on Pydantic AI model naming conventions: https://ai.pydantic.dev/models/overview/
# Supports direct model names, provider-prefixed formats (e.g., "openai:gpt-4o"),
# and gateway formats (e.g., "gateway/openai:gpt-4o")
# References:
# - OpenAI: https://platform.openai.com/docs/pricing
# - Anthropic: https://platform.claude.com/docs/en/about-claude/models/overview
# - Google: https://ai.google.dev/gemini-api/docs/models
# - xAI: https://docs.x.ai/docs/models
# - Pydantic AI: https://ai.pydantic.dev/models/overview/
# - Pydantic AI Gateway: https://ai.pydantic.dev/gateway/
MODEL_PROVIDER_PATTERNS: Dict[AIModelProvider, Pattern[str]] = {
    # OpenAI models: gpt-* prefix (gpt-4o, gpt-4, gpt-3.5-turbo, gpt-4o-2024-11-20, etc.)
    # Also supports OpenAI-compatible providers: deepseek-*, qwen-*, etc. via OpenAIChatModel
    # Supports formats: gpt-4o, openai:gpt-4o, gateway/openai:gpt-4o
    AIModelProvider.OPENAI: re.compile(
        r"^(?:gateway/)?(?:openai:)?gpt-|^(?:gateway/)?(?:openai:)?(?:deepseek-|qwen-|yi-|llama-|mistral-|command-|cohere\.command)",
        re.IGNORECASE
    ),
    # Anthropic models: claude-* prefix (claude-3, claude-2, claude-instant, claude-3-5-sonnet, etc.)
    # Supports various Claude versions and variants
    # Supports formats: claude-3-opus, anthropic:claude-3-opus, gateway/anthropic:claude-3-opus
    AIModelProvider.ANTHROPIC: re.compile(
        r"^(?:gateway/)?(?:anthropic:)?claude-",
        re.IGNORECASE
    ),
    # Google models: gemini-* prefix (gemini-1.5-pro, gemini-2.0-flash, etc.)
    # Also supports legacy models (palm-2, text-bison) and VertexAI models
    # Supports formats: gemini-2.0-flash, google:gemini-2.0-flash, google-vertex:gemini-2.0-flash, gateway/google-vertex:gemini-2.0-flash
    AIModelProvider.GOOGLE: re.compile(
        r"^(?:gateway/)?(?:google(?:-vertex)?:)?(?:gemini-|palm-|text-|models/gemini-|models/palm-)",
        re.IGNORECASE
    ),
    # xAI Grok models: grok-* prefix (grok-1, grok-2, grok-beta, etc.)
    # Also supports grok via OpenAI-compatible API
    # Supports formats: grok-2, grok:grok-2, gateway/grok:grok-2
    AIModelProvider.GROK: re.compile(
        r"^(?:gateway/)?(?:grok:)?grok-",
        re.IGNORECASE
    ),
}


class APIKeyValidator:
    """Validator for AI provider API keys."""

    @staticmethod
    def get_provider_for_model(model: str) -> Optional[AIModelProvider]:
        """Get the provider for a given model identifier using regex matching.

        Supports flexible model identification across different naming conventions,
        versions, and date suffixes.

        Args:
            model: Model identifier (e.g., 'gpt-4o', 'claude-3-opus', 'gpt-4o-2024-11-20')

        Returns:
            AIModelProvider enum if found, None otherwise
        """
        if not model:
            return None

        model_lower = model.lower()

        # Check each provider's regex pattern
        for provider, pattern in MODEL_PROVIDER_PATTERNS.items():
            if pattern.match(model_lower):
                return provider

        return None

    @staticmethod
    def get_required_api_keys(provider: AIModelProvider) -> List[str]:
        """Get the list of environment variable names for a provider.

        Args:
            provider: AIModelProvider enum value

        Returns:
            List of environment variable names that could contain the API key
        """
        return PROVIDER_API_KEY_MAPPING.get(provider, [])

    @staticmethod
    def has_api_key(provider: AIModelProvider) -> bool:
        """Check if API key is present for a provider.

        Args:
            provider: AIModelProvider enum value

        Returns:
            True if at least one API key environment variable is set, False otherwise
        """
        api_key_vars = APIKeyValidator.get_required_api_keys(provider)

        for var_name in api_key_vars:
            if os.getenv(var_name):
                return True

        return False

    @staticmethod
    def validate_api_key(provider: AIModelProvider) -> None:
        """Validate that API key is present for a provider.

        Args:
            provider: AIModelProvider enum value

        Raises:
            ValueError: If API key is not found for the provider
        """
        if not APIKeyValidator.has_api_key(provider):
            api_key_vars = APIKeyValidator.get_required_api_keys(provider)

            if not api_key_vars:
                raise ValueError(
                    f"Unknown provider: {provider}. "
                    f"Supported providers: {', '.join(p.value for p in AIModelProvider)}"
                )

            raise ValueError(
                f"API key not found for provider '{provider.value}'. "
                f"Please set one of the following environment variables: "
                f"{', '.join(api_key_vars)}"
            )

    @staticmethod
    def validate_model_api_key(model: str) -> None:
        """Validate that API key is present for a given model.

        Args:
            model: Model identifier (e.g., 'gpt-4o', 'claude-3-opus', 'gemini-2.0-flash')

        Raises:
            ValueError: If provider is unknown or API key is not found
        """
        provider = APIKeyValidator.get_provider_for_model(model)

        if provider is None:
            raise ValueError(
                f"Unknown model: {model}. "
                f"Supported models: "
                f"gpt-* (OpenAI), claude-* (Anthropic), "
                f"gemini-*/palm-*/text-* (Google), grok-* (xAI). "
                f"Also supports OpenAI-compatible models: deepseek-*, qwen-*, yi-*, llama-*, mistral-*, command-*, cohere.command-*"
            )

        APIKeyValidator.validate_api_key(provider)

    @staticmethod
    def validate_providers(providers: List[AIModelProvider]) -> Dict[AIModelProvider, bool]:
        """Validate API keys for multiple providers.

        Args:
            providers: List of AIModelProvider enum values

        Returns:
            Dictionary mapping providers to validation status (True = valid, False = missing)
        """
        results = {}

        for provider in providers:
            results[provider] = APIKeyValidator.has_api_key(provider)

        return results

    @staticmethod
    def get_missing_api_keys(providers: List[AIModelProvider]) -> List[AIModelProvider]:
        """Get list of providers missing API keys.

        Args:
            providers: List of AIModelProvider enum values

        Returns:
            List of providers that are missing API keys
        """
        missing = []

        for provider in providers:
            if not APIKeyValidator.has_api_key(provider):
                missing.append(provider)

        return missing

    @staticmethod
    def log_api_key_status(providers: Optional[List[AIModelProvider]] = None) -> None:
        """Log the API key status for providers.

        Args:
            providers: List of AIModelProvider enum values to check. If None, checks all known providers.
        """
        if providers is None:
            providers = list(AIModelProvider)

        logger.debug("API Key Status:")

        for provider in providers:
            has_key = APIKeyValidator.has_api_key(provider)
            status = "✓ Present" if has_key else "✗ Missing"
            logger.debug(f"  {provider.value}: {status}")
