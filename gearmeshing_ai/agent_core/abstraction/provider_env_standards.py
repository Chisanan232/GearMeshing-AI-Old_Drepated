"""
Provider Environment Variable Standards Module.

This module handles exporting API keys from the internal settings model
to official provider environment variables.

It provides:
- Official environment variable naming standards for each AI provider
- Functions to export API keys from settings to official env vars
- Support for all major AI providers (OpenAI, Anthropic, Google, xAI/Grok)

The module ensures consistency between internal configuration and external
library expectations by mapping internal secrets to official provider env vars.

References:
- OpenAI: https://platform.openai.com/docs/quickstart
- Anthropic: https://docs.anthropic.com/en/api/getting-started
- Google: https://ai.google.dev/gemini-api/docs/api-key
- xAI (Grok): https://docs.x.ai/docs/api
"""

import os
from typing import TYPE_CHECKING, Dict, List, Optional

from pydantic import BaseModel, Field

from gearmeshing_ai.agent_core.abstraction.api_key_validator import AIModelProvider

if TYPE_CHECKING:
    from gearmeshing_ai.server.core.config import BaseAISetting, Settings


class ProviderEnvStandard(BaseModel):
    """Data model describing environment variable standards for an AI provider.

    Attributes:
        primary_env_var: The primary official environment variable name for the provider
        alternative_env_vars: List of alternative environment variable names the provider accepts
        description: Human-readable description of the API key
        official_docs: URL to the provider's official documentation
        supported_models: List of supported model names for this provider
    """

    primary_env_var: str = Field(description="Primary official environment variable name (e.g., 'OPENAI_API_KEY')")
    alternative_env_vars: List[str] = Field(default_factory=list, description="Alternative environment variable names")
    description: str = Field(description="Description of the API key and its purpose")
    official_docs: str = Field(description="URL to official provider documentation")
    supported_models: List[str] = Field(description="List of supported model names")


class ProviderEnvStandards(BaseModel):
    """Data model containing environment variable standards for all AI providers.

    This model serves as a container for all provider standards, providing
    type-safe access to provider-specific environment variable information.

    Attributes:
        openai: OpenAI provider standards
        anthropic: Anthropic provider standards
        google: Google provider standards
        grok: xAI (Grok) provider standards
    """

    openai: ProviderEnvStandard = Field(description="OpenAI environment variable standards")
    anthropic: ProviderEnvStandard = Field(description="Anthropic environment variable standards")
    google: ProviderEnvStandard = Field(description="Google environment variable standards")
    grok: ProviderEnvStandard = Field(description="xAI (Grok) environment variable standards")


# Official environment variable standards for each AI provider
# These are the standard names defined by each provider's official documentation
PROVIDER_ENV_STANDARDS: Dict[AIModelProvider, ProviderEnvStandard] = {
    AIModelProvider.OPENAI: ProviderEnvStandard(
        primary_env_var="OPENAI_API_KEY",
        alternative_env_vars=[],
        description="OpenAI API key for GPT models",
        official_docs="https://platform.openai.com/docs/quickstart",
        supported_models=["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
    ),
    AIModelProvider.ANTHROPIC: ProviderEnvStandard(
        primary_env_var="ANTHROPIC_API_KEY",
        alternative_env_vars=[],
        description="Anthropic API key for Claude models",
        official_docs="https://docs.anthropic.com/en/api/getting-started",
        supported_models=["claude-3-5-sonnet", "claude-3-opus", "claude-3-haiku"],
    ),
    AIModelProvider.GOOGLE: ProviderEnvStandard(
        primary_env_var="GOOGLE_API_KEY",
        alternative_env_vars=["GOOGLE_GENERATIVE_AI_API_KEY"],
        description="Google API key for Gemini models",
        official_docs="https://ai.google.dev/gemini-api/docs/api-key",
        supported_models=["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
    ),
    AIModelProvider.GROK: ProviderEnvStandard(
        primary_env_var="XAI_API_KEY",
        alternative_env_vars=["GROK_API_KEY"],
        description="xAI API key for Grok models",
        official_docs="https://docs.x.ai/docs/api",
        supported_models=["grok-2", "grok-beta"],
    ),
}

# Container model for all provider standards
PROVIDER_ENV_STANDARDS_MODEL = ProviderEnvStandards(
    openai=PROVIDER_ENV_STANDARDS[AIModelProvider.OPENAI],
    anthropic=PROVIDER_ENV_STANDARDS[AIModelProvider.ANTHROPIC],
    google=PROVIDER_ENV_STANDARDS[AIModelProvider.GOOGLE],
    grok=PROVIDER_ENV_STANDARDS[AIModelProvider.GROK],
)


def get_provider_secret_from_settings(
    provider: AIModelProvider, settings: Optional["Settings"] = None
) -> Optional[str]:
    """Get the API key secret value from the settings model for a provider.

    This function retrieves the actual secret value from the settings model,
    which loads from the internal environment variable naming scheme
    (AI_PROVIDER__*__API_KEY).

    Args:
        provider: AIModelProvider enum value
        settings: Optional Settings instance. If not provided, imports from config module.

    Returns:
        The API key secret value if available, None otherwise

    Raises:
        ValueError: If provider is not recognized
    """
    if settings is None:
        from gearmeshing_ai.server.core.config import settings as config_settings

        settings = config_settings

    if provider not in PROVIDER_ENV_STANDARDS:
        raise ValueError(f"Unknown provider: {provider}")

    if provider == AIModelProvider.OPENAI:
        api_key_secret = settings.ai_provider.openai.api_key
        return api_key_secret.get_secret_value() if api_key_secret else None

    elif provider == AIModelProvider.ANTHROPIC:
        api_key_secret = settings.ai_provider.anthropic.api_key
        return api_key_secret.get_secret_value() if api_key_secret else None

    elif provider == AIModelProvider.GOOGLE:
        api_key_secret = settings.ai_provider.google.api_key
        return api_key_secret.get_secret_value() if api_key_secret else None

    elif provider == AIModelProvider.GROK:
        # Grok is not yet in the settings model, return None
        return None

    return None


def export_provider_env_vars_from_settings(provider: AIModelProvider, settings: Optional["BaseAISetting"] = None) -> bool:
    """Export provider API key from settings to official environment variables.

    This function retrieves the API key from the settings model and sets it
    as the official provider environment variable (e.g., OPENAI_API_KEY).

    This is useful for:
    - Initializing external libraries that expect official env var names
    - Ensuring consistency between internal settings and external tools
    - Setting up environment for third-party integrations

    Args:
        provider: AIModelProvider enum value
        settings: Optional BaseAISetting instance. If not provided, imports from config module.

    Returns:
        True if environment variable was set, False if no API key found in settings

    Raises:
        ValueError: If provider is not recognized
    """
    if provider not in PROVIDER_ENV_STANDARDS:
        raise ValueError(f"Unknown provider: {provider}")

    # Get the API key from settings
    api_key = get_provider_secret_from_settings(provider, settings)

    if not api_key:
        return False

    # Get the primary environment variable name
    env_var_name = PROVIDER_ENV_STANDARDS[provider].primary_env_var

    # Set the environment variable
    os.environ[env_var_name] = api_key

    return True


def export_all_provider_env_vars_from_settings(settings: Optional["BaseAISetting"] = None) -> Dict[str, bool]:
    """Export all provider API keys from settings to official environment variables.

    This function retrieves all available API keys from the settings model
    and sets them as official provider environment variables.

    Args:
        settings: Optional BaseAISetting instance. If not provided, imports from config module.

    Returns:
        Dictionary mapping provider names to success status (True if set, False if not found)

    Example:
        ```python
        from gearmeshing_ai.agent_core.abstraction.provider_env_standards import (
            export_all_provider_env_vars_from_settings,
        )

        results = export_all_provider_env_vars_from_settings()
        for provider, success in results.items():
            if success:
                print(f"{provider}: Environment variable set successfully")
            else:
                print(f"{provider}: No API key found in settings")
        ```
    """
    results = {}

    for provider in AIModelProvider:
        try:
            success = export_provider_env_vars_from_settings(provider, settings)
            results[provider.value] = success
        except ValueError:
            results[provider.value] = False

    return results
