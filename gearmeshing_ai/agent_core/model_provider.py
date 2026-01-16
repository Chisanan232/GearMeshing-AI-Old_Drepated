"""
Model provider for creating LLM instances with database-driven configuration.

This module provides utilities to create language model instances from Pydantic AI framework,
supporting multiple providers (OpenAI, Anthropic, Google) with configurable parameters
like temperature, max_tokens, and top_p. All configuration is stored in the database.

Integration with Abstraction Layer
-----------------------------------
This module works seamlessly with the AI agent abstraction layer defined in
gearmeshing_ai.agent_core.abstraction. The abstraction layer provides:
- AIAgentFactory: Factory for creating and managing agent instances
- AIAgentProvider: Provider for selecting and configuring agents
- AIAgentConfig: Configuration for agent initialization

The model_provider creates the underlying Pydantic AI models that are wrapped
by the abstraction layer's adapters (e.g., PydanticAIAdapter).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

from pydantic_ai import ModelSettings
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIResponsesModel

if TYPE_CHECKING:
    from sqlmodel import Session

    from gearmeshing_ai.agent_core.db_config_provider import DatabaseConfigProvider

logger = logging.getLogger(__name__)


class ModelProvider:
    """
    Provider for creating LLM model instances from database configuration.

    Supports multiple AI providers (OpenAI, Anthropic, Google) with
    configurable model parameters stored in the database.
    """

    def __init__(self, db_session: Session) -> None:
        """
        Initialize the model provider.

        Args:
            db_session: SQLModel database session for database-driven configuration.

        Raises:
            ValueError: If db_session is not provided.
        """
        if db_session is None:
            raise ValueError("db_session is required for ModelProvider")
        self.db_session: Session = db_session
        self._db_provider: Optional[DatabaseConfigProvider] = None

    def _get_db_provider(self) -> DatabaseConfigProvider:
        """Get or create database configuration provider.

        Returns:
            DatabaseConfigProvider instance for accessing database configuration.
        """
        if self._db_provider is None:
            from .db_config_provider import DatabaseConfigProvider

            self._db_provider = DatabaseConfigProvider(self.db_session)
        return self._db_provider

    def create_model(
        self,
        provider: str,
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Model:
        """
        Create an LLM model instance with LangSmith tracing support.

        When LangSmith tracing is enabled, all LLM calls made through this model
        will be automatically traced and visible in the LangSmith dashboard.

        Args:
            provider: Provider name ('openai', 'anthropic', 'google').
            model: Model name (e.g., 'gpt-4o', 'claude-3-5-sonnet').
            temperature: Model temperature (0.0-2.0). If None, uses config default.
            max_tokens: Maximum tokens. If None, uses config default.
            top_p: Top-p sampling (0.0-1.0). If None, uses config default.

        Returns:
            Model: The created model instance with LangSmith tracing enabled if configured.

        Raises:
            ValueError: If provider or model is not supported.
            RuntimeError: If required API keys are not configured.
        """
        provider_lower = provider.lower()

        if provider_lower == "openai":
            return self._create_openai_model(model, temperature, max_tokens, top_p)
        elif provider_lower == "anthropic":
            return self._create_anthropic_model(model, temperature, max_tokens, top_p)
        elif provider_lower == "google":
            return self._create_google_model(model, temperature, max_tokens, top_p)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _create_openai_model(
        self,
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Model:
        """Create an OpenAI model instance using Pydantic AI.

        Args:
            model: Model name (e.g., 'gpt-4o').
            temperature: Model temperature (0.0-2.0).
            max_tokens: Maximum output tokens.
            top_p: Top-p sampling (0.0-1.0).

        Returns:
            OpenAIResponsesModel instance.

        Raises:
            RuntimeError: If OPENAI_API_KEY is not set.
        """
        api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")

        # Use defaults if not provided
        temperature_val: float = temperature or 0.7
        max_tokens_val: int = max_tokens or 4096
        top_p_val: float = top_p or 0.9

        logger.debug(
            f"Creating OpenAI model: {model} (temp={temperature_val}, max_tokens={max_tokens_val}, top_p={top_p_val})"
        )

        # Create model settings for Pydantic AI
        settings: ModelSettings = ModelSettings(
            temperature=temperature_val,
            max_tokens=max_tokens_val,
            top_p=top_p_val,
        )

        return OpenAIResponsesModel(model, settings=settings)

    def _create_anthropic_model(
        self,
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Model:
        """Create an Anthropic (Claude) model instance using Pydantic AI.

        Args:
            model: Model name (e.g., 'claude-3-5-sonnet').
            temperature: Model temperature (0.0-2.0).
            max_tokens: Maximum output tokens.
            top_p: Top-p sampling (0.0-1.0).

        Returns:
            AnthropicModel instance.

        Raises:
            RuntimeError: If ANTHROPIC_API_KEY is not set.
        """
        api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")

        # Use defaults if not provided
        temperature_val: float = temperature or 0.7
        max_tokens_val: int = max_tokens or 4096
        top_p_val: float = top_p or 0.9

        logger.debug(
            f"Creating Anthropic model: {model} (temp={temperature_val}, max_tokens={max_tokens_val}, top_p={top_p_val})"
        )

        # Create model settings for Pydantic AI
        settings: ModelSettings = ModelSettings(
            temperature=temperature_val,
            max_tokens=max_tokens_val,
            top_p=top_p_val,
        )

        return AnthropicModel(model, settings=settings)

    def _create_google_model(
        self,
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Model:
        """Create a Google (Gemini) model instance using Pydantic AI.

        Args:
            model: Model name (e.g., 'gemini-2.0-flash').
            temperature: Model temperature (0.0-2.0).
            max_tokens: Maximum output tokens.
            top_p: Top-p sampling (0.0-1.0).

        Returns:
            GoogleModel instance.

        Raises:
            RuntimeError: If GOOGLE_API_KEY is not set.
        """
        api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable is not set")

        # Use defaults if not provided
        temperature_val: float = temperature or 0.7
        max_tokens_val: int = max_tokens or 4096
        top_p_val: float = top_p or 0.9

        logger.debug(
            f"Creating Google model: {model} (temp={temperature_val}, max_tokens={max_tokens_val}, top_p={top_p_val})"
        )

        # Create model settings for Pydantic AI
        settings: ModelSettings = ModelSettings(
            temperature=temperature_val,
            max_tokens=max_tokens_val,
            top_p=top_p_val,
        )

        return GoogleModel(model, settings=settings)

    def get_provider_from_model_name(self, model_name: str) -> str:
        """Determine the provider from a model name using regex patterns.

        This method integrates with the abstraction layer's model-to-provider
        identification system to determine which provider a model belongs to.

        Args:
            model_name: The model name (e.g., 'gpt-4o', 'claude-3-opus', 'gemini-2.0-flash').

        Returns:
            str: The provider name ('openai', 'anthropic', 'google', 'grok').

        Raises:
            ValueError: If the model name doesn't match any known provider pattern.
        """
        from gearmeshing_ai.agent_core.abstraction.api_key_validator import (
            APIKeyValidator,
        )

        provider = APIKeyValidator.get_provider_for_model(model_name)
        if provider is None:
            raise ValueError(
                f"Could not determine provider for model '{model_name}'. "
                f"Supported prefixes: gpt-*, claude-*, gemini-*, grok-*"
            )
        return provider.value

    def create_fallback_model(
        self,
        primary_provider: str,
        primary_model: str,
        fallback_provider: str,
        fallback_model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> FallbackModel:
        """
        Create a fallback model that tries primary model first, then falls back.

        This implements Pydantic AI's FallbackModel pattern for provider/model fallback chains.

        Args:
            primary_provider: Primary provider name ('openai', 'anthropic', 'google').
            primary_model: Primary model name.
            fallback_provider: Fallback provider name.
            fallback_model: Fallback model name.
            temperature: Model temperature. If None, uses config default.
            max_tokens: Maximum tokens. If None, uses config default.
            top_p: Top-p sampling. If None, uses config default.

        Returns:
            FallbackModel instance with primary and fallback models.
        """
        primary: Model = self.create_model(primary_provider, primary_model, temperature, max_tokens, top_p)
        fallback: Model = self.create_model(fallback_provider, fallback_model, temperature, max_tokens, top_p)

        logger.debug(
            f"Creating fallback model: {primary_provider}/{primary_model} -> {fallback_provider}/{fallback_model}"
        )

        return FallbackModel(primary, fallback)

    def create_model_for_role(
        self,
        role: str,
        tenant_id: Optional[str] = None,
    ) -> Model:
        """
        Create a model instance for a specific role.

        Supports role-specific model configuration with tenant overrides.
        Configuration is loaded from the database.

        Args:
            role: Role name (e.g., 'dev', 'planner').
            tenant_id: Optional tenant identifier for tenant-specific overrides.

        Returns:
            Model: The created model instance.

        Raises:
            ValueError: If role is not found in database configuration.
        """
        from gearmeshing_ai.agent_core.schemas.config import ModelConfig

        db_provider: DatabaseConfigProvider = self._get_db_provider()
        model_config: ModelConfig = db_provider.get_model_config(role, tenant_id)
        logger.debug(f"Creating model for role '{role}': {model_config.model}")
        return self.create_model(
            provider=model_config.provider,
            model=model_config.model,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            top_p=model_config.top_p,
        )


def get_model_provider(db_session: Session) -> ModelProvider:
    """
    Get a model provider instance.

    Args:
        db_session: SQLModel database session for configuration.

    Returns:
        ModelProvider: A model provider instance.
    """
    return ModelProvider(db_session)


def create_model_for_role(
    db_session: Session,
    role: str,
    tenant_id: Optional[str] = None,
) -> Model:
    """
    Create a model instance for a specific role from database configuration.

    Args:
        db_session: SQLModel database session.
        role: Role name (e.g., 'dev', 'qa', 'planner').
        tenant_id: Optional tenant identifier for tenant-specific overrides.

    Returns:
        Model: The created Pydantic AI model instance.

    Raises:
        ValueError: If role is not found in database configuration.

    Example:
        >>> model = create_model_for_role(session, 'dev', tenant_id='acme-corp')
        >>> agent = Agent(model, system_prompt="You are a developer assistant")
    """
    provider: ModelProvider = get_model_provider(db_session)
    return provider.create_model_for_role(role, tenant_id)


async def async_create_model_for_role(
    role: str,
    tenant_id: Optional[str] = None,
) -> Model:
    """
    Create a model instance for a specific role from database configuration (async).

    This is the async-friendly version for use in async contexts like the engine.
    It automatically handles session creation and cleanup.

    Args:
        role: Role name (e.g., 'dev', 'qa', 'planner').
        tenant_id: Optional tenant identifier for tenant-specific overrides.

    Returns:
        Model: The created Pydantic AI model instance.

    Raises:
        ValueError: If role is not found in database configuration.
        RuntimeError: If database session cannot be created.

    Example:
        >>> model = await async_create_model_for_role('dev', tenant_id='acme-corp')
        >>> agent = Agent(model, system_prompt="You are a developer assistant")
    """
    from sqlalchemy import create_engine as sync_create_engine
    from sqlmodel import Session

    from gearmeshing_ai.server.core.config import settings

    try:
        # Create a sync engine and session for model provider
        # The model provider requires a sync session, so we create one here
        sync_engine = sync_create_engine(settings.database.url)

        session = Session(sync_engine)
        try:
            provider: ModelProvider = get_model_provider(session)
            model = provider.create_model_for_role(role, tenant_id)
            logger.debug(f"Created model for role '{role}' in async context")
            return model
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Failed to create model for role '{role}' in async context: {e}")
        raise


async def async_get_model_provider(
    role: str,
    tenant_id: Optional[str] = None,
) -> Model:
    """Get a model for a specific role (async convenience function).

    Alias for async_create_model_for_role for consistency with sync version.

    Integration with Abstraction Layer
    -----------------------------------
    This function works with the abstraction layer to provide models that can be
    wrapped by AIAgentFactory and managed by AIAgentProvider. The returned model
    can be used to create AIAgentConfig instances for agent creation.

    Args:
        role: Role name (e.g., 'dev', 'qa', 'planner').
        tenant_id: Optional tenant identifier for tenant-specific overrides.

    Returns:
        Model: The created Pydantic AI model instance.

    Raises:
        ValueError: If role is not found in database configuration.
        RuntimeError: If database session cannot be created.
    """
    return await async_create_model_for_role(role, tenant_id)
