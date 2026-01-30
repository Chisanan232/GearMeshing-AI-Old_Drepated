"""
Refactored Model Provider using Abstraction Layer.

This module provides a unified interface for creating LLM model instances
using the abstraction layer, supporting multiple AI frameworks through
a consistent API.

This replaces the original Pydantic AI-specific implementation with
a framework-agnostic approach.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sqlmodel import Session

    from gearmeshing_ai.info_provider.model.base import (
        ModelProvider as InfoModelProvider,
    )

from gearmeshing_ai.agent_core.abstraction import (
    ModelInstance,
    ModelProvider,
    ModelProviderFactory,
)
from gearmeshing_ai.agent_core.abstraction.adapters import (
    PydanticAIModelProviderFactory,
)
from gearmeshing_ai.core.models.config import ModelConfig

logger = logging.getLogger(__name__)


class UnifiedModelProvider:
    """
    Unified model provider that supports multiple AI frameworks.

    This class acts as a facade over the abstraction layer, providing
    the same interface as the original ModelProvider but with support
    for multiple frameworks.
    """

    def __init__(self, db_session: Session, framework: str = "pydantic_ai") -> None:
        """
        Initialize the unified model provider.

        Args:
            db_session: SQLModel database session for database-driven configuration
            framework: AI framework to use (default: pydantic_ai)

        Raises:
            ValueError: If db_session is not provided or framework is unsupported
        """
        if db_session is None:
            raise ValueError("db_session is required for UnifiedModelProvider")

        self.db_session: Session = db_session
        self.framework = framework
        self._db_provider: Optional[InfoModelProvider] = None
        self._provider: Optional[ModelProvider] = None

        # Initialize the framework-specific provider
        self._initialize_provider()

    def _initialize_provider(self) -> None:
        """Initialize the framework-specific model provider."""
        try:
            factory = self._get_provider_factory()
            self._provider = factory.create_provider(self.framework, db_session=self.db_session)
            logger.debug(f"Initialized model provider for framework: {self.framework}")
        except Exception as e:
            logger.error(f"Failed to initialize provider for framework {self.framework}: {e}")
            raise

    def _get_provider_factory(self) -> ModelProviderFactory:
        """Get the provider factory for the specified framework."""
        factories = {
            "pydantic_ai": PydanticAIModelProviderFactory(),
        }

        if self.framework not in factories:
            raise ValueError(f"Unsupported framework: {self.framework}")

        return factories[self.framework]

    def _get_db_provider(self) -> InfoModelProvider:
        """Get or create database configuration provider."""
        if self._db_provider is None:
            from gearmeshing_ai.info_provider.model.provider import (
                DatabaseModelProvider,
            )

            def session_factory():
                return self.db_session

            self._db_provider = DatabaseModelProvider(session_factory)
        return self._db_provider

    def create_model(
        self,
        provider: str,
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> ModelInstance:
        """
        Create an LLM model instance using the abstraction layer.

        Args:
            provider: Provider name ('openai', 'anthropic', 'google').
            model: Model name (e.g., 'gpt-4o', 'claude-3-5-sonnet').
            temperature: Model temperature (0.0-2.0). If None, uses config default.
            max_tokens: Maximum tokens. If None, uses config default.
            top_p: Top-p sampling (0.0-1.0). If None, uses config default.

        Returns:
            ModelInstance: The created model instance

        Raises:
            ValueError: If provider or model is not supported.
            RuntimeError: If required API keys are not configured.
        """
        if not self._provider:
            raise RuntimeError("Model provider not initialized")

        config = ModelConfig(
            provider=provider,
            model=model,
            temperature=temperature or 0.7,
            max_tokens=max_tokens,
            top_p=top_p or 0.9,
        )

        return self._provider.create_model(config)

    def get_provider_from_model_name(self, model_name: str) -> str:
        """Determine the provider from a model name using patterns.

        Args:
            model_name: The model name (e.g., 'gpt-4o', 'claude-3-opus', 'gemini-2.0-flash').

        Returns:
            str: The provider name ('openai', 'anthropic', 'google', 'grok').

        Raises:
            ValueError: If the model name doesn't match any known provider pattern.
        """
        if not self._provider:
            raise RuntimeError("Model provider not initialized")

        return self._provider.get_provider_from_model_name(model_name)

    def create_fallback_model(
        self,
        primary_provider: str,
        primary_model: str,
        fallback_provider: str,
        fallback_model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> ModelInstance:
        """
        Create a fallback model that tries primary model first, then falls back.

        Args:
            primary_provider: Primary provider name ('openai', 'anthropic', 'google').
            primary_model: Primary model name.
            fallback_provider: Fallback provider name.
            fallback_model: Fallback model name.
            temperature: Model temperature. If None, uses config default.
            max_tokens: Maximum tokens. If None, uses config default.
            top_p: Top-p sampling. If None, uses config default.

        Returns:
            ModelInstance: Instance with primary and fallback models.
        """
        if not self._provider:
            raise RuntimeError("Model provider not initialized")

        primary_config = ModelConfig(
            provider=primary_provider,
            model=primary_model,
            temperature=temperature or 0.7,
            max_tokens=max_tokens,
            top_p=top_p or 0.9,
        )

        fallback_config = ModelConfig(
            provider=fallback_provider,
            model=fallback_model,
            temperature=temperature or 0.7,
            max_tokens=max_tokens,
            top_p=top_p or 0.9,
        )

        return self._provider.create_fallback_model(primary_config, fallback_config)

    def create_model_for_role(
        self,
        role: str,
        tenant_id: Optional[str] = None,
    ) -> ModelInstance:
        """
        Create a model instance for a specific role.

        Supports role-specific model configuration with tenant overrides.
        Configuration is loaded from the database.

        Args:
            role: Role name (e.g., 'dev', 'planner').
            tenant_id: Optional tenant identifier for tenant-specific overrides.

        Returns:
            ModelInstance: The created model instance.

        Raises:
            ValueError: If role is not found in database configuration.
        """
        from gearmeshing_ai.core.models.config import ModelConfig as DbModelConfig

        db_provider = self._get_db_provider()
        model_config: DbModelConfig = db_provider.get(role, tenant_id)
        logger.debug(f"Creating model for role '{role}': {model_config.model}")

        config = ModelConfig(
            provider=model_config.provider,
            model=model_config.model,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            top_p=model_config.top_p,
        )

        assert self._provider
        return self._provider.create_model(config)

    def get_supported_providers(self) -> list[str]:
        """Get list of supported AI providers."""
        if not self._provider:
            raise RuntimeError("Model provider not initialized")

        return self._provider.get_supported_providers()

    def get_supported_models(self, provider: str) -> list[str]:
        """Get list of supported models for a provider."""
        if not self._provider:
            raise RuntimeError("Model provider not initialized")

        return self._provider.get_supported_models(provider)


# Backward compatibility functions
def get_model_provider(db_session: Session, framework: str = "pydantic_ai") -> UnifiedModelProvider:
    """
    Get a unified model provider instance.

    Args:
        db_session: SQLModel database session for configuration.
        framework: AI framework to use (default: pydantic_ai)

    Returns:
        UnifiedModelProvider: A model provider instance.
    """
    return UnifiedModelProvider(db_session, framework)


def create_model_for_role(
    db_session: Session,
    role: str,
    tenant_id: Optional[str] = None,
    framework: str = "pydantic_ai",
) -> ModelInstance:
    """
    Create a model instance for a specific role from database configuration.

    Args:
        db_session: SQLModel database session.
        role: Role name (e.g., 'dev', 'qa', 'planner').
        tenant_id: Optional tenant identifier for tenant-specific overrides.
        framework: AI framework to use (default: pydantic_ai)

    Returns:
        ModelInstance: The created model instance.

    Raises:
        ValueError: If role is not found in database configuration.
    """
    provider = get_model_provider(db_session, framework)
    return provider.create_model_for_role(role, tenant_id)


async def async_create_model_for_role(
    role: str,
    tenant_id: Optional[str] = None,
    framework: str = "pydantic_ai",
) -> ModelInstance:
    """
    Create a model instance for a specific role from database configuration (async).

    Args:
        role: Role name (e.g., 'dev', 'qa', 'planner').
        tenant_id: Optional tenant identifier for tenant-specific overrides.
        framework: AI framework to use (default: pydantic_ai)

    Returns:
        ModelInstance: The created model instance.

    Raises:
        ValueError: If role is not found in database configuration.
    """
    from sqlalchemy import create_engine as sync_create_engine
    from sqlmodel import Session

    from gearmeshing_ai.server.core.config import settings

    try:
        # Create a sync engine and session for model provider
        db_url = settings.database.url
        if db_url.startswith("postgresql+asyncpg://"):
            sync_db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
        elif db_url.startswith("sqlite+aiosqlite://"):
            sync_db_url = db_url.replace("sqlite+aiosqlite://", "sqlite://")
        else:
            sync_db_url = db_url

        sync_engine = sync_create_engine(sync_db_url)
        session = Session(sync_engine)

        try:
            provider = get_model_provider(session, framework)
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
    framework: str = "pydantic_ai",
) -> ModelInstance:
    """Get a model for a specific role (async convenience function).

    Alias for async_create_model_for_role for consistency with sync version.

    Args:
        role: Role name (e.g., 'dev', 'qa', 'planner').
        tenant_id: Optional tenant identifier for tenant-specific overrides.
        framework: AI framework to use (default: pydantic_ai)

    Returns:
        ModelInstance: The created model instance.

    Raises:
        ValueError: If role is not found in database configuration.
    """
    return await async_create_model_for_role(role, tenant_id, framework)
