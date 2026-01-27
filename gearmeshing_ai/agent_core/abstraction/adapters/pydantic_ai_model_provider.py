"""Pydantic AI adapter for the model provider abstraction.

This module implements the ModelProvider interface using Pydantic AI framework,
providing concrete implementations for creating and managing Pydantic AI models.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from pydantic_ai import ModelSettings
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIResponsesModel

from ..model_provider import (
    FallbackModelInstance,
    ModelConfig,
    ModelInstance,
    ModelProvider,
    ModelProviderFactory,
    ModelResponse,
)

logger = logging.getLogger(__name__)


class PydanticAIModelInstance(ModelInstance):
    """Pydantic AI implementation of ModelInstance protocol."""

    def __init__(self, model: Model, agent: Optional[Any] = None):
        """Initialize Pydantic AI model instance.

        Args:
            model: Pydantic AI model instance
            agent: Optional Pydantic AI agent for structured generation
        """
        self.model = model
        self.agent = agent

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate text using Pydantic AI model."""
        try:
            # For now, we'll use a simple approach - in a full implementation,
            # we'd need to handle the agent creation and execution properly
            # This is a simplified version for demonstration
            if self.agent:
                result = await self.agent.run(prompt)
                content = result.data
            else:
                # Direct model usage would need to be implemented
                # based on Pydantic AI's specific API
                content = "Generated content (simplified implementation)"

            return ModelResponse(
                content=content,
                finish_reason="stop",
                usage={"prompt_tokens": 0, "completion_tokens": 0},
                metadata={"framework": "pydantic_ai"},
            )
        except Exception as e:
            logger.error(f"Pydantic AI generation failed: {e}")
            raise

    async def generate_structured(
        self,
        prompt: str,
        output_schema: type,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """Generate structured output using Pydantic AI."""
        try:
            # This would need proper Pydantic AI agent implementation
            # For now, this is a placeholder
            if self.agent:
                result = await self.agent.run(prompt)
                return result.data
            else:
                # Create a simple structured response
                return {"result": "Structured output (simplified)"}
        except Exception as e:
            logger.error(f"Pydantic AI structured generation failed: {e}")
            raise


class PydanticAIModelProvider(ModelProvider):
    """Pydantic AI implementation of ModelProvider."""

    def __init__(self, db_session: Optional[Any] = None):
        """Initialize Pydantic AI model provider.

        Args:
            db_session: Optional database session for configuration
        """
        self.db_session = db_session
        self._db_provider = None

    def create_model(self, config: ModelConfig) -> ModelInstance:
        """Create a Pydantic AI model instance from configuration."""
        self.validate_config(config)

        provider_lower = config.provider.lower()

        if provider_lower == "openai":
            pydantic_model = self._create_openai_model(config)
        elif provider_lower == "anthropic":
            pydantic_model = self._create_anthropic_model(config)
        elif provider_lower == "google":
            pydantic_model = self._create_google_model(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

        return PydanticAIModelInstance(pydantic_model)

    def _create_openai_model(self, config: ModelConfig) -> Model:
        """Create OpenAI model using Pydantic AI."""
        from gearmeshing_ai.server.core.config import settings

        api_key_secret = settings.ai_provider.openai.api_key
        api_key: Optional[str] = api_key_secret.get_secret_value() if api_key_secret else None
        if not api_key:
            raise RuntimeError("AI_PROVIDER__OPENAI__API_KEY environment variable is not set")

        model_settings = ModelSettings(
            temperature=config.temperature,
            max_tokens=config.max_tokens or 4096,
            top_p=config.top_p,
        )

        logger.debug(f"Creating OpenAI model: {config.model} with Pydantic AI")
        return OpenAIResponsesModel(config.model, settings=model_settings)

    def _create_anthropic_model(self, config: ModelConfig) -> Model:
        """Create Anthropic model using Pydantic AI."""
        from gearmeshing_ai.server.core.config import settings

        api_key_secret = settings.ai_provider.anthropic.api_key
        api_key: Optional[str] = api_key_secret.get_secret_value() if api_key_secret else None
        if not api_key:
            raise RuntimeError("AI_PROVIDER__ANTHROPIC__API_KEY environment variable is not set")

        model_settings = ModelSettings(
            temperature=config.temperature,
            max_tokens=config.max_tokens or 4096,
            top_p=config.top_p,
        )

        logger.debug(f"Creating Anthropic model: {config.model} with Pydantic AI")
        return AnthropicModel(config.model, settings=model_settings)

    def _create_google_model(self, config: ModelConfig) -> Model:
        """Create Google model using Pydantic AI."""
        from gearmeshing_ai.server.core.config import settings

        api_key_secret = settings.ai_provider.google.api_key
        api_key: Optional[str] = api_key_secret.get_secret_value() if api_key_secret else None
        if not api_key:
            raise RuntimeError("AI_PROVIDER__GOOGLE__API_KEY environment variable is not set")

        model_settings = ModelSettings(
            temperature=config.temperature,
            max_tokens=config.max_tokens or 4096,
            top_p=config.top_p,
        )

        logger.debug(f"Creating Google model: {config.model} with Pydantic AI")
        return GoogleModel(config.model, settings=model_settings)

    def get_supported_providers(self) -> List[str]:
        """Get list of supported providers."""
        return ["openai", "anthropic", "google"]

    def get_supported_models(self, provider: str) -> List[str]:
        """Get list of supported models for a provider."""
        provider_models = {
            "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-opus-latest"],
            "google": ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
        }

        if provider not in provider_models:
            raise ValueError(f"Unsupported provider: {provider}")

        return provider_models[provider]

    def validate_config(self, config: ModelConfig) -> None:
        """Validate model configuration."""
        if not config.provider:
            raise ValueError("Provider is required")

        if not config.model:
            raise ValueError("Model is required")

        if config.provider not in self.get_supported_providers():
            raise ValueError(f"Unsupported provider: {config.provider}")

        if config.model not in self.get_supported_models(config.provider):
            raise ValueError(f"Unsupported model for provider {config.provider}: {config.model}")

        if config.temperature < 0 or config.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")

        if config.max_tokens and config.max_tokens < 1:
            raise ValueError("Max tokens must be at least 1")

        if config.top_p and (config.top_p < 0 or config.top_p > 1):
            raise ValueError("Top-p must be between 0 and 1")

    def create_fallback_model(
        self,
        primary_config: ModelConfig,
        fallback_config: ModelConfig,
    ) -> ModelInstance:
        """Create Pydantic AI fallback model."""
        primary_model = self.create_model(primary_config)
        fallback_model = self.create_model(fallback_config)

        # Use Pydantic AI's native FallbackModel if both are Pydantic AI models
        if hasattr(primary_model, "model") and hasattr(fallback_model, "model"):
            fallback_pydantic_model = FallbackModel(primary_model.model, fallback_model.model)
            return PydanticAIModelInstance(fallback_pydantic_model)

        # Fall back to generic implementation
        return FallbackModelInstance(primary_model, fallback_model)


class PydanticAIModelProviderFactory(ModelProviderFactory):
    """Factory for creating Pydantic AI model providers."""

    def create_provider(self, framework: str, **kwargs: Any) -> ModelProvider:
        """Create a Pydantic AI model provider."""
        if framework != "pydantic_ai":
            raise ValueError(f"Unsupported framework: {framework}")

        db_session = kwargs.get("db_session")
        return PydanticAIModelProvider(db_session=db_session)

    def get_supported_frameworks(self) -> List[str]:
        """Get supported frameworks."""
        return ["pydantic_ai"]
