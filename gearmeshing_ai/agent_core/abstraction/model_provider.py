"""Base abstractions for AI model providers.

This module defines framework-agnostic interfaces for creating and managing
LLM model instances, enabling seamless switching between AI frameworks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    """Framework-agnostic model configuration.

    Attributes:
        provider: AI provider name (openai, anthropic, google, etc.)
        model: Model name (gpt-4o, claude-3-opus, gemini-2.0-flash, etc.)
        temperature: Model temperature (0.0-2.0)
        max_tokens: Maximum output tokens
        top_p: Top-p sampling (0.0-1.0)
        timeout: Request timeout in seconds
        api_key: API key for the provider (optional, can be from env)
        model_settings: Additional framework-specific settings
    """

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    provider: str = Field(..., description="AI provider name")
    model: str = Field(..., description="Model name")
    temperature: float = Field(default=0.7, description="Model temperature (0.0-2.0)")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum output tokens")
    top_p: Optional[float] = Field(default=0.9, description="Top-p sampling (0.0-1.0)")
    timeout: Optional[float] = Field(None, description="Request timeout in seconds")
    api_key: Optional[str] = Field(None, description="API key (optional, can be from env)")
    model_settings: Dict[str, Any] = Field(default_factory=dict, description="Framework-specific settings")


class ModelResponse(BaseModel):
    """Framework-agnostic model response.

    Attributes:
        content: The generated text content
        finish_reason: Why the generation finished (length, stop, etc.)
        usage: Token usage information
        metadata: Additional framework-specific metadata
    """

    content: str = Field(..., description="Generated text content")
    finish_reason: Optional[str] = Field(None, description="Why generation finished")
    usage: Dict[str, Any] = Field(default_factory=dict, description="Token usage info")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Framework-specific metadata")


class ModelInstance(Protocol):
    """Protocol for framework-agnostic model instances.

    Any model instance from any framework should implement this protocol
    to ensure consistent usage across the system.
    """

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate text from the model."""
        ...

    async def generate_structured(
        self,
        prompt: str,
        output_schema: type,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """Generate structured output from the model."""
        ...


class ModelProvider(ABC):
    """Abstract base class for AI model providers.

    This defines the contract that all framework-specific model providers
    must implement, ensuring consistent behavior across different AI frameworks.
    """

    @abstractmethod
    def create_model(self, config: ModelConfig) -> ModelInstance:
        """Create a model instance from configuration.

        Args:
            config: Model configuration with provider, model, and parameters

        Returns:
            ModelInstance that can be used for generation

        Raises:
            ValueError: If provider or model is not supported
            RuntimeError: If required API keys are not configured
        """
        ...

    @abstractmethod
    def get_supported_providers(self) -> List[str]:
        """Get list of supported AI providers.

        Returns:
            List of provider names (e.g., ['openai', 'anthropic', 'google'])
        """
        ...

    @abstractmethod
    def get_supported_models(self, provider: str) -> List[str]:
        """Get list of supported models for a provider.

        Args:
            provider: Provider name

        Returns:
            List of model names for the provider

        Raises:
            ValueError: If provider is not supported
        """
        ...

    @abstractmethod
    def validate_config(self, config: ModelConfig) -> None:
        """Validate model configuration.

        Args:
            config: Model configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        ...

    def create_fallback_model(
        self,
        primary_config: ModelConfig,
        fallback_config: ModelConfig,
    ) -> ModelInstance:
        """Create a fallback model that tries primary first, then fallback.

        Default implementation creates a simple fallback wrapper.
        Subclasses can override for framework-specific optimizations.

        Args:
            primary_config: Primary model configuration
            fallback_config: Fallback model configuration

        Returns:
            ModelInstance with fallback behavior
        """
        return FallbackModelInstance(
            primary=self.create_model(primary_config),
            fallback=self.create_model(fallback_config),
        )

    def get_provider_from_model_name(self, model_name: str) -> str:
        """Determine provider from model name using patterns.

        Args:
            model_name: Model name (e.g., 'gpt-4o', 'claude-3-opus')

        Returns:
            Provider name

        Raises:
            ValueError: If provider cannot be determined
        """
        from ..api_key_validator import APIKeyValidator

        provider = APIKeyValidator.get_provider_for_model(model_name)
        if provider is None:
            raise ValueError(
                f"Could not determine provider for model '{model_name}'. "
                f"Supported prefixes: gpt-*, claude-*, gemini-*, grok-*"
            )
        return provider.value


class FallbackModelInstance:
    """Fallback model implementation that tries primary then fallback.

    This is a framework-agnostic fallback implementation that can be used
    by any model provider. Framework-specific providers can override
    create_fallback_model for optimized implementations.
    """

    def __init__(self, primary: ModelInstance, fallback: ModelInstance):
        """Initialize fallback model.

        Args:
            primary: Primary model instance
            fallback: Fallback model instance
        """
        self.primary = primary
        self.fallback = fallback

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate using primary, fallback to secondary on failure."""
        try:
            return await self.primary.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
        except Exception as e:
            # Log the primary failure and try fallback
            print(f"Primary model failed: {e}, trying fallback")
            return await self.fallback.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)

    async def generate_structured(
        self,
        prompt: str,
        output_schema: type,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """Generate structured output using primary, fallback to secondary."""
        try:
            return await self.primary.generate_structured(
                prompt, output_schema, max_tokens=max_tokens, temperature=temperature, **kwargs
            )
        except Exception as e:
            # Log the primary failure and try fallback
            print(f"Primary model failed: {e}, trying fallback")
            return await self.fallback.generate_structured(
                prompt, output_schema, max_tokens=max_tokens, temperature=temperature, **kwargs
            )


class ModelProviderFactory(ABC):
    """Abstract factory for creating model providers.

    This enables dependency injection and configuration-driven
    selection of model provider implementations.
    """

    @abstractmethod
    def create_provider(self, framework: str, **kwargs: Any) -> ModelProvider:
        """Create a model provider for the specified framework.

        Args:
            framework: Framework name (e.g., 'pydantic_ai', 'langchain')
            **kwargs: Framework-specific configuration

        Returns:
            ModelProvider instance

        Raises:
            ValueError: If framework is not supported
        """
        ...

    @abstractmethod
    def get_supported_frameworks(self) -> List[str]:
        """Get list of supported frameworks.

        Returns:
            List of framework names
        """
        ...
