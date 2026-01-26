"""Base interface for AI model configuration providers.

This module defines the core interface that all model provider implementations
must adhere to, enabling framework-agnostic model configuration management
throughout the project.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from gearmeshing_ai.agent_core.schemas.config import ModelConfig


class ModelProvider(ABC):
    """Base interface for AI model configuration providers.

    This interface mirrors the PromptProvider interface to maintain consistency
    across the info_provider package. All model providers must implement
    these methods to ensure compatibility with the ModelProviderLoader.
    """

    @abstractmethod
    def get(self, name: str, tenant: Optional[str] = None) -> ModelConfig:
        """
        Retrieve a model configuration by name.

        Args:
            name: The model configuration key (e.g., 'gpt4_default', 'claude_senior').
            tenant: Optional tenant identifier for multi-tenant configurations.

        Returns:
            ModelConfig: The model configuration with provider and parameters.

        Raises:
            KeyError: If the model configuration name is not found.
        """

    @abstractmethod
    def version(self) -> str:
        """
        Return the version identifier of this provider.

        Returns:
            str: Version string for tracking and debugging purposes.
        """

    @abstractmethod
    def refresh(self) -> None:
        """
        Refresh the provider state.

        For static providers (like hardcoded), this is a no-op.
        For dynamic providers (database, external), this reloads configurations.
        """
