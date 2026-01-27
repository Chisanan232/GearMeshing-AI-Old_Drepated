"""AI Framework Adapters.

This module provides adapter implementations for various AI frameworks,
allowing them to work with the unified AIAgentBase abstraction.

Available Adapters:
- PydanticAIAgent: Adapter for Pydantic AI framework
- PydanticAIModelProvider: Model provider adapter for Pydantic AI framework
"""

from .pydantic_ai import PydanticAIAgent
from .pydantic_ai_model_provider import (
    PydanticAIModelProvider,
    PydanticAIModelProviderFactory,
)

__all__ = [
    "PydanticAIAgent",
    "PydanticAIModelProvider",
    "PydanticAIModelProviderFactory",
]
