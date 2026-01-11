"""AI Framework Adapters.

This module provides adapter implementations for various AI frameworks,
allowing them to work with the unified AIAgentBase abstraction.

Available Adapters:
- PydanticAIAgent: Adapter for Pydantic AI framework
"""

from .pydantic_ai import PydanticAIAgent

__all__ = ["PydanticAIAgent"]
