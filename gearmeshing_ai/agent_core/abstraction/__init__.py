"""AI Agent Abstraction Layer.

This module provides a framework-agnostic abstraction for AI agents,
allowing the project to support multiple AI frameworks (Pydantic AI, LangChain, etc.)
through a unified interface.

Key Components:
- AIAgentBase: Core abstraction for AI agent implementations
- AIAgentFactory: Factory for creating and managing agent instances
- AIAgentProvider: Provider for selecting and configuring agents
- AIAgentCache: Caching layer for agent instances
- AIAgentConfig: Configuration for agent initialization
"""

from .base import AIAgentBase, AIAgentConfig, AIAgentResponse

__all__ = [
    "AIAgentBase",
    "AIAgentConfig",
    "AIAgentResponse",
]
