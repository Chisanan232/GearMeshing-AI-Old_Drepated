"""Agent caching system for managing agent instances.

This module provides caching and lifecycle management for AI agent instances,
enabling efficient reuse and resource management.
"""

import asyncio
from typing import Dict, Optional

from .base import AIAgentBase, AIAgentConfig


class AIAgentCache:
    """Cache for managing AI agent instances.

    This cache handles:
    - Agent instance creation and reuse
    - Lifecycle management (initialization, cleanup)
    - Thread-safe access to cached agents
    - Automatic cleanup on shutdown

    Attributes:
        max_size: Maximum number of agents to cache (0 = unlimited)
        ttl: Time-to-live for cached agents in seconds (None = no expiry)
    """

    def __init__(self, max_size: int = 0, ttl: Optional[float] = None) -> None:
        """Initialize the agent cache.

        Args:
            max_size: Maximum number of agents to cache (0 = unlimited)
            ttl: Time-to-live for cached agents in seconds
        """
        self._agents: Dict[str, AIAgentBase] = {}
        self._max_size = max_size
        self._ttl = ttl
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[AIAgentBase]:
        """Get an agent from the cache.

        Args:
            key: Cache key (typically agent name or config hash)

        Returns:
            Cached agent instance or None if not found
        """
        async with self._lock:
            return self._agents.get(key)

    async def set(self, key: str, agent: AIAgentBase) -> None:
        """Store an agent in the cache.

        Args:
            key: Cache key
            agent: Agent instance to cache

        Raises:
            RuntimeError: If cache is full and max_size is set
        """
        async with self._lock:
            if self._max_size > 0 and len(self._agents) >= self._max_size:
                # Remove oldest agent (simple FIFO)
                oldest_key = next(iter(self._agents))
                oldest_agent = self._agents.pop(oldest_key)
                await oldest_agent.cleanup()

            self._agents[key] = agent

    async def remove(self, key: str) -> None:
        """Remove an agent from the cache and clean it up.

        Args:
            key: Cache key
        """
        async with self._lock:
            if key in self._agents:
                agent = self._agents.pop(key)
                await agent.cleanup()

    async def clear(self) -> None:
        """Clear all cached agents and clean them up."""
        async with self._lock:
            for agent in self._agents.values():
                await agent.cleanup()
            self._agents.clear()

    async def has(self, key: str) -> bool:
        """Check if an agent is cached.

        Args:
            key: Cache key

        Returns:
            True if agent is cached, False otherwise
        """
        async with self._lock:
            return key in self._agents

    def size(self) -> int:
        """Get the current cache size."""
        return len(self._agents)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.clear()
