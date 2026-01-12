"""Unit tests for AI agent cache."""

import pytest

from gearmeshing_ai.agent_core.abstraction.base import (
    AIAgentBase,
    AIAgentConfig,
    AIAgentResponse,
)
from gearmeshing_ai.agent_core.abstraction.cache import AIAgentCache


class MockAgent(AIAgentBase):
    """Mock agent for testing."""

    def build_init_kwargs(self):
        """Build initialization kwargs."""
        return {"model": self._config.model}

    async def initialize(self) -> None:
        self._initialized = True

    async def invoke(self, input_text: str, context=None, **kwargs):
        return AIAgentResponse(content="mock response", success=True)

    async def stream(self, input_text: str, context=None, **kwargs):
        yield "mock chunk"

    async def cleanup(self) -> None:
        self._initialized = False


class TestAIAgentCache:
    """Test AIAgentCache."""

    def test_cache_creation(self):
        """Test creating a cache."""
        cache = AIAgentCache()
        assert cache.size() == 0

    def test_cache_with_limits(self):
        """Test cache with size and TTL limits."""
        cache = AIAgentCache(max_size=10, ttl=3600)
        assert cache.size() == 0

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """Test setting and getting agents."""
        cache = AIAgentCache()
        config = AIAgentConfig(name="test", framework="test", model="test")
        agent = MockAgent(config)
        await agent.initialize()

        await cache.set("key1", agent)
        retrieved = await cache.get("key1")

        assert retrieved is agent
        assert cache.size() == 1

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self):
        """Test getting nonexistent agent returns None."""
        cache = AIAgentCache()
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_has(self):
        """Test checking if agent is cached."""
        cache = AIAgentCache()
        config = AIAgentConfig(name="test", framework="test", model="test")
        agent = MockAgent(config)
        await agent.initialize()

        assert not await cache.has("key1")
        await cache.set("key1", agent)
        assert await cache.has("key1")

    @pytest.mark.asyncio
    async def test_remove(self):
        """Test removing agent from cache."""
        cache = AIAgentCache()
        config = AIAgentConfig(name="test", framework="test", model="test")
        agent = MockAgent(config)
        await agent.initialize()

        await cache.set("key1", agent)
        assert cache.size() == 1

        await cache.remove("key1")
        assert cache.size() == 0
        assert not await cache.has("key1")

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing all agents."""
        cache = AIAgentCache()
        config = AIAgentConfig(name="test", framework="test", model="test")

        for i in range(3):
            agent = MockAgent(config)
            await agent.initialize()
            await cache.set(f"key{i}", agent)

        assert cache.size() == 3
        await cache.clear()
        assert cache.size() == 0

    @pytest.mark.asyncio
    async def test_max_size_eviction(self):
        """Test that cache evicts oldest agent when full."""
        cache = AIAgentCache(max_size=2)
        config = AIAgentConfig(name="test", framework="test", model="test")

        # Add first agent
        agent1 = MockAgent(config)
        await agent1.initialize()
        await cache.set("key1", agent1)
        assert cache.size() == 1

        # Add second agent
        agent2 = MockAgent(config)
        await agent2.initialize()
        await cache.set("key2", agent2)
        assert cache.size() == 2

        # Add third agent - should evict first
        agent3 = MockAgent(config)
        await agent3.initialize()
        await cache.set("key3", agent3)
        assert cache.size() == 2
        assert not await cache.has("key1")
        assert await cache.has("key2")
        assert await cache.has("key3")

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test cache as context manager."""
        config = AIAgentConfig(name="test", framework="test", model="test")

        async with AIAgentCache() as cache:
            agent = MockAgent(config)
            await agent.initialize()
            await cache.set("key1", agent)
            assert cache.size() == 1

        # Cache should be cleared after exiting context
        assert cache.size() == 0

    @pytest.mark.asyncio
    async def test_multiple_agents(self):
        """Test caching multiple agents."""
        cache = AIAgentCache()
        config = AIAgentConfig(name="test", framework="test", model="test")

        agents = []
        for i in range(5):
            agent = MockAgent(config)
            await agent.initialize()
            await cache.set(f"agent_{i}", agent)
            agents.append(agent)

        assert cache.size() == 5

        # Verify all agents are retrievable
        for i, agent in enumerate(agents):
            retrieved = await cache.get(f"agent_{i}")
            assert retrieved is agent

    @pytest.mark.asyncio
    async def test_thread_safety(self):
        """Test that cache operations are thread-safe."""
        import asyncio

        cache = AIAgentCache()
        config = AIAgentConfig(name="test", framework="test", model="test")

        async def add_agent(i):
            agent = MockAgent(config)
            await agent.initialize()
            await cache.set(f"agent_{i}", agent)

        # Run multiple concurrent operations
        tasks = [add_agent(i) for i in range(10)]
        await asyncio.gather(*tasks)

        assert cache.size() == 10
