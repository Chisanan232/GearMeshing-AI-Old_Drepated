"""Unit tests for AI agent factory."""

import pytest
from unittest.mock import AsyncMock

from gearmeshing_ai.agent_core.abstraction.base import AIAgentBase, AIAgentConfig
from gearmeshing_ai.agent_core.abstraction.factory import AIAgentFactory
from gearmeshing_ai.agent_core.abstraction.cache import AIAgentCache


class MockAgent(AIAgentBase):
    """Mock agent for testing."""

    async def initialize(self) -> None:
        self._initialized = True

    async def invoke(self, input_text: str, context=None, **kwargs):
        from gearmeshing_ai.agent_core.abstraction.base import AIAgentResponse
        return AIAgentResponse(content="mock response", success=True)

    async def stream(self, input_text: str, context=None, **kwargs):
        yield "mock chunk"

    async def cleanup(self) -> None:
        self._initialized = False


class TestAIAgentFactory:
    """Test AIAgentFactory."""

    def test_factory_creation(self):
        """Test creating a factory."""
        factory = AIAgentFactory()
        assert factory.get_cache().size() == 0
        assert factory.get_registered_frameworks() == []

    def test_register_implementation(self):
        """Test registering an implementation."""
        factory = AIAgentFactory()
        factory.register("mock", MockAgent)

        assert factory.is_registered("mock")
        assert "mock" in factory.get_registered_frameworks()

    def test_register_duplicate_raises_error(self):
        """Test that registering duplicate framework raises error."""
        factory = AIAgentFactory()
        factory.register("mock", MockAgent)

        with pytest.raises(ValueError, match="already registered"):
            factory.register("mock", MockAgent)

    def test_register_factory_function(self):
        """Test registering a factory function."""
        factory = AIAgentFactory()

        def mock_factory(config):
            return MockAgent(config)

        factory.register_factory("mock", mock_factory)
        assert factory.is_registered("mock")

    @pytest.mark.asyncio
    async def test_create_agent(self):
        """Test creating an agent."""
        factory = AIAgentFactory()
        factory.register("mock", MockAgent)

        config = AIAgentConfig(
            name="test",
            framework="mock",
            model="test",
        )

        agent = await factory.create(config, use_cache=False)
        assert agent is not None
        assert agent.is_initialized is True
        assert agent.config.name == "test"

    @pytest.mark.asyncio
    async def test_create_unregistered_framework_raises_error(self):
        """Test that creating with unregistered framework raises error."""
        factory = AIAgentFactory()

        config = AIAgentConfig(
            name="test",
            framework="unknown",
            model="test",
        )

        with pytest.raises(ValueError, match="not registered"):
            await factory.create(config, use_cache=False)

    @pytest.mark.asyncio
    async def test_cache_functionality(self):
        """Test that factory caches agents."""
        cache = AIAgentCache()
        factory = AIAgentFactory(cache=cache)
        factory.register("mock", MockAgent)

        config = AIAgentConfig(
            name="test",
            framework="mock",
            model="test",
        )

        # Create first agent
        agent1 = await factory.create(config, use_cache=True)
        assert factory.get_cache().size() == 1

        # Create second agent (should be cached)
        agent2 = await factory.create(config, use_cache=True)
        assert agent1 is agent2
        assert factory.get_cache().size() == 1

    @pytest.mark.asyncio
    async def test_create_batch(self):
        """Test creating multiple agents."""
        factory = AIAgentFactory()
        factory.register("mock", MockAgent)

        configs = [
            AIAgentConfig(name=f"agent{i}", framework="mock", model="test")
            for i in range(3)
        ]

        agents = await factory.create_batch(configs, use_cache=False)
        assert len(agents) == 3
        assert all(a.is_initialized for a in agents)

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Test clearing the cache."""
        cache = AIAgentCache()
        factory = AIAgentFactory(cache=cache)
        factory.register("mock", MockAgent)

        config = AIAgentConfig(
            name="test",
            framework="mock",
            model="test",
        )

        await factory.create(config, use_cache=True)
        assert factory.get_cache().size() == 1

        await factory.clear_cache()
        assert factory.get_cache().size() == 0

    @pytest.mark.asyncio
    async def test_get_cached_agent(self):
        """Test getting a cached agent."""
        factory = AIAgentFactory()
        factory.register("mock", MockAgent)

        config = AIAgentConfig(
            name="test",
            framework="mock",
            model="test",
        )

        agent = await factory.create(config, use_cache=True)
        cached = await factory.get_cached("test", "mock")
        assert cached is agent

    @pytest.mark.asyncio
    async def test_factory_context_manager(self):
        """Test factory as context manager."""
        factory = AIAgentFactory()
        factory.register("mock", MockAgent)

        config = AIAgentConfig(
            name="test",
            framework="mock",
            model="test",
        )

        async with factory as f:
            agent = await f.create(config, use_cache=True)
            assert agent.is_initialized is True
