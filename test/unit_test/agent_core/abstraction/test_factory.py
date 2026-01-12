"""Unit tests for AI agent factory."""

import pytest

from gearmeshing_ai.agent_core.abstraction.base import (
    AIAgentBase,
    AIAgentConfig,
    AIAgentResponse,
)
from gearmeshing_ai.agent_core.abstraction.cache import AIAgentCache
from gearmeshing_ai.agent_core.abstraction.factory import AIAgentFactory


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

        configs = [AIAgentConfig(name=f"agent{i}", framework="mock", model="test") for i in range(3)]

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

    def test_register_factory_duplicate_raises_error(self):
        """Test that registering duplicate factory function raises ValueError (L73-74)."""
        factory = AIAgentFactory()

        def mock_factory(config):
            return MockAgent(config)

        factory.register_factory("mock", mock_factory)

        with pytest.raises(ValueError, match="already registered"):
            factory.register_factory("mock", mock_factory)

    def test_unregister_implementation_and_factory(self):
        """Test that unregister removes both implementation and factory (L82-84)."""
        factory = AIAgentFactory()

        # Register both implementation and factory
        factory.register("mock1", MockAgent)

        def mock_factory(config):
            return MockAgent(config)

        factory.register_factory("mock2", mock_factory)

        # Verify both are registered
        assert factory.is_registered("mock1")
        assert factory.is_registered("mock2")

        # Unregister both
        factory.unregister("mock1")
        factory.unregister("mock2")

        # Verify both are unregistered
        assert not factory.is_registered("mock1")
        assert not factory.is_registered("mock2")

    def test_unregister_nonexistent_framework(self):
        """Test that unregistering nonexistent framework doesn't raise error."""
        factory = AIAgentFactory()

        # Should not raise error
        factory.unregister("nonexistent")

        assert not factory.is_registered("nonexistent")

    @pytest.mark.asyncio
    async def test_create_agent_using_factory_function(self):
        """Test that create uses factory function when registered (L123-124)."""
        factory = AIAgentFactory()

        def mock_factory(config):
            agent = MockAgent(config)
            return agent

        factory.register_factory("mock", mock_factory)

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
    async def test_create_agent_using_implementation_class(self):
        """Test that create uses implementation class when factory not registered (L123-124)."""
        factory = AIAgentFactory()

        # Register only implementation, not factory
        factory.register("mock", MockAgent)

        config = AIAgentConfig(
            name="test",
            framework="mock",
            model="test",
        )

        agent = await factory.create(config, use_cache=False)

        assert agent is not None
        assert agent.is_initialized is True
        assert isinstance(agent, MockAgent)

    @pytest.mark.asyncio
    async def test_create_prefers_factory_function_over_implementation(self):
        """Test that factory function is preferred over implementation class."""
        factory = AIAgentFactory()

        # Register both factory function and implementation
        factory.register("mock", MockAgent)

        call_count = 0

        def mock_factory(config):
            nonlocal call_count
            call_count += 1
            agent = MockAgent(config)
            return agent

        factory.register_factory("mock", mock_factory)

        config = AIAgentConfig(
            name="test",
            framework="mock",
            model="test",
        )

        agent = await factory.create(config, use_cache=False)

        # Verify factory function was called
        assert call_count == 1
        assert agent is not None

    def test_register_factory_duplicate_with_different_functions(self):
        """Test that registering duplicate factory with different function raises error."""
        factory = AIAgentFactory()

        def factory_func1(config):
            return MockAgent(config)

        def factory_func2(config):
            return MockAgent(config)

        factory.register_factory("mock", factory_func1)

        with pytest.raises(ValueError, match="already registered"):
            factory.register_factory("mock", factory_func2)

    def test_unregister_removes_from_both_dicts(self):
        """Test that unregister removes from both _implementations and _factories."""
        factory = AIAgentFactory()

        factory.register("mock", MockAgent)

        def mock_factory(config):
            return MockAgent(config)

        factory.register_factory("mock_factory", mock_factory)

        # Verify registrations
        assert factory.is_registered("mock")
        assert factory.is_registered("mock_factory")

        # Unregister
        factory.unregister("mock")
        factory.unregister("mock_factory")

        # Verify both are gone
        assert not factory.is_registered("mock")
        assert not factory.is_registered("mock_factory")

    @pytest.mark.asyncio
    async def test_create_with_factory_function_initialization(self):
        """Test that agent created via factory function is properly initialized."""
        factory = AIAgentFactory()

        initialized_agents = []

        def tracking_factory(config):
            agent = MockAgent(config)
            initialized_agents.append(agent)
            return agent

        factory.register_factory("mock", tracking_factory)

        config = AIAgentConfig(
            name="test",
            framework="mock",
            model="test",
        )

        agent = await factory.create(config, use_cache=False)

        assert len(initialized_agents) == 1
        assert agent.is_initialized is True
        assert agent in initialized_agents
