"""Unit tests for AI agent provider."""

from unittest.mock import patch

import pytest

from gearmeshing_ai.agent_core.abstraction.base import (
    AIAgentBase,
    AIAgentConfig,
    AIAgentResponse,
)
from gearmeshing_ai.agent_core.abstraction.factory import AIAgentFactory
from gearmeshing_ai.agent_core.abstraction.provider import (
    AIAgentProvider,
    get_agent_provider,
    initialize_agent_provider,
    reset_agent_provider,
    set_agent_provider,
)


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


class TestAIAgentProvider:
    """Test AIAgentProvider."""

    def test_provider_creation(self):
        """Test creating a provider."""
        provider = AIAgentProvider()
        assert provider.get_framework() is None
        assert provider.get_factory() is None

    def test_set_framework(self):
        """Test setting framework."""
        factory = AIAgentFactory()
        factory.register("mock", MockAgent)

        provider = AIAgentProvider()
        provider.set_factory(factory)
        provider.set_framework("mock")

        assert provider.get_framework() == "mock"

    def test_set_unregistered_framework_raises_error(self):
        """Test that setting unregistered framework raises error."""
        factory = AIAgentFactory()
        provider = AIAgentProvider()
        provider.set_factory(factory)

        with pytest.raises(ValueError, match="not registered"):
            provider.set_framework("unknown")

    def test_set_framework_without_factory_raises_error(self):
        """Test that setting framework without factory raises error."""
        provider = AIAgentProvider()

        with pytest.raises(RuntimeError, match="Factory not initialized"):
            provider.set_framework("mock")

    @pytest.mark.asyncio
    async def test_create_agent(self):
        """Test creating an agent through provider."""
        factory = AIAgentFactory()
        factory.register("mock", MockAgent)

        provider = AIAgentProvider()
        provider.set_factory(factory)
        provider.set_framework("mock")

        config = AIAgentConfig(
            name="test",
            framework="other",  # Should be overridden
            model="test",
        )

        agent = await provider.create_agent(config, use_cache=False)
        assert agent is not None
        assert agent.is_initialized is True
        assert agent.framework == "mock"  # Should be overridden

    @pytest.mark.asyncio
    async def test_create_agent_without_framework_raises_error(self):
        """Test that creating agent without framework raises error."""
        factory = AIAgentFactory()
        provider = AIAgentProvider()
        provider.set_factory(factory)

        config = AIAgentConfig(
            name="test",
            framework="mock",
            model="test",
        )

        with pytest.raises(RuntimeError, match="Framework not set"):
            await provider.create_agent(config)

    def test_get_registered_frameworks(self):
        """Test getting registered frameworks."""
        factory = AIAgentFactory()
        factory.register("mock1", MockAgent)
        factory.register("mock2", MockAgent)

        provider = AIAgentProvider()
        provider.set_factory(factory)

        frameworks = provider.get_registered_frameworks()
        assert "mock1" in frameworks
        assert "mock2" in frameworks

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Test clearing cache through provider."""
        factory = AIAgentFactory()
        factory.register("mock", MockAgent)

        provider = AIAgentProvider()
        provider.set_factory(factory)
        provider.set_framework("mock")

        config = AIAgentConfig(
            name="test",
            framework="mock",
            model="test",
        )

        await provider.create_agent(config, use_cache=True)
        assert factory.get_cache().size() == 1

        await provider.clear_cache()
        assert factory.get_cache().size() == 0

    def test_provider_repr(self):
        """Test provider string representation."""
        factory = AIAgentFactory()
        factory.register("mock", MockAgent)

        provider = AIAgentProvider()
        provider.set_factory(factory)
        provider.set_framework("mock")

        repr_str = repr(provider)
        assert "AIAgentProvider" in repr_str
        assert "mock" in repr_str


class TestGlobalProvider:
    """Test global provider functions."""

    def teardown_method(self):
        """Clean up after each test."""
        reset_agent_provider()

    def test_initialize_agent_provider(self):
        """Test initializing global provider."""
        factory = AIAgentFactory()
        factory.register("mock", MockAgent)

        provider = initialize_agent_provider(factory=factory, framework="mock")
        assert provider.get_framework() == "mock"

    def test_get_agent_provider(self):
        """Test getting global provider."""
        factory = AIAgentFactory()
        factory.register("mock", MockAgent)

        initialize_agent_provider(factory=factory)
        provider = get_agent_provider()
        assert provider is not None

    def test_get_agent_provider_without_init_raises_error(self):
        """Test that getting provider without init raises error."""
        reset_agent_provider()

        with pytest.raises(RuntimeError, match="not initialized"):
            get_agent_provider()

    def test_set_agent_provider(self):
        """Test setting global provider."""
        provider = AIAgentProvider()
        set_agent_provider(provider)

        retrieved = get_agent_provider()
        assert retrieved is provider

    @pytest.mark.asyncio
    async def test_initialize_from_env(self):
        """Test initializing from settings."""
        factory = AIAgentFactory()
        factory.register("mock", MockAgent)

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.gearmeshing_ai_agent_framework = "mock"
            provider = initialize_agent_provider(factory=factory)
            assert provider.get_framework() == "mock"

    def test_reset_agent_provider(self):
        """Test resetting global provider."""
        factory = AIAgentFactory()
        initialize_agent_provider(factory=factory)

        reset_agent_provider()

        with pytest.raises(RuntimeError):
            get_agent_provider()

    def test_create_agent_without_factory_raises_error(self):
        """Test that create_agent raises RuntimeError when factory is None (L93-94)."""
        provider = AIAgentProvider()
        # Factory is not set, so _factory is None

        config = AIAgentConfig(
            name="test",
            framework="mock",
            model="test",
        )

        with pytest.raises(RuntimeError, match="Factory not initialized"):
            import asyncio

            asyncio.run(provider.create_agent(config))

    def test_get_registered_frameworks_without_factory_returns_empty_list(self):
        """Test that get_registered_frameworks returns empty list when factory is None (L120-121)."""
        provider = AIAgentProvider()
        # Factory is not set, so _factory is None

        frameworks = provider.get_registered_frameworks()

        assert frameworks == []
        assert isinstance(frameworks, list)

    def test_initialize_agent_provider_creates_factory_if_none(self):
        """Test that initialize_agent_provider creates factory if not provided (L184-185)."""
        reset_agent_provider()

        # Call without providing factory
        provider = initialize_agent_provider(factory=None, framework=None)

        assert provider is not None
        assert provider.get_factory() is not None
        # Verify factory was created
        assert isinstance(provider.get_factory(), AIAgentFactory)

    def test_initialize_agent_provider_with_explicit_factory(self):
        """Test initialize_agent_provider with explicitly provided factory."""
        reset_agent_provider()

        factory = AIAgentFactory()
        factory.register("mock", MockAgent)

        provider = initialize_agent_provider(factory=factory, framework="mock")

        assert provider.get_factory() is factory
        assert provider.get_framework() == "mock"

    def test_provider_create_agent_without_framework_raises_error(self):
        """Test that create_agent raises RuntimeError when framework is not set."""
        factory = AIAgentFactory()
        factory.register("mock", MockAgent)

        provider = AIAgentProvider()
        provider.set_factory(factory)
        # Framework is not set

        config = AIAgentConfig(
            name="test",
            framework="mock",
            model="test",
        )

        with pytest.raises(RuntimeError, match="Framework not set"):
            import asyncio

            asyncio.run(provider.create_agent(config))

    def test_provider_get_registered_frameworks_with_factory(self):
        """Test get_registered_frameworks returns frameworks when factory is set."""
        factory = AIAgentFactory()
        factory.register("mock1", MockAgent)
        factory.register("mock2", MockAgent)

        provider = AIAgentProvider()
        provider.set_factory(factory)

        frameworks = provider.get_registered_frameworks()

        assert len(frameworks) == 2
        assert "mock1" in frameworks
        assert "mock2" in frameworks

    @pytest.mark.asyncio
    async def test_provider_create_agent_overrides_framework_in_config(self):
        """Test that create_agent overrides framework in config with active framework."""
        factory = AIAgentFactory()
        factory.register("mock", MockAgent)

        provider = AIAgentProvider()
        provider.set_factory(factory)
        provider.set_framework("mock")

        config = AIAgentConfig(
            name="test",
            framework="different_framework",  # This should be overridden
            model="test",
        )

        agent = await provider.create_agent(config, use_cache=False)

        # Verify framework was overridden
        assert agent.framework == "mock"
        assert agent.config.framework == "mock"
