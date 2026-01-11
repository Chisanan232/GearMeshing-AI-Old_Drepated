"""Integration tests for adapters with factory."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from gearmeshing_ai.agent_core.abstraction.base import AIAgentConfig
from gearmeshing_ai.agent_core.abstraction.factory import AIAgentFactory
from gearmeshing_ai.agent_core.abstraction.adapters.pydantic_ai import PydanticAIAgent


class TestAdapterFactoryIntegration:
    """Test adapter integration with factory."""

    def test_register_pydantic_ai_adapter(self):
        """Test registering Pydantic AI adapter."""
        factory = AIAgentFactory()
        factory.register("pydantic_ai", PydanticAIAgent)

        assert factory.is_registered("pydantic_ai")
        assert "pydantic_ai" in factory.get_registered_frameworks()

    @pytest.mark.asyncio
    async def test_create_agent_with_pydantic_ai_adapter(self):
        """Test creating agent with Pydantic AI adapter."""
        factory = AIAgentFactory()
        factory.register("pydantic_ai", PydanticAIAgent)

        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        with patch("pydantic_ai.Agent"):
            agent = await factory.create(config, use_cache=False)

            assert agent is not None
            assert isinstance(agent, PydanticAIAgent)
            assert agent.is_initialized is True

    @pytest.mark.asyncio
    async def test_create_multiple_agents_with_adapter(self):
        """Test creating multiple agents with adapter."""
        factory = AIAgentFactory()
        factory.register("pydantic_ai", PydanticAIAgent)

        configs = [
            AIAgentConfig(
                name=f"agent_{i}",
                framework="pydantic_ai",
                model="gpt-4o",
            )
            for i in range(3)
        ]

        with patch("pydantic_ai.Agent"):
            agents = await factory.create_batch(configs, use_cache=False)

            assert len(agents) == 3
            assert all(isinstance(a, PydanticAIAgent) for a in agents)
            assert all(a.is_initialized for a in agents)

    @pytest.mark.asyncio
    async def test_adapter_with_caching(self):
        """Test adapter with factory caching."""
        factory = AIAgentFactory()
        factory.register("pydantic_ai", PydanticAIAgent)

        config = AIAgentConfig(
            name="cached_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        with patch("pydantic_ai.Agent"):
            agent1 = await factory.create(config, use_cache=True)
            agent2 = await factory.create(config, use_cache=True)

            assert agent1 is agent2
            assert factory.get_cache().size() == 1

    @pytest.mark.asyncio
    async def test_adapter_without_caching(self):
        """Test adapter without factory caching."""
        factory = AIAgentFactory()
        factory.register("pydantic_ai", PydanticAIAgent)

        config = AIAgentConfig(
            name="uncached_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        with patch("pydantic_ai.Agent"):
            agent1 = await factory.create(config, use_cache=False)
            agent2 = await factory.create(config, use_cache=False)

            assert agent1 is not agent2
            assert factory.get_cache().size() == 0

    @pytest.mark.asyncio
    async def test_adapter_initialization_error_handling(self):
        """Test adapter error handling during factory creation."""
        factory = AIAgentFactory()
        factory.register("pydantic_ai", PydanticAIAgent)

        config = AIAgentConfig(
            name="error_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        with patch("pydantic_ai.Agent", side_effect=Exception("Init failed")):
            with pytest.raises(RuntimeError):
                await factory.create(config, use_cache=False)


class TestAdapterWithProvider:
    """Test adapter integration with provider."""

    @pytest.mark.asyncio
    async def test_create_agent_via_provider(self):
        """Test creating agent through provider."""
        from gearmeshing_ai.agent_core.abstraction.provider import AIAgentProvider

        factory = AIAgentFactory()
        factory.register("pydantic_ai", PydanticAIAgent)

        provider = AIAgentProvider()
        provider.set_factory(factory)
        provider.set_framework("pydantic_ai")

        config = AIAgentConfig(
            name="provider_agent",
            framework="other",  # Should be overridden
            model="gpt-4o",
        )

        with patch("pydantic_ai.Agent"):
            agent = await provider.create_agent(config)

            assert isinstance(agent, PydanticAIAgent)
            assert agent.framework == "pydantic_ai"

    @pytest.mark.asyncio
    async def test_provider_framework_override(self):
        """Test that provider overrides agent config framework."""
        from gearmeshing_ai.agent_core.abstraction.provider import AIAgentProvider

        factory = AIAgentFactory()
        factory.register("pydantic_ai", PydanticAIAgent)

        provider = AIAgentProvider()
        provider.set_factory(factory)
        provider.set_framework("pydantic_ai")

        config = AIAgentConfig(
            name="override_agent",
            framework="langchain",  # Different framework
            model="gpt-4o",
        )

        with patch("pydantic_ai.Agent"):
            agent = await provider.create_agent(config)

            # Provider should override to pydantic_ai
            assert agent.framework == "pydantic_ai"


class TestAdapterConfigurationVariations:
    """Test adapters with various configurations."""

    @pytest.mark.asyncio
    async def test_adapter_with_minimal_config(self):
        """Test adapter with minimal configuration."""
        factory = AIAgentFactory()
        factory.register("pydantic_ai", PydanticAIAgent)

        config = AIAgentConfig(
            name="minimal",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        with patch("pydantic_ai.Agent"):
            agent = await factory.create(config, use_cache=False)
            assert agent.is_initialized is True

    @pytest.mark.asyncio
    async def test_adapter_with_full_config(self):
        """Test adapter with full configuration."""
        factory = AIAgentFactory()
        factory.register("pydantic_ai", PydanticAIAgent)

        config = AIAgentConfig(
            name="full_config",
            framework="pydantic_ai",
            model="gpt-4o",
            system_prompt="You are helpful",
            temperature=0.5,
            max_tokens=1000,
            timeout=30,
            tools=[
                {"name": "tool1", "description": "Tool 1"},
                {"name": "tool2", "description": "Tool 2"},
            ],
            metadata={"custom": "value"},
        )

        with patch("pydantic_ai.Agent"):
            agent = await factory.create(config, use_cache=False)

            assert agent.config.system_prompt == "You are helpful"
            assert agent.config.temperature == 0.5
            assert agent.config.max_tokens == 1000
            assert agent.config.timeout == 30
            assert len(agent.config.tools) == 2
            assert agent.config.metadata["custom"] == "value"

    @pytest.mark.asyncio
    async def test_adapter_with_different_models(self):
        """Test adapter with different model configurations."""
        factory = AIAgentFactory()
        factory.register("pydantic_ai", PydanticAIAgent)

        models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]

        with patch("pydantic_ai.Agent"):
            for model in models:
                config = AIAgentConfig(
                    name=f"agent_{model}",
                    framework="pydantic_ai",
                    model=model,
                )

                agent = await factory.create(config, use_cache=False)
                assert agent.model == model

    @pytest.mark.asyncio
    async def test_adapter_with_temperature_variations(self):
        """Test adapter with different temperature settings."""
        factory = AIAgentFactory()
        factory.register("pydantic_ai", PydanticAIAgent)

        temperatures = [0.0, 0.3, 0.7, 1.0, 2.0]

        with patch("pydantic_ai.Agent"):
            for temp in temperatures:
                config = AIAgentConfig(
                    name=f"agent_temp_{temp}",
                    framework="pydantic_ai",
                    model="gpt-4o",
                    temperature=temp,
                )

                agent = await factory.create(config, use_cache=False)
                assert agent.config.temperature == temp


class TestAdapterErrorScenarios:
    """Test adapter error handling scenarios."""

    @pytest.mark.asyncio
    async def test_adapter_with_invalid_config(self):
        """Test adapter with invalid configuration."""
        factory = AIAgentFactory()
        factory.register("pydantic_ai", PydanticAIAgent)

        # Missing required fields should be caught by AIAgentConfig validation
        # This tests that the adapter handles it gracefully
        config = AIAgentConfig(
            name="",  # Empty name
            framework="pydantic_ai",
            model="gpt-4o",
        )

        with patch("pydantic_ai.Agent"):
            agent = await factory.create(config, use_cache=False)
            # Should still create agent, validation is at config level
            assert agent is not None

    @pytest.mark.asyncio
    async def test_adapter_cleanup_on_factory_clear(self):
        """Test that adapters are cleaned up when factory cache is cleared."""
        factory = AIAgentFactory()
        factory.register("pydantic_ai", PydanticAIAgent)

        config = AIAgentConfig(
            name="cleanup_test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        with patch("pydantic_ai.Agent"):
            agent = await factory.create(config, use_cache=True)
            assert factory.get_cache().size() == 1

            await factory.clear_cache()
            assert factory.get_cache().size() == 0

    @pytest.mark.asyncio
    async def test_adapter_concurrent_creation(self):
        """Test concurrent adapter creation."""
        import asyncio

        factory = AIAgentFactory()
        factory.register("pydantic_ai", PydanticAIAgent)

        async def create_agent(i):
            config = AIAgentConfig(
                name=f"concurrent_{i}",
                framework="pydantic_ai",
                model="gpt-4o",
            )
            with patch("pydantic_ai.Agent"):
                return await factory.create(config, use_cache=False)

        with patch("pydantic_ai.Agent"):
            # Use asyncio.gather instead of asyncio.run since we're already in an async context
            agents = await asyncio.gather(*[create_agent(i) for i in range(5)])

            assert len(agents) == 5
            assert all(isinstance(a, PydanticAIAgent) for a in agents)
