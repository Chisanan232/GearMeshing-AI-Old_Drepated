"""Tests for adapter edge cases and error scenarios."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gearmeshing_ai.agent_core.abstraction.adapters.pydantic_ai import PydanticAIAgent
from gearmeshing_ai.agent_core.abstraction.base import AIAgentConfig


class TestAdapterEdgeCases:
    """Test edge cases in adapter implementations."""

    @pytest.mark.asyncio
    async def test_invoke_with_very_long_input(self):
        """Test invocation with very long input."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        # Create very long input
        long_input = "x" * 100000

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.data = "Response"
        mock_result.usage = None
        mock_agent.run = AsyncMock(return_value=mock_result)
        agent._agent = mock_agent

        response = await agent.invoke(long_input)

        assert response.success is True
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_with_special_characters(self):
        """Test invocation with special characters."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        special_inputs = [
            "Hello\nWorld",
            "Tab\tSeparated",
            "Unicode: 你好世界",
            "Emoji: 🎉🚀✨",
            "Special: !@#$%^&*()",
        ]

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.data = "Response"
        mock_result.usage = None
        mock_agent.run = AsyncMock(return_value=mock_result)
        agent._agent = mock_agent

        for special_input in special_inputs:
            response = await agent.invoke(special_input)
            assert response.success is True

    @pytest.mark.asyncio
    async def test_invoke_with_none_context_values(self):
        """Test invocation with None values in context."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.data = "Response"
        mock_result.usage = None
        mock_agent.run = AsyncMock(return_value=mock_result)
        agent._agent = mock_agent

        context = {"key1": None, "key2": "value", "key3": None}
        response = await agent.invoke("test", context=context)

        assert response.success is True

    @pytest.mark.asyncio
    async def test_invoke_with_empty_context(self):
        """Test invocation with empty context."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.data = "Response"
        mock_result.usage = None
        mock_agent.run = AsyncMock(return_value=mock_result)
        agent._agent = mock_agent

        response = await agent.invoke("test", context={})

        assert response.success is True

    @pytest.mark.asyncio
    async def test_invoke_with_small_timeout(self):
        """Test invocation with small timeout."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
            timeout=0.1,
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.data = "Response"
        mock_result.usage = None
        mock_agent.run = AsyncMock(return_value=mock_result)
        agent._agent = mock_agent

        response = await agent.invoke("test")

        assert response.success is True

    @pytest.mark.asyncio
    async def test_invoke_with_high_temperature(self):
        """Test invocation with high temperature."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
            temperature=2.0,
        )

        agent = PydanticAIAgent(config)

        with patch("pydantic_ai.Agent"):
            await agent.initialize()
            assert agent.is_initialized is True

    @pytest.mark.asyncio
    async def test_invoke_with_very_large_max_tokens(self):
        """Test invocation with very large max_tokens."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
            max_tokens=1000000,
        )

        agent = PydanticAIAgent(config)

        with patch("pydantic_ai.Agent"):
            await agent.initialize()
            assert agent.is_initialized is True

    @pytest.mark.asyncio
    async def test_multiple_cleanup_calls(self):
        """Test calling cleanup multiple times."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True
        agent._agent = MagicMock()

        # Call cleanup multiple times
        await agent.cleanup()
        await agent.cleanup()
        await agent.cleanup()

        assert agent._initialized is False

    @pytest.mark.asyncio
    async def test_invoke_after_cleanup(self):
        """Test that invoking after cleanup raises error."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)

        with patch("pydantic_ai.Agent"):
            await agent.initialize()
            await agent.cleanup()

            with pytest.raises(RuntimeError, match="Agent not initialized"):
                await agent.invoke("test")

    @pytest.mark.asyncio
    async def test_stream_after_cleanup(self):
        """Test that streaming after cleanup raises error."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)

        with patch("pydantic_ai.Agent"):
            await agent.initialize()
            await agent.cleanup()

            with pytest.raises(RuntimeError, match="Agent not initialized"):
                async for _ in agent.stream("test"):
                    pass

    @pytest.mark.asyncio
    async def test_config_with_empty_system_prompt(self):
        """Test agent with empty system prompt."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
            system_prompt="",
        )

        agent = PydanticAIAgent(config)

        with patch("pydantic_ai.Agent"):
            await agent.initialize()
            assert agent.is_initialized is True

    @pytest.mark.asyncio
    async def test_config_with_none_system_prompt(self):
        """Test agent with None system prompt."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
            system_prompt=None,
        )

        agent = PydanticAIAgent(config)

        with patch("pydantic_ai.Agent"):
            await agent.initialize()
            assert agent.is_initialized is True


class TestAdapterExceptionHandling:
    """Test exception handling in adapters."""

    @pytest.mark.asyncio
    async def test_invoke_with_timeout_error(self):
        """Test handling of timeout errors."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=TimeoutError("Request timed out"))
        agent._agent = mock_agent

        response = await agent.invoke("test")

        assert response.success is False
        assert "Request timed out" in response.error

    @pytest.mark.asyncio
    async def test_invoke_with_connection_error(self):
        """Test handling of connection errors."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=ConnectionError("Connection failed"))
        agent._agent = mock_agent

        response = await agent.invoke("test")

        assert response.success is False
        assert "Connection failed" in response.error

    @pytest.mark.asyncio
    async def test_invoke_with_value_error(self):
        """Test handling of value errors."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=ValueError("Invalid value"))
        agent._agent = mock_agent

        response = await agent.invoke("test")

        assert response.success is False
        assert "Invalid value" in response.error

    @pytest.mark.asyncio
    async def test_stream_with_exception(self):
        """Test streaming with exception."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        mock_agent = MagicMock()
        mock_agent.run_stream = MagicMock(side_effect=RuntimeError("Stream error"))
        agent._agent = mock_agent

        with pytest.raises(RuntimeError, match="Stream error"):
            async for _ in agent.stream("test"):
                pass

    @pytest.mark.asyncio
    async def test_initialize_with_attribute_error(self):
        """Test initialization with attribute error."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)

        with patch("pydantic_ai.Agent", side_effect=AttributeError("Missing attribute")):
            with pytest.raises(RuntimeError, match="Failed to initialize"):
                await agent.initialize()


class TestAdapterResourceManagement:
    """Test resource management in adapters."""

    @pytest.mark.asyncio
    async def test_agent_cleanup_releases_references(self):
        """Test that cleanup releases all references."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._model = MagicMock()

        await agent.cleanup()

        assert agent._agent is None
        assert agent._model is None
        assert agent._initialized is False

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_exception(self):
        """Test context manager cleanup on exception."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        with patch("pydantic_ai.Agent"):
            agent = PydanticAIAgent(config)

            try:
                async with agent:
                    assert agent.is_initialized is True
                    raise ValueError("Test error")
            except ValueError:
                pass

            assert agent.is_initialized is False

    @pytest.mark.asyncio
    async def test_multiple_context_manager_uses(self):
        """Test using agent context manager multiple times."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        with patch("pydantic_ai.Agent"):
            agent = PydanticAIAgent(config)

            # First use
            async with agent:
                assert agent.is_initialized is True

            assert agent.is_initialized is False

            # Second use
            async with agent:
                assert agent.is_initialized is True

            assert agent.is_initialized is False
