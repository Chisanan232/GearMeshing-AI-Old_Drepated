"""Tests for adapter logging and monitoring."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gearmeshing_ai.agent_core.abstraction.adapters.pydantic_ai import PydanticAIAgent
from gearmeshing_ai.agent_core.abstraction.base import AIAgentConfig


class TestAdapterLogging:
    """Test logging in adapters."""

    @pytest.mark.asyncio
    async def test_initialize_logging(self):
        """Test that initialization is logged."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)

        with patch("pydantic_ai.Agent"):
            # Logger is already instantiated in the module, so we patch it directly
            with patch("gearmeshing_ai.agent_core.abstraction.adapters.pydantic_ai.logger") as mock_logger:
                await agent.initialize()

                # Verify debug logs were called
                assert mock_logger.debug.called

    @pytest.mark.asyncio
    async def test_invoke_logging(self):
        """Test that invocation is logged."""
        config = AIAgentConfig(
            name="test_agent",
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

        with patch("gearmeshing_ai.agent_core.abstraction.adapters.pydantic_ai.logger") as mock_logger:
            response = await agent.invoke("test")

            assert response.success is True
            assert mock_logger.debug.called

    @pytest.mark.asyncio
    async def test_invoke_error_logging(self):
        """Test that invocation errors are logged."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=Exception("Test error"))
        agent._agent = mock_agent

        with patch("gearmeshing_ai.agent_core.abstraction.adapters.pydantic_ai.logger") as mock_logger:
            response = await agent.invoke("test")

            assert response.success is False
            assert mock_logger.error.called

    @pytest.mark.asyncio
    async def test_stream_logging(self):
        """Test that streaming is logged."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        async def mock_stream():
            yield "chunk"

        mock_agent = MagicMock()
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_context)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_context.__aiter__ = MagicMock(return_value=mock_stream().__aiter__())
        mock_agent.run_stream = MagicMock(return_value=mock_context)
        agent._agent = mock_agent

        with patch("gearmeshing_ai.agent_core.abstraction.adapters.pydantic_ai.logger") as mock_logger:
            async for _ in agent.stream("test"):
                pass

            assert mock_logger.debug.called

    @pytest.mark.asyncio
    async def test_cleanup_logging(self):
        """Test that cleanup is logged."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True
        agent._agent = MagicMock()

        with patch("gearmeshing_ai.agent_core.abstraction.adapters.pydantic_ai.logger") as mock_logger:
            await agent.cleanup()

            assert mock_logger.debug.called

    @pytest.mark.asyncio
    async def test_initialization_error_handling(self):
        """Test that initialization errors are handled properly."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)

        with patch("pydantic_ai.Agent", side_effect=Exception("Init failed")):
            # Verify that initialization errors are raised as RuntimeError
            with pytest.raises(RuntimeError, match="Failed to initialize"):
                await agent.initialize()

    @pytest.mark.asyncio
    async def test_stream_error_logging(self):
        """Test that streaming errors are logged."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        mock_agent = MagicMock()
        mock_agent.run_stream = MagicMock(side_effect=Exception("Stream error"))
        agent._agent = mock_agent

        with patch("gearmeshing_ai.agent_core.abstraction.adapters.pydantic_ai.logger") as mock_logger:
            with pytest.raises(Exception):
                async for _ in agent.stream("test"):
                    pass

            assert mock_logger.error.called


class TestAdapterMonitoring:
    """Test monitoring capabilities in adapters."""

    @pytest.mark.asyncio
    async def test_response_metadata_includes_model(self):
        """Test that response metadata includes model info."""
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

        response = await agent.invoke("test")

        assert response.metadata["model"] == "gpt-4o"
        assert response.metadata["framework"] == "pydantic_ai"

    @pytest.mark.asyncio
    async def test_response_metadata_includes_usage(self):
        """Test that response metadata includes usage info."""
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
        mock_result.usage = MagicMock(
            input_tokens=100,
            output_tokens=50,
        )
        mock_agent.run = AsyncMock(return_value=mock_result)
        agent._agent = mock_agent

        response = await agent.invoke("test")

        assert "usage" in response.metadata
        assert response.metadata["usage"]["input_tokens"] == 100
        assert response.metadata["usage"]["output_tokens"] == 50

    @pytest.mark.asyncio
    async def test_agent_config_accessible_for_monitoring(self):
        """Test that agent config is accessible for monitoring."""
        config = AIAgentConfig(
            name="monitored_agent",
            framework="pydantic_ai",
            model="gpt-4o",
            temperature=0.5,
            max_tokens=1000,
        )

        agent = PydanticAIAgent(config)

        # Config should be accessible
        assert agent.config.name == "monitored_agent"
        assert agent.config.framework == "pydantic_ai"
        assert agent.config.model == "gpt-4o"
        assert agent.config.temperature == 0.5
        assert agent.config.max_tokens == 1000

    @pytest.mark.asyncio
    async def test_agent_initialization_status_tracking(self):
        """Test that initialization status can be tracked."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)

        assert agent.is_initialized is False

        with patch("pydantic_ai.Agent"):
            await agent.initialize()
            assert agent.is_initialized is True

            await agent.cleanup()
            assert agent.is_initialized is False

    @pytest.mark.asyncio
    async def test_agent_framework_property(self):
        """Test that agent framework can be monitored."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)

        assert agent.framework == "pydantic_ai"

    @pytest.mark.asyncio
    async def test_agent_model_property(self):
        """Test that agent model can be monitored."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)

        assert agent.model == "gpt-4o"


class TestAdapterPerformanceMonitoring:
    """Test performance monitoring in adapters."""

    @pytest.mark.asyncio
    async def test_invoke_response_includes_metadata(self):
        """Test that invoke response includes metadata for monitoring."""
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
        mock_result.usage = MagicMock(
            input_tokens=50,
            output_tokens=25,
        )
        mock_agent.run = AsyncMock(return_value=mock_result)
        agent._agent = mock_agent

        response = await agent.invoke("test")

        # Metadata should be present for monitoring
        assert isinstance(response.metadata, dict)
        assert "model" in response.metadata
        assert "framework" in response.metadata
        assert "usage" in response.metadata

    @pytest.mark.asyncio
    async def test_error_response_includes_error_info(self):
        """Test that error responses include error information."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=ValueError("Invalid input"))
        agent._agent = mock_agent

        response = await agent.invoke("test")

        assert response.success is False
        assert response.error is not None
        assert "Invalid input" in response.error

    @pytest.mark.asyncio
    async def test_response_success_status(self):
        """Test that response success status is properly set."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        # Test successful response
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.data = "Response"
        mock_result.usage = None
        mock_agent.run = AsyncMock(return_value=mock_result)
        agent._agent = mock_agent

        response = await agent.invoke("test")
        assert response.success is True

        # Test error response
        mock_agent.run = AsyncMock(side_effect=Exception("Error"))
        response = await agent.invoke("test")
        assert response.success is False

    @pytest.mark.asyncio
    async def test_tool_calls_tracking(self):
        """Test that tool calls are tracked in response."""
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
        mock_result.tool_calls = [
            {"name": "tool1", "args": {"arg": "value"}},
        ]
        mock_result.usage = None
        mock_agent.run = AsyncMock(return_value=mock_result)
        agent._agent = mock_agent

        response = await agent.invoke("test")

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "tool1"
