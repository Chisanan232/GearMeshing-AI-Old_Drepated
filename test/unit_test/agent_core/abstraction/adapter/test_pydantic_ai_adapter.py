"""Unit tests for Pydantic AI adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gearmeshing_ai.agent_core.abstraction.adapters.pydantic_ai import PydanticAIAgent
from gearmeshing_ai.agent_core.abstraction.base import AIAgentConfig


class TestPydanticAIAgentInitialization:
    """Test Pydantic AI agent initialization."""

    def test_agent_creation(self):
        """Test creating a Pydantic AI agent."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
            system_prompt="You are helpful",
        )

        agent = PydanticAIAgent(config)

        assert agent.config == config
        assert agent.framework == "pydantic_ai"
        assert agent.model == "gpt-4o"
        assert agent.is_initialized is False

    def test_agent_with_all_config_options(self):
        """Test creating agent with all configuration options."""
        config = AIAgentConfig(
            name="full_config_agent",
            framework="pydantic_ai",
            model="gpt-4o",
            system_prompt="You are helpful",
            temperature=0.5,
            max_tokens=1000,
            timeout=30,
            tools=[{"name": "tool1", "description": "A tool"}],
            metadata={"custom": "value"},
        )

        agent = PydanticAIAgent(config)

        assert agent.config.temperature == 0.5
        assert agent.config.max_tokens == 1000
        assert agent.config.timeout == 30
        assert len(agent.config.tools) == 1
        assert agent.config.metadata["custom"] == "value"

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful agent initialization."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
            system_prompt="You are helpful",
        )

        agent = PydanticAIAgent(config)

        with patch("pydantic_ai.Agent") as mock_agent_class:
            mock_agent_instance = MagicMock()
            mock_agent_class.return_value = mock_agent_instance

            await agent.initialize()

            assert agent.is_initialized is True
            mock_agent_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_with_temperature(self):
        """Test initialization with temperature setting."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
            temperature=0.3,
        )

        agent = PydanticAIAgent(config)

        with patch("pydantic_ai.Agent") as mock_agent_class:
            mock_agent_instance = MagicMock()
            mock_agent_class.return_value = mock_agent_instance

            await agent.initialize()

            assert agent.is_initialized is True

    @pytest.mark.asyncio
    async def test_initialize_with_max_tokens(self):
        """Test initialization with max_tokens setting."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
            max_tokens=500,
        )

        agent = PydanticAIAgent(config)

        with patch("pydantic_ai.Agent") as mock_agent_class:
            mock_agent_instance = MagicMock()
            mock_agent_class.return_value = mock_agent_instance

            await agent.initialize()

            assert agent.is_initialized is True

    @pytest.mark.asyncio
    async def test_initialize_import_error(self):
        """Test initialization when pydantic_ai is not installed."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)

        with patch("pydantic_ai.Agent", side_effect=ImportError("Not installed")):
            with pytest.raises(RuntimeError, match="Pydantic AI is not installed"):
                await agent.initialize()

    @pytest.mark.asyncio
    async def test_initialize_general_error(self):
        """Test initialization with general error."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)

        with patch("pydantic_ai.Agent", side_effect=Exception("Setup failed")):
            with pytest.raises(RuntimeError, match="Failed to initialize"):
                await agent.initialize()


class TestPydanticAIAgentInvoke:
    """Test Pydantic AI agent invocation."""

    @pytest.mark.asyncio
    async def test_invoke_success(self):
        """Test successful agent invocation."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.data = "Test response"
        mock_result.usage = MagicMock(input_tokens=10, output_tokens=20)
        mock_agent.run = AsyncMock(return_value=mock_result)
        agent._agent = mock_agent

        response = await agent.invoke("Hello!")

        assert response.success is True
        assert response.content == "Test response"
        assert response.error is None
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_with_context(self):
        """Test invocation with context."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.data = "Test response"
        mock_result.usage = None
        mock_agent.run = AsyncMock(return_value=mock_result)
        agent._agent = mock_agent

        context = {"user_id": "123", "language": "en"}
        response = await agent.invoke("Hello!", context=context)

        assert response.success is True
        # Verify context was included in the prompt
        call_args = mock_agent.run.call_args
        assert "user_id" in str(call_args)

    @pytest.mark.asyncio
    async def test_invoke_with_timeout(self):
        """Test invocation with timeout."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
            timeout=15,
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.data = "Test response"
        mock_result.usage = None
        mock_agent.run = AsyncMock(return_value=mock_result)
        agent._agent = mock_agent

        response = await agent.invoke("Hello!", timeout=20)

        assert response.success is True
        # Verify timeout was passed
        call_kwargs = mock_agent.run.call_args[1]
        assert call_kwargs.get("timeout") == 20

    @pytest.mark.asyncio
    async def test_invoke_not_initialized_raises_error(self):
        """Test that invoking uninitialized agent raises error."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)

        with pytest.raises(RuntimeError, match="Agent not initialized"):
            await agent.invoke("Hello!")

    @pytest.mark.asyncio
    async def test_invoke_with_tool_calls(self):
        """Test invocation that returns tool calls."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.data = "Test response"
        mock_result.tool_calls = [
            {"name": "tool1", "args": {"arg1": "value1"}},
            {"name": "tool2", "args": {"arg2": "value2"}},
        ]
        mock_result.usage = None
        mock_agent.run = AsyncMock(return_value=mock_result)
        agent._agent = mock_agent

        response = await agent.invoke("Hello!")

        assert response.success is True
        assert len(response.tool_calls) == 2
        assert response.tool_calls[0]["name"] == "tool1"

    @pytest.mark.asyncio
    async def test_invoke_error_handling(self):
        """Test error handling during invocation."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=Exception("API error"))
        agent._agent = mock_agent

        response = await agent.invoke("Hello!")

        assert response.success is False
        assert response.error == "API error"
        assert response.content is None

    @pytest.mark.asyncio
    async def test_invoke_with_metadata(self):
        """Test that response includes metadata."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.data = "Test response"
        mock_result.usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_agent.run = AsyncMock(return_value=mock_result)
        agent._agent = mock_agent

        response = await agent.invoke("Hello!")

        assert response.success is True
        assert "model" in response.metadata
        assert "framework" in response.metadata
        assert response.metadata["framework"] == "pydantic_ai"


class TestPydanticAIAgentStream:
    """Test Pydantic AI agent streaming."""

    @pytest.mark.asyncio
    async def test_stream_success(self):
        """Test successful streaming."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        async def mock_stream():
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"

        mock_agent = MagicMock()
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_context)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_context.__aiter__ = MagicMock(return_value=mock_stream().__aiter__())
        mock_agent.run_stream = MagicMock(return_value=mock_context)
        agent._agent = mock_agent

        chunks = []
        async for chunk in agent.stream("Hello!"):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0] == "chunk1"
        assert chunks[1] == "chunk2"
        assert chunks[2] == "chunk3"

    @pytest.mark.asyncio
    async def test_stream_with_context(self):
        """Test streaming with context."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        async def mock_stream():
            yield "chunk1"

        mock_agent = MagicMock()
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_context)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_context.__aiter__ = MagicMock(return_value=mock_stream().__aiter__())
        mock_agent.run_stream = MagicMock(return_value=mock_context)
        agent._agent = mock_agent

        context = {"user_id": "123"}
        chunks = []
        async for chunk in agent.stream("Hello!", context=context):
            chunks.append(chunk)

        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_stream_not_initialized_raises_error(self):
        """Test that streaming uninitialized agent raises error."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)

        with pytest.raises(RuntimeError, match="Agent not initialized"):
            async for _ in agent.stream("Hello!"):
                pass

    @pytest.mark.asyncio
    async def test_stream_error_handling(self):
        """Test error handling during streaming."""
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

        with pytest.raises(Exception, match="Stream error"):
            async for _ in agent.stream("Hello!"):
                pass


class TestPydanticAIAgentCleanup:
    """Test Pydantic AI agent cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_success(self):
        """Test successful cleanup."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._model = MagicMock()

        await agent.cleanup()

        assert agent._initialized is False
        assert agent._agent is None
        assert agent._model is None

    @pytest.mark.asyncio
    async def test_cleanup_when_not_initialized(self):
        """Test cleanup when agent not initialized."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)

        # Should not raise error
        await agent.cleanup()

        assert agent._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_error_handling(self):
        """Test cleanup with error handling."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True
        agent._agent = MagicMock()

        # Should not raise error even if cleanup fails
        await agent.cleanup()

        assert agent._initialized is False


class TestPydanticAIAgentContextManager:
    """Test Pydantic AI agent as context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_success(self):
        """Test using agent as context manager."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        with patch("pydantic_ai.Agent"):
            async with PydanticAIAgent(config) as agent:
                assert agent.is_initialized is True

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_exit(self):
        """Test that cleanup is called on context exit."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        with patch("pydantic_ai.Agent"):
            agent = PydanticAIAgent(config)
            async with agent:
                assert agent.is_initialized is True

            assert agent.is_initialized is False

    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self):
        """Test that cleanup is called even if exception occurs."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        with patch("pydantic_ai.Agent"):
            agent = PydanticAIAgent(config)

            with pytest.raises(ValueError):
                async with agent:
                    assert agent.is_initialized is True
                    raise ValueError("Test error")

            assert agent.is_initialized is False


class TestPydanticAIAgentIntegration:
    """Integration tests for Pydantic AI agent."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete agent workflow."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
            system_prompt="You are helpful",
        )

        with patch("pydantic_ai.Agent") as mock_agent_class:
            mock_agent_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.data = "Test response"
            mock_result.usage = None
            mock_agent_instance.run = AsyncMock(return_value=mock_result)
            mock_agent_class.return_value = mock_agent_instance

            async with PydanticAIAgent(config) as agent:
                response = await agent.invoke("Hello!")

                assert response.success is True
                assert response.content == "Test response"

    @pytest.mark.asyncio
    async def test_multiple_invocations(self):
        """Test multiple invocations with same agent."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        mock_agent = MagicMock()
        responses = [
            MagicMock(data="Response 1", usage=None),
            MagicMock(data="Response 2", usage=None),
            MagicMock(data="Response 3", usage=None),
        ]
        mock_agent.run = AsyncMock(side_effect=responses)
        agent._agent = mock_agent

        response1 = await agent.invoke("Query 1")
        response2 = await agent.invoke("Query 2")
        response3 = await agent.invoke("Query 3")

        assert response1.content == "Response 1"
        assert response2.content == "Response 2"
        assert response3.content == "Response 3"
        assert mock_agent.run.call_count == 3

    @pytest.mark.asyncio
    async def test_agent_repr(self):
        """Test agent string representation."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        repr_str = repr(agent)

        assert "PydanticAIAgent" in repr_str
        assert "test_agent" in repr_str
        assert "pydantic_ai" in repr_str
