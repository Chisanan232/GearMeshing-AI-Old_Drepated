"""Tests for adapter response handling."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gearmeshing_ai.agent_core.abstraction.adapters.pydantic_ai import PydanticAIAgent
from gearmeshing_ai.agent_core.abstraction.base import AIAgentConfig


class TestAdapterResponseHandling:
    """Test response handling in adapters."""

    @pytest.mark.asyncio
    async def test_response_with_no_usage_data(self):
        """Test response when usage data is not available."""
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

        assert response.success is True
        # When usage is None, adapter still creates usage dict with None values
        usage = response.metadata.get("usage")
        assert usage is None or (isinstance(usage, dict) and all(v is None for v in usage.values()))

    @pytest.mark.asyncio
    async def test_response_with_complete_usage_data(self):
        """Test response with complete usage data."""
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

        assert response.success is True
        assert response.metadata["usage"]["input_tokens"] == 100
        assert response.metadata["usage"]["output_tokens"] == 50

    @pytest.mark.asyncio
    async def test_response_with_partial_usage_data(self):
        """Test response with partial usage data."""
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
            output_tokens=None,
        )
        mock_agent.run = AsyncMock(return_value=mock_result)
        agent._agent = mock_agent

        response = await agent.invoke("test")

        assert response.success is True
        assert response.metadata["usage"]["input_tokens"] == 100
        assert response.metadata["usage"]["output_tokens"] is None

    @pytest.mark.asyncio
    async def test_response_content_types(self):
        """Test response with different content types."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        # Test string content
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.data = "String response"
        mock_result.usage = None
        mock_agent.run = AsyncMock(return_value=mock_result)
        agent._agent = mock_agent

        response = await agent.invoke("test")
        assert isinstance(response.content, str)

    @pytest.mark.asyncio
    async def test_response_with_empty_content(self):
        """Test response with empty content."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.data = ""
        mock_result.usage = None
        mock_agent.run = AsyncMock(return_value=mock_result)
        agent._agent = mock_agent

        response = await agent.invoke("test")

        assert response.success is True
        assert response.content == ""

    @pytest.mark.asyncio
    async def test_response_with_no_tool_calls(self):
        """Test response when no tool calls are made."""
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
        mock_result.tool_calls = None
        mock_result.usage = None
        mock_agent.run = AsyncMock(return_value=mock_result)
        agent._agent = mock_agent

        response = await agent.invoke("test")

        assert response.success is True
        assert response.tool_calls == []

    @pytest.mark.asyncio
    async def test_response_with_multiple_tool_calls(self):
        """Test response with multiple tool calls."""
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
            {"name": "tool1", "args": {"arg1": "val1"}},
            {"name": "tool2", "args": {"arg2": "val2"}},
            {"name": "tool3", "args": {"arg3": "val3"}},
        ]
        mock_result.usage = None
        mock_agent.run = AsyncMock(return_value=mock_result)
        agent._agent = mock_agent

        response = await agent.invoke("test")

        assert response.success is True
        assert len(response.tool_calls) == 3
        assert response.tool_calls[0]["name"] == "tool1"
        assert response.tool_calls[1]["name"] == "tool2"
        assert response.tool_calls[2]["name"] == "tool3"

    @pytest.mark.asyncio
    async def test_error_response_structure(self):
        """Test error response structure."""
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
        assert response.error == "Invalid input"
        assert response.content is None
        assert response.tool_calls == []

    @pytest.mark.asyncio
    async def test_response_metadata_always_present(self):
        """Test that response metadata is always present."""
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

        assert "model" in response.metadata
        assert "framework" in response.metadata
        assert response.metadata["framework"] == "pydantic_ai"

    @pytest.mark.asyncio
    async def test_response_to_dict(self):
        """Test converting response to dictionary."""
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
        response_dict = response.to_dict()

        assert isinstance(response_dict, dict)
        assert "content" in response_dict
        assert "success" in response_dict
        assert "error" in response_dict
        assert "tool_calls" in response_dict
        assert "metadata" in response_dict


class TestAdapterStreamingResponse:
    """Test streaming response handling."""

    @pytest.mark.asyncio
    async def test_stream_response_chunks(self):
        """Test streaming response chunks."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        async def mock_stream():
            yield "Hello "
            yield "world "
            yield "from "
            yield "stream"

        mock_agent = MagicMock()
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_context)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_context.__aiter__ = MagicMock(return_value=mock_stream().__aiter__())
        mock_agent.run_stream = MagicMock(return_value=mock_context)
        agent._agent = mock_agent

        chunks = []
        async for chunk in agent.stream("test"):
            chunks.append(chunk)

        assert len(chunks) == 4
        assert "".join(chunks) == "Hello world from stream"

    @pytest.mark.asyncio
    async def test_stream_empty_response(self):
        """Test streaming with empty response."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        async def mock_stream():
            return
            yield  # Make it a generator

        mock_agent = MagicMock()
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_context)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_context.__aiter__ = MagicMock(return_value=mock_stream().__aiter__())
        mock_agent.run_stream = MagicMock(return_value=mock_context)
        agent._agent = mock_agent

        chunks = []
        async for chunk in agent.stream("test"):
            chunks.append(chunk)

        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_stream_large_chunks(self):
        """Test streaming with large chunks."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        large_chunk = "x" * 10000

        async def mock_stream():
            yield large_chunk
            yield large_chunk

        mock_agent = MagicMock()
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_context)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_context.__aiter__ = MagicMock(return_value=mock_stream().__aiter__())
        mock_agent.run_stream = MagicMock(return_value=mock_context)
        agent._agent = mock_agent

        chunks = []
        async for chunk in agent.stream("test"):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert len(chunks[0]) == 10000
        assert len(chunks[1]) == 10000

    @pytest.mark.asyncio
    async def test_stream_special_characters(self):
        """Test streaming with special characters."""
        config = AIAgentConfig(
            name="test",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True

        special_chunks = ["Hello\n", "World\t", "🎉", "特殊文字"]

        async def mock_stream():
            for chunk in special_chunks:
                yield chunk

        mock_agent = MagicMock()
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_context)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_context.__aiter__ = MagicMock(return_value=mock_stream().__aiter__())
        mock_agent.run_stream = MagicMock(return_value=mock_context)
        agent._agent = mock_agent

        chunks = []
        async for chunk in agent.stream("test"):
            chunks.append(chunk)

        assert len(chunks) == 4
        assert chunks[0] == "Hello\n"
        assert chunks[1] == "World\t"
        assert chunks[2] == "🎉"
        assert chunks[3] == "特殊文字"
