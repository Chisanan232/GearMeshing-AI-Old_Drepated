"""Unit tests for AI agent abstraction base classes."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from gearmeshing_ai.agent_core.abstraction.base import (
    AIAgentBase,
    AIAgentConfig,
    AIAgentResponse,
)


class MockAgent(AIAgentBase):
    """Mock agent for testing."""

    def build_init_kwargs(self):
        """Build initialization kwargs."""
        return {"model": self._config.model}

    async def initialize(self) -> None:
        self._initialized = True

    async def invoke(self, input_text: str, context=None, **kwargs):
        return AIAgentResponse(
            content=f"Response to: {input_text}",
            success=True,
        )

    async def stream(self, input_text: str, context=None, **kwargs):
        yield f"Chunk 1: {input_text}"
        yield "Chunk 2"

    async def cleanup(self) -> None:
        self._initialized = False


class TestAIAgentConfig:
    """Test AIAgentConfig."""

    def test_config_creation(self):
        """Test creating a config."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
            system_prompt="You are helpful",
            temperature=0.7,
        )

        assert config.name == "test_agent"
        assert config.framework == "pydantic_ai"
        assert config.model == "gpt-4o"
        assert config.system_prompt == "You are helpful"
        assert config.temperature == 0.7

    def test_config_to_dict(self):
        """Test converting config to dict."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )

        config_dict = config.to_dict()
        assert config_dict["name"] == "test_agent"
        assert config_dict["framework"] == "pydantic_ai"
        assert config_dict["model"] == "gpt-4o"

    def test_config_defaults(self):
        """Test config default values."""
        config = AIAgentConfig(
            name="test",
            framework="test",
            model="test",
        )

        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.timeout is None
        assert config.tools == []
        assert config.metadata == {}


class TestAIAgentResponse:
    """Test AIAgentResponse."""

    def test_response_creation(self):
        """Test creating a response."""
        response = AIAgentResponse(
            content="Hello",
            tool_calls=[{"name": "tool1", "args": {}}],
            success=True,
        )

        assert response.content == "Hello"
        assert len(response.tool_calls) == 1
        assert response.success is True
        assert response.error is None

    def test_response_to_dict(self):
        """Test converting response to dict."""
        response = AIAgentResponse(
            content="Hello",
            success=True,
        )

        response_dict = response.to_dict()
        assert response_dict["content"] == "Hello"
        assert response_dict["success"] is True

    def test_error_response(self):
        """Test error response."""
        response = AIAgentResponse(
            content=None,
            error="Something went wrong",
            success=False,
        )

        assert response.success is False
        assert response.error == "Something went wrong"


class TestAIAgentBase:
    """Test AIAgentBase."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        config = AIAgentConfig(
            name="test",
            framework="test",
            model="test",
        )
        agent = MockAgent(config)

        assert agent.config == config
        assert agent.is_initialized is False
        assert agent.framework == "test"
        assert agent.model == "test"

    @pytest.mark.asyncio
    async def test_agent_invoke(self):
        """Test agent invocation."""
        config = AIAgentConfig(
            name="test",
            framework="test",
            model="test",
        )
        agent = MockAgent(config)
        await agent.initialize()

        response = await agent.invoke("Hello")
        assert response.success is True
        assert "Response to: Hello" in response.content

    @pytest.mark.asyncio
    async def test_agent_stream(self):
        """Test agent streaming."""
        config = AIAgentConfig(
            name="test",
            framework="test",
            model="test",
        )
        agent = MockAgent(config)
        await agent.initialize()

        chunks = []
        async for chunk in agent.stream("Hello"):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert "Chunk 1" in chunks[0]
        assert "Chunk 2" in chunks[1]

    @pytest.mark.asyncio
    async def test_agent_context_manager(self):
        """Test agent as context manager."""
        config = AIAgentConfig(
            name="test",
            framework="test",
            model="test",
        )

        async with MockAgent(config) as agent:
            assert agent.is_initialized is True

    def test_agent_repr(self):
        """Test agent string representation."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4o",
        )
        agent = MockAgent(config)

        repr_str = repr(agent)
        assert "MockAgent" in repr_str
        assert "test_agent" in repr_str
        assert "pydantic_ai" in repr_str
