"""
Smoke tests for AI agent abstraction layer with different roles and real AI models.

These tests verify the core AI agent functionality using real AI model providers
while mocking all other dependencies (database, cache, etc.).

Key objectives:
1. Verify AI agent can be initialized correctly
2. Verify real AI model calling from providers like OpenAI, Anthropic, Google
3. Test different agent roles and configurations
4. Ensure proper cleanup and error handling
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Any, Dict, List

from test.settings import test_settings
from gearmeshing_ai.agent_core.abstraction import (
    AIAgentConfig,
    AIAgentResponse,
    get_agent_provider,
)


class TestAIAgentRolesSmoke:
    """Smoke tests for different AI agent roles with real models."""

    @pytest.fixture
    def mock_cache(self):
        """Mock AI agent cache."""
        return AsyncMock()

    @pytest.fixture
    def mock_tools(self):
        """Mock tool definitions for agent testing."""
        return [
            {
                'name': 'read_file',
                'description': 'Read a file from the filesystem',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'file_path': {'type': 'string', 'description': 'Path to the file'}
                    },
                    'required': ['file_path']
                }
            },
            {
                'name': 'write_file',
                'description': 'Write content to a file',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'file_path': {'type': 'string'},
                        'content': {'type': 'string'}
                    },
                    'required': ['file_path', 'content']
                }
            },
            {
                'name': 'web_search',
                'description': 'Search the web for information',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {'type': 'string'},
                        'max_results': {'type': 'integer', 'default': 5}
                    },
                    'required': ['query']
                }
            }
        ]

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_agent_initialization_with_openai(self, mock_cache, mock_tools, mock_settings_for_ai):
        """Test AI agent can be initialized with OpenAI model."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")
        
        provider = get_agent_provider()
        
        config = AIAgentConfig(
            name="smoke-test-openai",
            framework="pydantic_ai",
            model=test_settings.ai_provider.openai.model,
            system_prompt=(
                "You are a helpful AI assistant for testing purposes. "
                "Provide clear and concise responses."
            ),
            tools=mock_tools,
            temperature=0.3,
            max_tokens=500,
        )
        
        # Test agent initialization
        agent = await provider.create_agent(config, use_cache=True)
        
        # Verify agent was created successfully
        assert agent is not None
        assert hasattr(agent, 'invoke')
        assert hasattr(agent, 'cleanup')
        
        # Test basic AI model calling
        response = await agent.invoke(input_text="Hello! Please respond with 'AI agent initialized successfully'.")
        
        # Verify response structure and content
        assert isinstance(response, AIAgentResponse)
        assert response.content is not None
        assert response.metadata is not None
        
        # Content should be meaningful (not empty)
        content_str = str(response.content).strip()
        assert len(content_str) > 0
        
        # Should contain some indication of successful initialization
        assert any(word in content_str.lower() for word in ['success', 'initialized', 'hello', 'hi'])
        
        # Cleanup
        await agent.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_agent_initialization_with_anthropic(self, mock_cache, mock_tools, mock_settings_for_ai):
        """Test AI agent can be initialized with Anthropic model."""
        if not test_settings.ai_provider.anthropic.api_key:
            pytest.skip("Anthropic API key not configured")
        
        provider = get_agent_provider()
        
        config = AIAgentConfig(
            name="smoke-test-anthropic",
            framework="pydantic_ai",
            model=test_settings.ai_provider.anthropic.model,
            system_prompt=(
                "You are a helpful AI assistant for testing purposes. "
                "Provide clear and concise responses."
            ),
            tools=mock_tools,
            temperature=0.2,
            max_tokens=500,
        )
        
        # Test agent initialization
        agent = await provider.create_agent(config, use_cache=True)
        
        # Verify agent was created successfully
        assert agent is not None
        assert hasattr(agent, 'invoke')
        assert hasattr(agent, 'cleanup')
        
        # Test basic AI model calling
        response = await agent.invoke(input_text="Hello! Please confirm you're working with a simple 'OK'.")
        
        # Verify response
        assert isinstance(response, AIAgentResponse)
        assert response.content is not None
        
        content_str = str(response.content).strip()
        assert len(content_str) > 0
        assert 'ok' in content_str.lower()
        
        # Cleanup
        await agent.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_agent_initialization_with_google(self, mock_cache, mock_tools, mock_settings_for_ai):
        """Test AI agent can be initialized with Google model."""
        if not test_settings.ai_provider.google.api_key:
            pytest.skip("Google API key not configured")
        
        provider = get_agent_provider()
        
        config = AIAgentConfig(
            name="smoke-test-google",
            framework="pydantic_ai",
            model=test_settings.ai_provider.google.model,
            system_prompt=(
                "You are a helpful AI assistant for testing purposes. "
                "Provide clear and concise responses."
            ),
            tools=mock_tools,
            temperature=0.1,
            max_tokens=500,
        )
        
        # Test agent initialization
        agent = await provider.create_agent(config, use_cache=True)
        
        # Verify agent was created successfully
        assert agent is not None
        assert hasattr(agent, 'invoke')
        assert hasattr(agent, 'cleanup')
        
        # Test basic AI model calling
        response = await agent.invoke(input_text="Hello! Please respond with just 'Google AI working'.")
        
        # Verify response
        assert isinstance(response, AIAgentResponse)
        assert response.content is not None
        
        content_str = str(response.content).strip()
        assert len(content_str) > 0
        assert 'google' in content_str.lower() or 'working' in content_str.lower()
        
        # Cleanup
        await agent.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_agent_real_ai_calling_verification(self, mock_cache, mock_tools, mock_settings_for_ai):
        """Test that AI agent makes real AI model calls, not just mock responses."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")
        
        provider = get_agent_provider()
        
        config = AIAgentConfig(
            name="real-ai-test",
            framework="pydantic_ai",
            model=test_settings.ai_provider.openai.model,
            system_prompt=(
                "You are a helpful AI assistant. "
                "Always include the current year in your responses."
            ),
            tools=mock_tools,
            temperature=0.1,
            max_tokens=100,
        )
        
        agent = await provider.create_agent(config, use_cache=True)
        
        # Test with a unique prompt that requires real AI processing
        unique_prompt = f"What is 123 + 456? Please include the answer and the current year."
        response = await agent.invoke(input_text=unique_prompt)
        
        # Verify response contains real AI processing
        assert isinstance(response, AIAgentResponse)
        content_str = str(response.content).lower()
        
        # Should contain the correct calculation result (579)
        assert '579' in content_str or 'five hundred' in content_str
        
        # Should contain current year (2023, 2024, 2025 or 2026)
        assert any(year in content_str for year in ['2023', '2024', '2025', '2026'])
        
        # Verify usage metadata (indicates real API call)
        assert response.metadata is not None
        if 'usage' in response.metadata:
            usage = response.metadata['usage']
            assert 'input_tokens' in usage or 'output_tokens' in usage
        
        # Cleanup
        await agent.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_agent_different_prompts_different_responses(self, mock_cache, mock_tools, mock_settings_for_ai):
        """Test that different prompts produce different responses (not cached/static)."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")
        
        provider = get_agent_provider()
        
        config = AIAgentConfig(
            name="dynamic-test",
            framework="pydantic_ai",
            model=test_settings.ai_provider.openai.model,
            system_prompt="You are a helpful assistant. Be very brief.",
            tools=mock_tools,
            temperature=0.1,
            max_tokens=50,
        )
        
        agent = await provider.create_agent(config, use_cache=True)
        
        # Test with different prompts
        prompts = [
            "Say 'FIRST'",
            "Say 'SECOND'", 
            "Say 'THIRD'"
        ]
        
        responses = []
        for prompt in prompts:
            response = await agent.invoke(input_text=prompt)
            assert isinstance(response, AIAgentResponse)
            responses.append(str(response.content).strip())
        
        # Verify responses are different and correspond to prompts
        assert len(responses) == 3
        assert 'first' in responses[0].lower()
        assert 'second' in responses[1].lower()
        assert 'third' in responses[2].lower()
        
        # Cleanup
        await agent.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_agent_error_handling_with_real_api(self, mock_cache, mock_tools, mock_settings_for_ai):
        """Test agent error handling with real API calls."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")
        
        provider = get_agent_provider()
        
        config = AIAgentConfig(
            name="error-test",
            framework="pydantic_ai",
            model=test_settings.ai_provider.openai.model,
            system_prompt="You are a helpful assistant.",
            tools=mock_tools,
            temperature=0.1,
            max_tokens=100,
        )
        
        agent = await provider.create_agent(config, use_cache=True)
        
        # Test with empty prompt (should still work)
        response = await agent.invoke(input_text="")
        assert isinstance(response, AIAgentResponse)
        
        # Test with very long prompt (should handle gracefully)
        long_prompt = "Test " * 1000
        response = await agent.invoke(input_text=long_prompt)
        assert isinstance(response, AIAgentResponse)
        
        # Cleanup
        await agent.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_agent_concurrent_initialization(self, mock_cache, mock_tools, mock_settings_for_ai):
        """Test multiple agents can be initialized and used concurrently."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")
        
        provider = get_agent_provider()
        
        # Create multiple agent configurations
        configs = [
            AIAgentConfig(
                name=f"concurrent-agent-{i}",
                framework="pydantic_ai",
                model=test_settings.ai_provider.openai.model,
                system_prompt=f"You are agent {i}. Respond with 'Agent {i} working'.",
                tools=mock_tools,
                temperature=0.1,
                max_tokens=50,
            )
            for i in range(3)
        ]
        
        # Create agents concurrently
        agents = await asyncio.gather(*[
            provider.create_agent(config, use_cache=True)
            for config in configs
        ])
        
        # Verify all agents were created
        assert len(agents) == 3
        for agent in agents:
            assert agent is not None
            assert hasattr(agent, 'invoke')
        
        # Test concurrent AI calls
        tasks = [
            agent.invoke(input_text=f"Test message {i}")
            for i, agent in enumerate(agents)
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all responses are valid
        for i, response in enumerate(responses):
            assert not isinstance(response, Exception), f"Agent {i} failed: {response}"
            assert isinstance(response, AIAgentResponse)
            assert response.content is not None
            
            content_str = str(response.content).lower()
            assert f'agent {i}' in content_str or f'{i}' in content_str
        
        # Cleanup all agents
        await asyncio.gather(*[agent.cleanup() for agent in agents])

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_agent_framework_configuration(self, mock_cache, mock_tools, mock_settings_for_ai):
        """Test agent framework configuration works correctly."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")
        
        provider = get_agent_provider()
        
        # Verify framework is set
        assert provider._framework == "pydantic_ai"
        
        config = AIAgentConfig(
            name="framework-test",
            framework="pydantic_ai",
            model=test_settings.ai_provider.openai.model,
            system_prompt="You are a test agent.",
            tools=mock_tools,
            temperature=0.1,
        )
        
        agent = await provider.create_agent(config, use_cache=True)
        
        # Test that agent works with configured framework
        response = await agent.invoke(input_text="Say 'Framework test passed'")
        
        assert isinstance(response, AIAgentResponse)
        content_str = str(response.content).lower()
        
        # Just verify we got a response (content validation can be flexible)
        assert len(content_str) > 0
        
        await agent.cleanup()
