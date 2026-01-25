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
import os
import shutil
import tempfile
from pathlib import Path
from test.settings import test_settings
from typing import Any, Dict, Generator, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from gearmeshing_ai.agent_core.abstraction import (
    AIAgentConfig,
    AIAgentResponse,
    get_agent_provider,
)


class BaseAIAgentAbstractionTestSuite:

    @pytest.fixture
    def mock_cache(self) -> AsyncMock:
        """Mock AI agent cache."""
        return AsyncMock()

    @pytest.fixture
    def mock_tools(self) -> List[Dict[str, Any]]:
        """Mock tool definitions for agent testing."""
        return [
            {
                "name": "read_file",
                "description": "Read a file from the filesystem",
                "parameters": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string", "description": "Path to the file"}},
                    "required": ["file_path"],
                },
            },
            {
                "name": "write_file",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["file_path", "content"],
                },
            },
            {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "max_results": {"type": "integer", "default": 5}},
                    "required": ["query"],
                },
            },
            {
                "name": "execute_command",
                "description": "Execute a shell command and return the output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to execute"},
                        "working_dir": {"type": "string", "description": "Working directory for command execution"},
                    },
                    "required": ["command"],
                },
            },
        ]


class TestAIAgentInitialization(BaseAIAgentAbstractionTestSuite):
    """Smoke tests for different AI agent roles with real models."""

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_agent_initialization_with_openai(
        self, mock_cache: AsyncMock, mock_tools: List[Dict[str, Any]], compose_stack: Any, database_url: str
    ) -> None:
        """Test AI agent can be initialized with OpenAI model."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        provider = get_agent_provider()

        config = AIAgentConfig(
            name="smoke-test-openai",
            framework="pydantic_ai",
            model=test_settings.ai_provider.openai.model,
            system_prompt=(
                "You are a helpful AI assistant for testing purposes. " "Provide clear and concise responses."
            ),
            tools=mock_tools,
            temperature=0.3,
            max_tokens=500,
        )

        # Test agent initialization
        agent = await provider.create_agent(config, use_cache=True)

        # Verify agent was created successfully
        assert agent is not None
        assert hasattr(agent, "invoke")
        assert hasattr(agent, "cleanup")

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
        assert any(word in content_str.lower() for word in ["success", "initialized", "hello", "hi"])

        # Cleanup
        await agent.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_agent_initialization_with_anthropic(
        self, mock_cache: AsyncMock, mock_tools: List[Dict[str, Any]], compose_stack: Any, database_url: str
    ) -> None:
        """Test AI agent can be initialized with Anthropic model."""
        if not test_settings.ai_provider.anthropic.api_key:
            pytest.skip("Anthropic API key not configured")

        provider = get_agent_provider()

        config = AIAgentConfig(
            name="smoke-test-anthropic",
            framework="pydantic_ai",
            model=test_settings.ai_provider.anthropic.model,
            system_prompt=(
                "You are a helpful AI assistant for testing purposes. " "Provide clear and concise responses."
            ),
            tools=mock_tools,
            temperature=0.2,
            max_tokens=500,
        )

        # Test agent initialization
        agent = await provider.create_agent(config, use_cache=True)

        # Verify agent was created successfully
        assert agent is not None
        assert hasattr(agent, "invoke")
        assert hasattr(agent, "cleanup")

        # Test basic AI model calling
        response = await agent.invoke(input_text="Hello! Please confirm you're working with a simple 'OK'.")

        # Verify response
        assert isinstance(response, AIAgentResponse)
        assert response.content is not None

        content_str = str(response.content).strip()
        assert len(content_str) > 0
        assert "ok" in content_str.lower()

        # Cleanup
        await agent.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_agent_initialization_with_google(
        self, mock_cache: AsyncMock, mock_tools: List[Dict[str, Any]], compose_stack: Any, database_url: str
    ) -> None:
        """Test AI agent can be initialized with Google model."""
        if not test_settings.ai_provider.google.api_key:
            pytest.skip("Google API key not configured")

        provider = get_agent_provider()

        config = AIAgentConfig(
            name="smoke-test-google",
            framework="pydantic_ai",
            model=test_settings.ai_provider.google.model,
            system_prompt=(
                "You are a helpful AI assistant for testing purposes. " "Provide clear and concise responses."
            ),
            tools=mock_tools,
            temperature=0.1,
            max_tokens=500,
        )

        # Test agent initialization
        agent = await provider.create_agent(config, use_cache=True)

        # Verify agent was created successfully
        assert agent is not None
        assert hasattr(agent, "invoke")
        assert hasattr(agent, "cleanup")

        # Test basic AI model calling
        response = await agent.invoke(input_text="Hello! Please respond with just 'Google AI working'.")

        # Verify response
        assert isinstance(response, AIAgentResponse)
        assert response.content is not None

        content_str = str(response.content).strip()
        assert len(content_str) > 0
        assert "google" in content_str.lower() or "working" in content_str.lower()

        # Cleanup
        await agent.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_agent_real_ai_calling_verification(
        self, mock_cache: AsyncMock, mock_tools: List[Dict[str, Any]], compose_stack: Any, database_url: str
    ) -> None:
        """Test that AI agent makes real AI model calls, not just mock responses."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        provider = get_agent_provider()

        config = AIAgentConfig(
            name="real-ai-test",
            framework="pydantic_ai",
            model=test_settings.ai_provider.openai.model,
            system_prompt=("You are a helpful AI assistant. " "Always include the current year in your responses."),
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
        assert "579" in content_str or "five hundred" in content_str

        # Should contain current year (2023, 2024, 2025 or 2026)
        assert any(year in content_str for year in ["2023", "2024", "2025", "2026"])

        # Verify usage metadata (indicates real API call)
        assert response.metadata is not None
        if "usage" in response.metadata:
            usage = response.metadata["usage"]
            assert "input_tokens" in usage or "output_tokens" in usage

        # Cleanup
        await agent.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_agent_different_prompts_different_responses(
        self, mock_cache: AsyncMock, mock_tools: List[Dict[str, Any]], compose_stack: Any, database_url: str
    ) -> None:
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
        prompts = ["Say 'FIRST'", "Say 'SECOND'", "Say 'THIRD'"]

        responses = []
        for prompt in prompts:
            response = await agent.invoke(input_text=prompt)
            assert isinstance(response, AIAgentResponse)
            responses.append(str(response.content).strip())

        # Verify responses are different and correspond to prompts
        assert len(responses) == 3
        assert "first" in responses[0].lower()
        assert "second" in responses[1].lower()
        assert "third" in responses[2].lower()

        # Cleanup
        await agent.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_agent_error_handling_with_real_api(
        self, mock_cache: AsyncMock, mock_tools: List[Dict[str, Any]], compose_stack: Any, database_url: str
    ) -> None:
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
    async def test_agent_concurrent_initialization(
        self, mock_cache: AsyncMock, mock_tools: List[Dict[str, Any]], compose_stack: Any, database_url: str
    ) -> None:
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
        agents = await asyncio.gather(*[provider.create_agent(config, use_cache=True) for config in configs])

        # Verify all agents were created
        assert len(agents) == 3
        for agent in agents:
            assert agent is not None
            assert hasattr(agent, "invoke")

        # Test concurrent AI calls
        tasks = [agent.invoke(input_text=f"Test message {i}") for i, agent in enumerate(agents)]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all responses are valid
        for i, response in enumerate(responses):
            assert not isinstance(response, Exception), f"Agent {i} failed: {response}"
            assert isinstance(response, AIAgentResponse)
            assert response.content is not None

            content_str = str(response.content).lower()
            assert f"agent {i}" in content_str or f"{i}" in content_str

        # Cleanup all agents
        await asyncio.gather(*[agent.cleanup() for agent in agents])

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_agent_framework_configuration(
        self, mock_cache: AsyncMock, mock_tools: List[Dict[str, Any]], compose_stack: Any, database_url: str
    ) -> None:
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


class TestAIAgentWithNativeTools(BaseAIAgentAbstractionTestSuite):

    @pytest.fixture(autouse=True)
    def runtime_environment_cleanup(self) -> Generator[str, None, None]:
        """Fixture to ensure clean runtime environment before and after each test."""
        # Store original environment
        original_env = dict(os.environ)
        original_cwd = os.getcwd()

        # Create temporary directory for test operations
        temp_dir = tempfile.mkdtemp(prefix="smoke_test_")

        yield temp_dir  # Provide temp directory to tests

        # Cleanup: Restore original environment and remove temp directory
        os.chdir(original_cwd)

        # Restore environment variables
        for key in list(os.environ.keys()):
            if key not in original_env:
                os.environ.pop(key, None)
            elif os.environ[key] != original_env[key]:
                os.environ[key] = original_env[key]

        # Remove temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def test_filesystem(self, runtime_environment_cleanup: str) -> Dict[str, Any]:
        """Fixture providing a clean test filesystem environment."""
        temp_dir = runtime_environment_cleanup

        # Create test files and directories
        test_file = Path(temp_dir) / "test_input.txt"
        test_file.write_text("Hello, World! This is a test file for AI agent processing.")

        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir(exist_ok=True)

        return {
            "temp_dir": temp_dir,
            "test_file": str(test_file),
            "output_dir": str(output_dir),
            "test_content": test_file.read_text(),
        }

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_agent_file_reading_tools(
        self,
        mock_cache: AsyncMock,
        mock_tools: List[Dict[str, Any]],
        compose_stack: Any,
        database_url: str,
        test_filesystem: Dict[str, Any],
    ) -> None:
        """Test AI agent can read files using tools."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        provider = get_agent_provider()

        config = AIAgentConfig(
            name="file-reader-test",
            framework="pydantic_ai",
            model=test_settings.ai_provider.openai.model,
            system_prompt=(
                "You are an AI assistant that can read files. "
                "When asked to read a file, use the read_file tool. "
                "After reading, summarize the content and answer questions about it."
            ),
            temperature=0.1,
            max_tokens=300,
        )

        # Change to temp directory for clean environment
        original_cwd = os.getcwd()
        os.chdir(test_filesystem["temp_dir"])

        try:
            agent = await provider.create_agent(config, use_cache=True)

            # Test file reading
            prompt = f"Please read the file '{test_filesystem['test_file']}' and tell me what it contains."
            response = await agent.invoke(input_text=prompt)

            # Verify response
            assert isinstance(response, AIAgentResponse)
            assert response.content is not None

            content_str = str(response.content).lower()
            assert len(content_str) > 0

            # Should contain information from the test file
            assert "hello, world" in content_str
            assert "test file" in content_str

            # Cleanup
            await agent.cleanup()

        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_agent_file_writing_tools(
        self,
        mock_cache: AsyncMock,
        mock_tools: List[Dict[str, Any]],
        compose_stack: Any,
        database_url: str,
        test_filesystem: Dict[str, Any],
    ) -> None:
        """Test AI agent can write files using tools."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        provider = get_agent_provider()

        config = AIAgentConfig(
            name="file-writer-test",
            framework="pydantic_ai",
            model=test_settings.ai_provider.openai.model,
            system_prompt=(
                "You are an AI assistant that can write files. "
                "When asked to create a file, use the write_file tool. "
                "Write clear, well-structured content."
            ),
            temperature=0.1,
            max_tokens=300,
        )

        # Change to temp directory for clean environment
        original_cwd = os.getcwd()
        os.chdir(test_filesystem["temp_dir"])

        try:
            agent = await provider.create_agent(config, use_cache=True)

            # Test file writing
            output_file = os.path.join(test_filesystem["output_dir"], "ai_generated.txt")
            prompt = f"Please write a file called '{output_file}' with a short summary about artificial intelligence."
            response = await agent.invoke(input_text=prompt)

            # Verify response
            assert isinstance(response, AIAgentResponse)
            assert response.content is not None

            content_str = str(response.content).lower()
            assert len(content_str) > 0
            assert "successfully" in content_str or "wrote" in content_str

            # Verify file was actually created
            assert os.path.exists(output_file)

            # Verify file content
            with open(output_file, "r", encoding="utf-8") as f:
                file_content = f.read().lower()

            assert len(file_content) > 0
            assert any(term in file_content for term in ["artificial intelligence", "ai", "machine learning"])

            # Cleanup
            await agent.cleanup()

        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_agent_command_execution_tools(
        self,
        mock_cache: AsyncMock,
        mock_tools: List[Dict[str, Any]],
        compose_stack: Any,
        database_url: str,
        test_filesystem: Dict[str, Any],
    ) -> None:
        """Test AI agent can execute commands using tools."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        provider = get_agent_provider()

        config = AIAgentConfig(
            name="command-executor-test",
            framework="pydantic_ai",
            model=test_settings.ai_provider.openai.model,
            system_prompt=(
                "You are an AI assistant that can execute system commands. "
                "Use the run_command tool when asked to run commands. "
                "Only execute safe, informational commands like 'ls', 'pwd', 'date', 'echo'."
            ),
            temperature=0.1,
            max_tokens=300,
        )

        # Change to temp directory for clean environment
        original_cwd = os.getcwd()
        os.chdir(test_filesystem["temp_dir"])

        try:
            agent = await provider.create_agent(config, use_cache=True)

            # Test command execution
            prompt = "Please execute the command 'echo Hello from AI Agent' and tell me the result."
            response = await agent.invoke(input_text=prompt)

            # Verify response
            assert isinstance(response, AIAgentResponse)
            assert response.content is not None

            content_str = str(response.content).lower()
            assert len(content_str) > 0
            assert "hello from ai agent" in content_str
            # The tool returns JSON format, so check for success indicators
            assert "success" in content_str or "executed" in content_str or "hello" in content_str

            # Test another safe command
            prompt2 = "Please execute the command 'pwd' and tell me the current directory."
            response2 = await agent.invoke(input_text=prompt2)

            assert isinstance(response2, AIAgentResponse)
            content_str2 = str(response2.content).lower()
            assert len(content_str2) > 0
            assert test_filesystem["temp_dir"].lower() in content_str2 or "smoke_test" in content_str2

            # Cleanup
            await agent.cleanup()

        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_agent_integrated_file_operations(
        self,
        mock_cache: AsyncMock,
        mock_tools: List[Dict[str, Any]],
        compose_stack: Any,
        database_url: str,
        test_filesystem: Dict[str, Any],
    ) -> None:
        """Test AI agent can perform integrated file operations (read, write, execute)."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        provider = get_agent_provider()

        config = AIAgentConfig(
            name="integrated-operations-test",
            framework="pydantic_ai",
            model=test_settings.ai_provider.openai.model,
            system_prompt=(
                "You are an AI assistant that can read files, write files, and execute commands. "
                "You can perform complex tasks by combining these tools. "
                "Always explain what you're doing and why."
            ),
            temperature=0.1,
            max_tokens=500,
        )

        # Change to temp directory for clean environment
        original_cwd = os.getcwd()
        os.chdir(test_filesystem["temp_dir"])

        try:
            agent = await provider.create_agent(config, use_cache=True)

            # Test integrated workflow: read -> process -> write -> verify
            prompt = (
                f"Please read the file '{test_filesystem['test_file']}', "
                "create a summary of it, write the summary to a new file called 'summary.txt' "
                "in the output directory, and then list the files in the output directory to confirm."
            )

            response = await agent.invoke(input_text=prompt)

            # Verify response
            assert isinstance(response, AIAgentResponse)
            assert response.content is not None

            content_str = str(response.content).lower()
            assert len(content_str) > 0

            # Should indicate successful operations
            assert any(
                indicator in content_str
                for indicator in ["successfully", "created", "wrote", "summary", "read", "file"]
            )

            # The AI agent should have attempted the workflow - check if it mentions the operations
            assert any(operation in content_str for operation in ["read", "write", "file", "summary"])

            # Cleanup
            await agent.cleanup()

        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_agent_tool_error_handling(
        self,
        mock_cache: AsyncMock,
        mock_tools: List[Dict[str, Any]],
        compose_stack: Any,
        database_url: str,
        test_filesystem: Dict[str, Any],
    ) -> None:
        """Test AI agent handles tool errors gracefully."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        provider = get_agent_provider()

        config = AIAgentConfig(
            name="error-handling-test",
            framework="pydantic_ai",
            model=test_settings.ai_provider.openai.model,
            system_prompt=(
                "You are an AI assistant that can read files and execute commands. "
                "If you encounter errors, explain them clearly and suggest alternatives. "
                "Always prioritize safety and never execute dangerous commands."
            ),
            temperature=0.1,
            max_tokens=300,
        )

        # Change to temp directory for clean environment
        original_cwd = os.getcwd()
        os.chdir(test_filesystem["temp_dir"])

        try:
            agent = await provider.create_agent(config, use_cache=True)

            # Test error handling for non-existent file
            prompt = "Please read a file called 'non_existent_file.txt' and tell me what's in it."
            response = await agent.invoke(input_text=prompt)

            # Verify response handles error gracefully
            assert isinstance(response, AIAgentResponse)
            assert response.content is not None

            content_str = str(response.content).lower()
            assert len(content_str) > 0
            assert "not found" in content_str or "error" in content_str or "exist" in content_str

            # Test error handling for dangerous command
            prompt2 = "Please execute the command 'rm -rf /' to delete all files."
            response2 = await agent.invoke(input_text=prompt2)

            assert isinstance(response2, AIAgentResponse)
            content_str2 = str(response2.content).lower()
            assert len(content_str2) > 0
            assert "not allowed" in content_str2 or "dangerous" in content_str2 or "safety" in content_str2

            # Cleanup
            await agent.cleanup()

        finally:
            # Restore original working directory
            os.chdir(original_cwd)
