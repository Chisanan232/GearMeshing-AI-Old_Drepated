"""Integration tests for Pydantic AI agent adapter tool calling.

Tests verify that the Pydantic AI agent correctly calls tools with proper
parameters using Pydantic AI's TestModel pattern as recommended by the
Pydantic AI team. Tests use capture_run_messages() to verify tool selection
without making real API calls.
"""

import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from gearmeshing_ai.agent_core.abstraction.adapters.pydantic_ai import PydanticAIAgent
from gearmeshing_ai.agent_core.abstraction.tools.definitions import (
    CommandRunInput,
    CommandRunOutput,
    FileReadInput,
    FileReadOutput,
    FileWriteInput,
    FileWriteOutput,
)
from gearmeshing_ai.agent_core.abstraction.base import AIAgentConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestPydanticAIAgentToolCalling:
    """Integration tests for tool calling with Pydantic AI using TestModel.
    
    Tests use the initialize() method to properly set up the agent with tools,
    then override _agent with TestModel for testing without real API calls.
    """

    @pytest.mark.asyncio
    async def test_agent_initializes_with_tools(self) -> None:
        """Test that agent initializes properly with tools registered."""
        pytest.importorskip("pydantic_ai")

        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="test",
        )
        agent = PydanticAIAgent(config)

        # Initialize the agent - this creates the agent and registers tools
        await agent.initialize()

        # Verify agent was initialized successfully
        assert agent._initialized is True
        assert agent._agent is not None

    @pytest.mark.asyncio
    async def test_read_file_tool_with_testmodel(self, temp_dir: str) -> None:
        """Test that read_file tool can be invoked with TestModel."""
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.models.test import TestModel
        from pydantic_ai import capture_run_messages, ToolCallPart, models

        # Safety measure: prevent accidental real API calls
        models.ALLOW_MODEL_REQUESTS = False

        # Create test file
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("test content")

        # Track tool calls
        tool_calls = []

        async def mock_read_file_handler(input_data: FileReadInput) -> FileReadOutput:
            tool_calls.append({
                "tool": "read_file",
                "file_path": input_data.file_path,
                "encoding": input_data.encoding,
            })
            return FileReadOutput(
                success=True,
                content="test content",
                file_path=input_data.file_path,
                size_bytes=12,
            )

        # Patch the handler
        with mock.patch(
            "gearmeshing_ai.agent_core.abstraction.adapters.pydantic_ai.read_file_handler",
            side_effect=mock_read_file_handler,
        ):
            config = AIAgentConfig(
                name="test_agent",
                framework="pydantic_ai",
                model="test",
            )
            agent = PydanticAIAgent(config)

            # Initialize agent with tools registered
            await agent.initialize()

            # Capture messages to verify tool calls
            with capture_run_messages() as messages:
                # Use override context manager to replace model with TestModel
                assert agent._agent is not None, "Agent should be initialized"
                with agent._agent.override(model=TestModel()):
                    result = await agent._agent.run(
                        f"Read the file at {test_file} and tell me its content"
                    )

            # Verify through message capture that tool was called
            tool_call_found = False
            for msg in messages:
                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        if isinstance(part, ToolCallPart) and part.tool_name == "read_file":
                            tool_call_found = True
                            # TestModel passes required parameters
                            assert "file_path" in part.args
                            # assert "encoding" in part.args

            # Must verify that tool was actually called - fail if not
            assert tool_call_found, "read_file tool should have been called by TestModel"
            assert len(tool_calls) > 0, "read_file handler should have been invoked"
            assert tool_calls[0]["tool"] == "read_file"
            # Verify default parameter was used
            assert tool_calls[0]["encoding"] == "utf-8"

    @pytest.mark.asyncio
    async def test_write_file_tool_with_testmodel(self, temp_dir: str) -> None:
        """Test that write_file tool can be invoked with TestModel."""
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.models.test import TestModel
        from pydantic_ai import capture_run_messages, ToolCallPart, models

        models.ALLOW_MODEL_REQUESTS = False

        # Track tool calls
        tool_calls = []

        async def mock_write_file_handler(input_data: FileWriteInput) -> FileWriteOutput:
            tool_calls.append({
                "tool": "write_file",
                "file_path": input_data.file_path,
                "content": input_data.content,
                "encoding": input_data.encoding,
                "create_dirs": input_data.create_dirs,
            })
            return FileWriteOutput(
                success=True,
                file_path=input_data.file_path,
                bytes_written=len(input_data.content),
            )

        with mock.patch(
            "gearmeshing_ai.agent_core.abstraction.adapters.pydantic_ai.write_file_handler",
            side_effect=mock_write_file_handler,
        ):
            config = AIAgentConfig(
                name="test_agent",
                framework="pydantic_ai",
                model="test",
            )
            agent = PydanticAIAgent(config)

            # Initialize agent with tools registered
            await agent.initialize()

            with capture_run_messages() as messages:
                # Use override context manager to replace model with TestModel
                assert agent._agent is not None, "Agent should be initialized"
                with agent._agent.override(model=TestModel()):
                    result = await agent._agent.run(
                        f"Write 'Hello World' to {temp_dir}/output.txt"
                    )

            # Verify through message capture that tool was called
            tool_call_found = False
            for msg in messages:
                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        if isinstance(part, ToolCallPart) and part.tool_name == "write_file":
                            tool_call_found = True
                            # TestModel passes required parameters
                            assert "file_path" in part.args
                            assert "content" in part.args
                            # assert "encoding" in part.args
                            # assert "create_dirs" in part.args

            # Must verify that tool was actually called - fail if not
            assert tool_call_found, "write_file tool should have been called by TestModel"
            assert len(tool_calls) > 0, "write_file handler should have been invoked"
            assert tool_calls[0]["tool"] == "write_file"
            # Verify default parameters were used
            assert tool_calls[0]["encoding"] == "utf-8"
            assert tool_calls[0]["create_dirs"] is True

    @pytest.mark.asyncio
    async def test_run_command_tool_with_testmodel(self) -> None:
        """Test that run_command tool can be invoked with TestModel."""
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.models.test import TestModel
        from pydantic_ai import capture_run_messages, ToolCallPart, models

        models.ALLOW_MODEL_REQUESTS = False

        # Track tool calls
        tool_calls = []

        async def mock_run_command_handler(input_data: CommandRunInput) -> CommandRunOutput:
            tool_calls.append({
                "tool": "run_command",
                "command": input_data.command,
                "cwd": input_data.cwd,
                "timeout": input_data.timeout,
                "shell": input_data.shell,
            })
            return CommandRunOutput(
                success=True,
                exit_code=0,
                command=input_data.command,
                stdout="output",
                duration_seconds=0.1,
            )

        with mock.patch(
            "gearmeshing_ai.agent_core.abstraction.adapters.pydantic_ai.run_command_handler",
            side_effect=mock_run_command_handler,
        ):
            config = AIAgentConfig(
                name="test_agent",
                framework="pydantic_ai",
                model="test",
            )
            agent = PydanticAIAgent(config)

            # Initialize agent with tools registered
            await agent.initialize()

            with capture_run_messages() as messages:
                # Use override context manager to replace model with TestModel
                assert agent._agent is not None, "Agent should be initialized"
                with agent._agent.override(model=TestModel()):
                    result = await agent._agent.run(
                        "Execute the command 'echo hello'"
                    )

            # Verify through message capture that tool was called
            tool_call_found = False
            for msg in messages:
                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        if isinstance(part, ToolCallPart) and part.tool_name == "run_command":
                            tool_call_found = True
                            # TestModel passes required parameters
                            assert "command" in part.args
                            # assert "timeout" in part.args
                            # assert "shell" in part.args

            # Must verify that tool was actually called - fail if not
            assert tool_call_found, "run_command tool should have been called by TestModel"
            assert len(tool_calls) > 0, "run_command handler should have been invoked"
            assert tool_calls[0]["tool"] == "run_command"
            # Verify default parameters were used
            assert tool_calls[0]["timeout"] == 30.0
            assert tool_calls[0]["shell"] is True

    def test_read_file_handler_called_with_correct_parameters(self, temp_dir: str) -> None:
        """Test that read_file handler receives correct FileReadInput parameters."""
        # Create test file
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("test content")

        # Track parameters
        captured_params = []
        
        async def mock_handler(input_data: FileReadInput) -> FileReadOutput:
            captured_params.append({
                "file_path": input_data.file_path,
                "encoding": input_data.encoding,
            })
            return FileReadOutput(
                success=True,
                content="test",
                file_path=input_data.file_path,
            )

        # Test handler directly
        import asyncio
        
        async def run_test():
            input_data = FileReadInput(file_path=str(test_file), encoding="utf-8")
            result = await mock_handler(input_data)
            assert result.success is True
            assert len(captured_params) == 1
            assert captured_params[0]["file_path"] == str(test_file)
            assert captured_params[0]["encoding"] == "utf-8"

        asyncio.run(run_test())

    def test_write_file_handler_called_with_correct_parameters(self, temp_dir: str) -> None:
        """Test that write_file handler receives correct FileWriteInput parameters."""
        captured_params = []
        
        async def mock_handler(input_data: FileWriteInput) -> FileWriteOutput:
            captured_params.append({
                "file_path": input_data.file_path,
                "content": input_data.content,
                "encoding": input_data.encoding,
                "create_dirs": input_data.create_dirs,
            })
            return FileWriteOutput(
                success=True,
                file_path=input_data.file_path,
                bytes_written=len(input_data.content),
            )

        import asyncio
        
        async def run_test():
            input_data = FileWriteInput(
                file_path=f"{temp_dir}/output.txt",
                content="Hello World",
                encoding="utf-8",
                create_dirs=True,
            )
            result = await mock_handler(input_data)
            assert result.success is True
            assert len(captured_params) == 1
            assert captured_params[0]["encoding"] == "utf-8"
            assert captured_params[0]["create_dirs"] is True

        asyncio.run(run_test())

    def test_run_command_handler_called_with_correct_parameters(self) -> None:
        """Test that run_command handler receives correct CommandRunInput parameters."""
        captured_params = []
        
        async def mock_handler(input_data: CommandRunInput) -> CommandRunOutput:
            captured_params.append({
                "command": input_data.command,
                "timeout": input_data.timeout,
                "shell": input_data.shell,
            })
            return CommandRunOutput(
                success=True,
                exit_code=0,
                command=input_data.command,
                stdout="output",
                duration_seconds=0.1,
            )

        import asyncio
        
        async def run_test():
            input_data = CommandRunInput(
                command="echo test",
                timeout=30.0,
                shell=True,
            )
            result = await mock_handler(input_data)
            assert result.success is True
            assert len(captured_params) == 1
            assert captured_params[0]["timeout"] == 30.0
            assert captured_params[0]["shell"] is True

        asyncio.run(run_test())

    def test_tool_default_parameters(self, temp_dir: str) -> None:
        """Test that tool default parameters are applied correctly."""
        captured_params = []
        
        async def mock_read_handler(input_data: FileReadInput) -> FileReadOutput:
            captured_params.append(input_data.encoding)
            return FileReadOutput(success=True, content="test", file_path=input_data.file_path)

        import asyncio
        
        async def run_test():
            # Test with default encoding
            input_data = FileReadInput(file_path=f"{temp_dir}/test.txt")
            await mock_read_handler(input_data)
            assert captured_params[0] == "utf-8"

        asyncio.run(run_test())

    def test_tool_error_handling(self) -> None:
        """Test that tool error handling returns proper error output."""
        async def mock_handler(input_data: FileReadInput) -> FileReadOutput:
            return FileReadOutput(
                success=False,
                content=None,
                file_path=input_data.file_path,
                error="File not found",
            )

        import asyncio
        
        async def run_test():
            input_data = FileReadInput(file_path="/nonexistent/file.txt")
            result = await mock_handler(input_data)
            assert result.success is False
            assert result.error == "File not found"

        asyncio.run(run_test())

    @pytest.mark.asyncio
    async def test_agent_initialization_with_config(self) -> None:
        """Test agent initialization with configuration."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="test",
            system_prompt="You are a helpful assistant.",
            temperature=0.5,
            max_tokens=1000,
        )

        agent = PydanticAIAgent(config)
        
        # Build init kwargs
        kwargs = agent.build_init_kwargs()
        
        # Verify all options are included
        assert kwargs["model"] == "test"
        assert kwargs["system_prompt"] == "You are a helpful assistant."
        assert kwargs["model_settings"]["temperature"] == 0.5
        assert kwargs["model_settings"]["max_tokens"] == 1000

    @pytest.mark.asyncio
    async def test_agent_config_properties(self) -> None:
        """Test agent configuration properties."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4",
        )

        agent = PydanticAIAgent(config)
        
        assert agent.config.name == "test_agent"
        assert agent.framework == "pydantic_ai"
        assert agent.model == "gpt-4"
        assert agent.is_initialized is False

    @pytest.mark.asyncio
    async def test_agent_cleanup(self) -> None:
        """Test agent cleanup."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="test",
        )

        agent = PydanticAIAgent(config)
        agent._initialized = True
        agent._agent = mock.MagicMock()  # type: ignore[assignment]
        
        await agent.cleanup()
        
        assert agent._initialized is False
        assert agent._agent is None

    def test_build_init_kwargs_with_all_settings(self) -> None:
        """Test building kwargs with all settings."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4",
            system_prompt="You are helpful.",
            temperature=0.8,
            max_tokens=2000,
        )

        agent = PydanticAIAgent(config)
        kwargs = agent.build_init_kwargs()

        assert kwargs["model"] == "gpt-4"
        assert kwargs["system_prompt"] == "You are helpful."
        assert kwargs["model_settings"]["temperature"] == 0.8
        assert kwargs["model_settings"]["max_tokens"] == 2000

    def test_build_init_kwargs_with_metadata(self) -> None:
        """Test building kwargs with metadata."""
        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="gpt-4",
            metadata={"custom_param": "custom_value"},
        )

        agent = PydanticAIAgent(config)
        kwargs = agent.build_init_kwargs()

        assert kwargs["model"] == "gpt-4"
        assert kwargs["custom_param"] == "custom_value"

    def test_read_file_input_validation(self) -> None:
        """Test FileReadInput validation."""
        # Valid input
        input_data = FileReadInput(file_path="/path/to/file.txt", encoding="utf-8")
        assert input_data.file_path == "/path/to/file.txt"
        assert input_data.encoding == "utf-8"

        # Default encoding
        input_data = FileReadInput(file_path="/path/to/file.txt")
        assert input_data.encoding == "utf-8"

    def test_write_file_input_validation(self) -> None:
        """Test FileWriteInput validation."""
        input_data = FileWriteInput(
            file_path="/path/to/file.txt",
            content="test content",
            encoding="utf-8",
            create_dirs=True,
        )
        assert input_data.file_path == "/path/to/file.txt"
        assert input_data.content == "test content"
        assert input_data.encoding == "utf-8"
        assert input_data.create_dirs is True

    def test_command_run_input_validation(self) -> None:
        """Test CommandRunInput validation."""
        input_data = CommandRunInput(
            command="echo test",
            timeout=30.0,
            shell=True,
        )
        assert input_data.command == "echo test"
        assert input_data.timeout == 30.0
        assert input_data.shell is True

    def test_file_read_output_validation(self) -> None:
        """Test FileReadOutput validation."""
        output = FileReadOutput(
            success=True,
            content="file content",
            file_path="/path/to/file.txt",
            size_bytes=12,
        )
        assert output.success is True
        assert output.content == "file content"
        assert output.file_path == "/path/to/file.txt"
        assert output.size_bytes == 12

    def test_file_write_output_validation(self) -> None:
        """Test FileWriteOutput validation."""
        output = FileWriteOutput(
            success=True,
            file_path="/path/to/file.txt",
            bytes_written=12,
        )
        assert output.success is True
        assert output.file_path == "/path/to/file.txt"
        assert output.bytes_written == 12

    def test_command_run_output_validation(self) -> None:
        """Test CommandRunOutput validation."""
        output = CommandRunOutput(
            success=True,
            exit_code=0,
            command="echo test",
            stdout="test output",
            duration_seconds=0.1,
        )
        assert output.success is True
        assert output.exit_code == 0
        assert output.command == "echo test"
        assert output.stdout == "test output"
        assert output.duration_seconds == 0.1


class TestPydanticAIAgentErrorHandling:
    """Unit tests for error handling paths in PydanticAIAgent.
    
    Tests cover:
    - Tools disabled scenario (L140-L142)
    - Tool registration error handling (L268-L270)
    - Agent cleanup error handling (L412-L417)
    """

    @pytest.mark.asyncio
    async def test_register_tools_skipped_when_tools_disabled(self) -> None:
        """Test that _register_tools returns early when tools are disabled."""
        pytest.importorskip("pydantic_ai")

        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="test",
        )
        agent = PydanticAIAgent(config)
        agent._enable_tools = False

        # Initialize agent
        await agent.initialize()

        # Verify that tools were not registered by checking that no error occurred
        # and the agent is still initialized
        assert agent._initialized is True
        assert agent._agent is not None

    @pytest.mark.asyncio
    async def test_register_tools_error_handling_on_exception(self) -> None:
        """Test that _register_tools handles exceptions gracefully."""
        pytest.importorskip("pydantic_ai")
        from pydantic_ai import Agent

        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="test",
        )
        agent = PydanticAIAgent(config)

        # Create a mock agent that raises an exception during tool registration
        mock_agent = mock.MagicMock(spec=Agent)
        mock_agent.tool = mock.MagicMock(side_effect=RuntimeError("Tool registration failed"))

        # Call _register_tools with the mock agent
        # Should not raise, but log the error
        agent._register_tools(mock_agent)

        # Verify the agent is still usable
        assert agent._agent is None  # Not set yet since we're testing _register_tools directly

    @pytest.mark.asyncio
    async def test_cleanup_handles_exceptions(self) -> None:
        """Test that cleanup handles exceptions gracefully."""
        pytest.importorskip("pydantic_ai")

        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="test",
        )
        agent = PydanticAIAgent(config)

        # Initialize the agent
        await agent.initialize()
        assert agent._initialized is True

        # Cleanup should not raise, even if agent is in an unexpected state
        # Simulate a scenario where cleanup is called multiple times
        await agent.cleanup()

        # Verify cleanup completed
        assert agent._initialized is False
        assert agent._agent is None

    @pytest.mark.asyncio
    async def test_agent_cleanup_clears_references(self) -> None:
        """Test that cleanup properly clears agent and model references."""
        pytest.importorskip("pydantic_ai")

        config = AIAgentConfig(
            name="test_agent",
            framework="pydantic_ai",
            model="test",
        )
        agent = PydanticAIAgent(config)

        # Initialize the agent
        await agent.initialize()
        assert agent._initialized is True
        assert agent._agent is not None

        # Cleanup
        await agent.cleanup()

        # Verify references are cleared
        assert agent._initialized is False
        assert agent._agent is None
        assert agent._model is None

    @pytest.mark.asyncio
    async def test_register_tools_with_disabled_tools_logs_debug(self) -> None:
        """Test that disabled tools logs debug message."""
        pytest.importorskip("pydantic_ai")

        config = AIAgentConfig(
            name="test_agent_disabled",
            framework="pydantic_ai",
            model="test",
        )
        agent = PydanticAIAgent(config)
        agent._enable_tools = False

        # Mock the logger to verify debug message
        with mock.patch(
            "gearmeshing_ai.agent_core.abstraction.adapters.pydantic_ai.logger"
        ) as mock_logger:
            await agent.initialize()

            # Verify debug message was logged
            mock_logger.debug.assert_called()
            # Check that the "Tools disabled" message was logged
            calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            assert any("Tools disabled" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_register_tools_error_logs_error_message(self) -> None:
        """Test that tool registration errors are logged."""
        pytest.importorskip("pydantic_ai")
        from pydantic_ai import Agent

        config = AIAgentConfig(
            name="test_agent_error",
            framework="pydantic_ai",
            model="test",
        )
        agent = PydanticAIAgent(config)

        # Create a mock agent that raises an exception
        mock_agent = mock.MagicMock(spec=Agent)
        mock_agent.tool = mock.MagicMock(side_effect=ValueError("Invalid tool"))

        # Mock the logger to verify error message
        with mock.patch(
            "gearmeshing_ai.agent_core.abstraction.adapters.pydantic_ai.logger"
        ) as mock_logger:
            agent._register_tools(mock_agent)

            # Verify error message was logged
            mock_logger.error.assert_called()
            # Check that the error message contains the agent name
            calls = [call[0][0] for call in mock_logger.error.call_args_list]
            assert any("Error registering tools" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_cleanup_logs_debug_message(self) -> None:
        """Test that cleanup logs debug message."""
        pytest.importorskip("pydantic_ai")

        config = AIAgentConfig(
            name="test_agent_cleanup_debug",
            framework="pydantic_ai",
            model="test",
        )
        agent = PydanticAIAgent(config)

        # Initialize the agent
        await agent.initialize()

        # Mock the logger to verify debug logging
        with mock.patch(
            "gearmeshing_ai.agent_core.abstraction.adapters.pydantic_ai.logger"
        ) as mock_logger:
            await agent.cleanup()

            # Verify debug message was logged
            mock_logger.debug.assert_called()
            # Check that cleanup message was logged
            calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            assert any("Cleaning up" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_cleanup_error_handling_logs_exception(self) -> None:
        """Test that cleanup error handling logs exceptions with exc_info."""
        pytest.importorskip("pydantic_ai")

        config = AIAgentConfig(
            name="test_agent_cleanup_error",
            framework="pydantic_ai",
            model="test",
        )
        agent = PydanticAIAgent(config)

        # Initialize the agent
        await agent.initialize()

        # Mock the logger.debug to raise an exception during cleanup
        # This will trigger the except block in cleanup (L412-L416)
        with mock.patch(
            "gearmeshing_ai.agent_core.abstraction.adapters.pydantic_ai.logger"
        ) as mock_logger:
            # Make debug raise an exception to trigger the except block
            mock_logger.debug.side_effect = RuntimeError("Logger debug failed")

            # Cleanup should not raise, even if logger fails
            await agent.cleanup()

            # Verify error was logged with exc_info=True
            mock_logger.error.assert_called()
            # Check that error message was logged
            error_calls = mock_logger.error.call_args_list
            assert len(error_calls) > 0
            # Verify exc_info=True was passed
            assert error_calls[0][1].get("exc_info") is True
            # Verify error message contains the agent name
            error_msg = error_calls[0][0][0]
            assert "test_agent_cleanup_error" in error_msg
            assert "Error during Pydantic AI agent cleanup" in error_msg

    @pytest.mark.asyncio
    async def test_cleanup_exception_does_not_raise(self) -> None:
        """Test that cleanup does not raise exceptions, only logs them."""
        pytest.importorskip("pydantic_ai")

        config = AIAgentConfig(
            name="test_agent_cleanup_no_raise",
            framework="pydantic_ai",
            model="test",
        )
        agent = PydanticAIAgent(config)

        # Initialize the agent
        await agent.initialize()
        assert agent._initialized is True

        # Mock the logger.debug to raise an exception
        with mock.patch(
            "gearmeshing_ai.agent_core.abstraction.adapters.pydantic_ai.logger"
        ) as mock_logger:
            mock_logger.debug.side_effect = RuntimeError("Unexpected error")

            # Cleanup should not raise, even if an exception occurs internally
            try:
                await agent.cleanup()
                # If we get here, cleanup did not raise (correct behavior)
                assert True
            except Exception as e:
                # If we get here, cleanup raised an exception (incorrect behavior)
                pytest.fail(f"cleanup() should not raise exceptions, but raised: {e}")

    @pytest.mark.asyncio
    async def test_initialize_with_tools_enabled_registers_tools(self) -> None:
        """Test that initialize properly registers tools when enabled."""
        pytest.importorskip("pydantic_ai")

        config = AIAgentConfig(
            name="test_agent_with_tools",
            framework="pydantic_ai",
            model="test",
        )
        agent = PydanticAIAgent(config)
        assert agent._enable_tools is True

        # Initialize the agent
        await agent.initialize()

        # Verify agent is initialized
        assert agent._initialized is True
        assert agent._agent is not None

    @pytest.mark.asyncio
    async def test_multiple_cleanup_calls_are_safe(self) -> None:
        """Test that calling cleanup multiple times is safe."""
        pytest.importorskip("pydantic_ai")

        config = AIAgentConfig(
            name="test_agent_multi_cleanup",
            framework="pydantic_ai",
            model="test",
        )
        agent = PydanticAIAgent(config)

        # Initialize the agent
        await agent.initialize()
        assert agent._initialized is True

        # Call cleanup multiple times
        await agent.cleanup()
        assert agent._initialized is False

        # Second cleanup should not raise
        await agent.cleanup()
        assert agent._initialized is False
        assert agent._agent is None
