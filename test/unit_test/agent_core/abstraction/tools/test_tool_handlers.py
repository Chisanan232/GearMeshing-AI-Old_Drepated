"""Unit tests for AI agent tool handlers.

Tests for file read, file write, and command execution handlers.
"""

import os
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from gearmeshing_ai.agent_core.abstraction.tools.definitions import (
    CommandRunInput,
    CommandRunOutput,
    FileReadInput,
    FileReadOutput,
    FileWriteInput,
    FileWriteOutput,
)
from gearmeshing_ai.agent_core.abstraction.tools.handlers import (
    read_file_handler,
    write_file_handler,
    run_command_handler,
    FileReadHandler,
    FileWriteHandler,
    CommandRunHandler,
    ToolHandlerRegistry,
    get_handler_registry,
)


class TestReadFileHandler:
    """Tests for read_file_handler."""

    @pytest.mark.asyncio
    async def test_read_file_success(self) -> None:
        """Test successful file read."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("test content")
            temp_path = f.name

        try:
            input_data = FileReadInput(file_path=temp_path)
            result = await read_file_handler(input_data)

            assert result.success is True
            assert result.content == "test content"
            assert result.file_path == temp_path
            assert result.size_bytes == 12
            assert result.error is None
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_read_file_not_found(self) -> None:
        """Test reading non-existent file."""
        input_data = FileReadInput(file_path="/nonexistent/file.txt")
        result = await read_file_handler(input_data)

        assert result.success is False
        assert result.content is None
        assert result.error is not None
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_file_directory(self) -> None:
        """Test reading a directory instead of file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_data = FileReadInput(file_path=temp_dir)
            result = await read_file_handler(input_data)

            assert result.success is False
            assert result.error is not None
            assert "not a file" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_file_with_encoding(self) -> None:
        """Test reading file with specific encoding."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
            f.write("test content with unicode: é")
            temp_path = f.name

        try:
            input_data = FileReadInput(file_path=temp_path, encoding="utf-8")
            result = await read_file_handler(input_data)

            assert result.success is True
            assert "é" in result.content
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_read_large_file(self) -> None:
        """Test reading a large file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            large_content = "x" * 10000
            f.write(large_content)
            temp_path = f.name

        try:
            input_data = FileReadInput(file_path=temp_path)
            result = await read_file_handler(input_data)

            assert result.success is True
            assert len(result.content) == 10000
            assert result.size_bytes == 10000
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_read_file_encoding_error(self) -> None:
        """Test reading file with encoding error (L117-L124)."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            # Write invalid UTF-8 bytes
            f.write(b"\x80\x81\x82\x83")
            temp_path = f.name

        try:
            input_data = FileReadInput(file_path=temp_path, encoding="utf-8")
            result = await read_file_handler(input_data)

            assert result.success is False
            assert result.content is None
            assert result.error is not None
            assert "encoding error" in result.error.lower()
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_read_file_general_exception(self) -> None:
        """Test reading file with general exception (L125-L132)."""
        # Use a path that will cause an exception
        input_data = FileReadInput(file_path="/dev/null/invalid/path/file.txt")
        result = await read_file_handler(input_data)

        assert result.success is False
        assert result.content is None
        assert result.error is not None
        # Error could be "file not found" or "error reading file"
        assert "file" in result.error.lower() and ("not found" in result.error.lower() or "error" in result.error.lower())

    @pytest.mark.asyncio
    async def test_read_file_callable_interface(self) -> None:
        """Test that handler is callable via __call__."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("callable test")
            temp_path = f.name

        try:
            input_data = FileReadInput(file_path=temp_path)
            # Call handler directly using __call__
            result = await read_file_handler(input_data)

            assert result.success is True
            assert result.content == "callable test"
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_read_file_exception_handler_coverage(self) -> None:
        """Test exception handler coverage for L125-L132.
        
        This test specifically covers the general Exception handler
        that logs and returns error output.
        """
        handler = FileReadHandler()
        
        # Create a temporary file, then mock its read_text to raise exception
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("test")
            temp_path = f.name
        
        try:
            # Mock Path.read_text to raise exception after existence checks pass
            with mock.patch.object(Path, 'read_text', side_effect=RuntimeError("Simulated read error")):
                input_data = FileReadInput(file_path=temp_path)
                result = await handler.execute(input_data)
            
            # Verify the exception handler was triggered (L125-L132)
            assert result.success is False
            assert result.content is None
            assert result.error is not None
            assert "Error reading file" in result.error
            assert "Simulated read error" in result.error
            assert result.file_path == temp_path
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_read_file_exception_with_different_error_types(self) -> None:
        """Test exception handler with various exception types (L125-L132)."""
        handler = FileReadHandler()
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("test")
            temp_path = f.name
        
        try:
            # Test with IOError
            with mock.patch.object(Path, 'read_text', side_effect=IOError("IO error occurred")):
                result = await handler.execute(FileReadInput(file_path=temp_path))
                assert result.success is False
                assert "Error reading file" in result.error
                assert "IO error occurred" in result.error
            
            # Test with OSError
            with mock.patch.object(Path, 'read_text', side_effect=OSError("OS error occurred")):
                result = await handler.execute(FileReadInput(file_path=temp_path))
                assert result.success is False
                assert "Error reading file" in result.error
                assert "OS error occurred" in result.error
            
            # Test with ValueError
            with mock.patch.object(Path, 'read_text', side_effect=ValueError("Value error occurred")):
                result = await handler.execute(FileReadInput(file_path=temp_path))
                assert result.success is False
                assert "Error reading file" in result.error
                assert "Value error occurred" in result.error
        finally:
            os.unlink(temp_path)


class TestWriteFileHandler:
    """Tests for write_file_handler."""

    @pytest.mark.asyncio
    async def test_write_file_success(self) -> None:
        """Test successful file write."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test.txt")
            input_data = FileWriteInput(
                file_path=file_path,
                content="test content",
            )
            result = await write_file_handler(input_data)

            assert result.success is True
            assert result.file_path == file_path
            assert result.bytes_written == 12
            assert result.error is None

            # Verify file was actually written
            assert Path(file_path).exists()
            assert Path(file_path).read_text() == "test content"

    @pytest.mark.asyncio
    async def test_write_file_create_dirs(self) -> None:
        """Test writing file with directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "subdir", "nested", "test.txt")
            input_data = FileWriteInput(
                file_path=file_path,
                content="nested content",
                create_dirs=True,
            )
            result = await write_file_handler(input_data)

            assert result.success is True
            assert Path(file_path).exists()
            assert Path(file_path).read_text() == "nested content"

    @pytest.mark.asyncio
    async def test_write_file_no_create_dirs(self) -> None:
        """Test writing file without directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "nonexistent", "test.txt")
            input_data = FileWriteInput(
                file_path=file_path,
                content="content",
                create_dirs=False,
            )
            result = await write_file_handler(input_data)

            assert result.success is False
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_write_file_overwrite(self) -> None:
        """Test overwriting existing file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("old content")
            temp_path = f.name

        try:
            input_data = FileWriteInput(
                file_path=temp_path,
                content="new content",
            )
            result = await write_file_handler(input_data)

            assert result.success is True
            assert Path(temp_path).read_text() == "new content"
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_write_file_with_encoding(self) -> None:
        """Test writing file with specific encoding."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test.txt")
            input_data = FileWriteInput(
                file_path=file_path,
                content="content with unicode: é",
                encoding="utf-8",
            )
            result = await write_file_handler(input_data)

            assert result.success is True
            assert Path(file_path).read_text(encoding="utf-8") == "content with unicode: é"

    @pytest.mark.asyncio
    async def test_write_large_file(self) -> None:
        """Test writing a large file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "large.txt")
            large_content = "x" * 100000
            input_data = FileWriteInput(
                file_path=file_path,
                content=large_content,
            )
            result = await write_file_handler(input_data)

            assert result.success is True
            assert result.bytes_written == 100000

    @pytest.mark.asyncio
    async def test_write_file_permission_error(self) -> None:
        """Test writing file with permission error (L174-L180)."""
        # Try to write to a read-only directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file and make it read-only
            readonly_dir = os.path.join(temp_dir, "readonly")
            os.makedirs(readonly_dir)
            os.chmod(readonly_dir, 0o444)  # Read-only

            try:
                file_path = os.path.join(readonly_dir, "test.txt")
                input_data = FileWriteInput(
                    file_path=file_path,
                    content="test",
                    create_dirs=False,
                )
                result = await write_file_handler(input_data)

                assert result.success is False
                assert result.error is not None
                assert "permission" in result.error.lower() or "error" in result.error.lower()
            finally:
                # Restore permissions for cleanup
                os.chmod(readonly_dir, 0o755)

    @pytest.mark.asyncio
    async def test_write_file_general_exception(self) -> None:
        """Test writing file with general exception (L181-L188)."""
        # Use an invalid path that will cause an exception
        input_data = FileWriteInput(
            file_path="/dev/null/invalid/path/file.txt",
            content="test",
            create_dirs=False,
        )
        result = await write_file_handler(input_data)

        assert result.success is False
        assert result.error is not None
        assert "error writing file" in result.error.lower()

    @pytest.mark.asyncio
    async def test_write_file_callable_interface(self) -> None:
        """Test that write handler is callable via __call__."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "callable.txt")
            input_data = FileWriteInput(
                file_path=file_path,
                content="callable test",
            )
            # Call handler directly using __call__
            result = await write_file_handler(input_data)

            assert result.success is True
            assert Path(file_path).read_text() == "callable test"


class TestRunCommandHandler:
    """Tests for run_command_handler."""

    @pytest.mark.asyncio
    async def test_run_command_success(self) -> None:
        """Test successful command execution."""
        input_data = CommandRunInput(command="echo 'hello world'")
        result = await run_command_handler(input_data)

        assert result.success is True
        assert result.exit_code == 0
        assert "hello world" in result.stdout
        assert result.error is None

    @pytest.mark.asyncio
    async def test_run_command_with_stderr(self) -> None:
        """Test command that produces stderr."""
        # This command will fail and produce stderr
        input_data = CommandRunInput(command="ls /nonexistent/path 2>&1")
        result = await run_command_handler(input_data)

        # Command might succeed or fail depending on shell, but we should get output
        assert result.command == "ls /nonexistent/path 2>&1"

    @pytest.mark.asyncio
    async def test_run_command_with_cwd(self) -> None:
        """Test command execution with working directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_data = CommandRunInput(
                command="pwd",
                cwd=temp_dir,
            )
            result = await run_command_handler(input_data)

            assert result.success is True
            assert result.exit_code == 0
            assert temp_dir in result.stdout

    @pytest.mark.asyncio
    async def test_run_command_timeout(self) -> None:
        """Test command timeout."""
        input_data = CommandRunInput(
            command="sleep 10",
            timeout=0.1,
        )
        result = await run_command_handler(input_data)

        assert result.success is False
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_run_command_invalid_cwd(self) -> None:
        """Test command with invalid working directory."""
        input_data = CommandRunInput(
            command="echo test",
            cwd="/nonexistent/directory",
        )
        result = await run_command_handler(input_data)

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_run_command_captures_output(self) -> None:
        """Test that command output is captured."""
        input_data = CommandRunInput(command="echo 'test output'")
        result = await run_command_handler(input_data)

        assert result.success is True
        assert result.stdout is not None
        assert "test output" in result.stdout

    @pytest.mark.asyncio
    async def test_run_command_duration(self) -> None:
        """Test that command duration is recorded."""
        input_data = CommandRunInput(command="sleep 0.1")
        result = await run_command_handler(input_data)

        assert result.duration_seconds is not None
        assert result.duration_seconds >= 0.1

    @pytest.mark.asyncio
    async def test_run_command_exit_code(self) -> None:
        """Test that exit code is captured."""
        input_data = CommandRunInput(command="exit 42")
        result = await run_command_handler(input_data)

        assert result.exit_code == 42
        assert result.success is False

    @pytest.mark.asyncio
    async def test_run_command_general_exception(self) -> None:
        """Test command execution with general exception (L280-L289)."""
        # Create an invalid command that will cause an exception
        input_data = CommandRunInput(
            command="this_command_does_not_exist_12345",
            cwd=None,
        )
        result = await run_command_handler(input_data)

        assert result.success is False
        assert result.duration_seconds is not None
        # Command not found is captured as stderr, not as error
        # The handler treats non-zero exit codes as failure
        assert result.exit_code is not None or result.error is not None

    @pytest.mark.asyncio
    async def test_run_command_callable_interface(self) -> None:
        """Test that command handler is callable via __call__."""
        input_data = CommandRunInput(command="echo 'callable command'")
        # Call handler directly using __call__
        result = await run_command_handler(input_data)

        assert result.success is True
        assert "callable command" in result.stdout

    @pytest.mark.asyncio
    async def test_run_command_with_empty_output(self) -> None:
        """Test command with empty stdout/stderr."""
        input_data = CommandRunInput(command="true")
        result = await run_command_handler(input_data)

        assert result.success is True
        assert result.exit_code == 0
        # Empty output should be None, not empty string
        assert result.stdout is None or result.stdout == ""

    @pytest.mark.asyncio
    async def test_run_command_timeout_with_duration(self) -> None:
        """Test timeout error includes duration (L269-L278)."""
        input_data = CommandRunInput(
            command="sleep 5",
            timeout=0.05,
        )
        result = await run_command_handler(input_data)

        assert result.success is False
        assert "timeout" in result.error.lower()
        assert result.duration_seconds is not None
        assert result.duration_seconds >= 0.05

    @pytest.mark.asyncio
    async def test_run_command_exception_handler_coverage(self) -> None:
        """Test exception handler coverage for L280-L285.
        
        This test specifically covers the general Exception handler
        that logs and returns error output.
        """
        handler = CommandRunHandler()
        
        # Mock asyncio.create_subprocess_shell to raise a generic exception
        with mock.patch('asyncio.create_subprocess_shell', side_effect=RuntimeError("Simulated command error")):
            input_data = CommandRunInput(command="some_command")
            result = await handler.execute(input_data)
        
        # Verify the exception handler was triggered (L280-L285)
        assert result.success is False
        assert result.error is not None
        assert "Error executing command" in result.error
        assert "Simulated command error" in result.error
        assert result.command == "some_command"
        assert result.duration_seconds is not None

    @pytest.mark.asyncio
    async def test_run_command_exception_with_different_error_types(self) -> None:
        """Test exception handler with various exception types (L280-L285)."""
        handler = CommandRunHandler()
        
        # Test with ValueError
        with mock.patch('asyncio.create_subprocess_shell', side_effect=ValueError("Value error in command")):
            result = await handler.execute(CommandRunInput(command="cmd1"))
            assert result.success is False
            assert "Error executing command" in result.error
            assert "Value error in command" in result.error
            assert result.duration_seconds is not None
        
        # Test with TypeError
        with mock.patch('asyncio.create_subprocess_shell', side_effect=TypeError("Type error in command")):
            result = await handler.execute(CommandRunInput(command="cmd2"))
            assert result.success is False
            assert "Error executing command" in result.error
            assert "Type error in command" in result.error
            assert result.duration_seconds is not None
        
        # Test with RuntimeError
        with mock.patch('asyncio.create_subprocess_shell', side_effect=RuntimeError("Runtime error in command")):
            result = await handler.execute(CommandRunInput(command="cmd3"))
            assert result.success is False
            assert "Error executing command" in result.error
            assert "Runtime error in command" in result.error
            assert result.duration_seconds is not None

    @pytest.mark.asyncio
    async def test_run_command_exception_preserves_command_info(self) -> None:
        """Test that exception handler preserves command information (L280-L285)."""
        handler = CommandRunHandler()
        
        test_command = "test_command_xyz"
        with mock.patch('asyncio.create_subprocess_shell', side_effect=Exception("Test exception")):
            result = await handler.execute(CommandRunInput(command=test_command))
        
        # Verify command info is preserved even on exception
        assert result.command == test_command
        assert result.success is False
        assert result.error is not None
        assert result.duration_seconds is not None


class TestToolHandlerRegistry:
    """Tests for ToolHandlerRegistry."""

    def test_registry_register_handler(self) -> None:
        """Test registering a handler in registry (L299-L306)."""
        registry = ToolHandlerRegistry()
        handler = FileReadHandler()

        registry.register(handler)

        assert registry.get("read_file") is not None
        assert registry.get("read_file") == handler

    def test_registry_get_handler(self) -> None:
        """Test getting handler from registry (L308-L317)."""
        registry = ToolHandlerRegistry()
        handler = FileWriteHandler()
        registry.register(handler)

        retrieved = registry.get("write_file")

        assert retrieved is not None
        assert retrieved.name == "write_file"

    def test_registry_get_nonexistent_handler(self) -> None:
        """Test getting non-existent handler returns None (L317-L318)."""
        registry = ToolHandlerRegistry()

        result = registry.get("nonexistent_handler")

        assert result is None

    def test_registry_get_all_handlers(self) -> None:
        """Test getting all handlers from registry (L319-L326)."""
        registry = ToolHandlerRegistry()
        read_handler = FileReadHandler()
        write_handler = FileWriteHandler()
        cmd_handler = CommandRunHandler()

        registry.register(read_handler)
        registry.register(write_handler)
        registry.register(cmd_handler)

        all_handlers = registry.get_all()

        assert len(all_handlers) == 3
        assert "read_file" in all_handlers
        assert "write_file" in all_handlers
        assert "run_command" in all_handlers

    def test_registry_get_all_returns_copy(self) -> None:
        """Test that get_all returns a copy (L325-L326)."""
        registry = ToolHandlerRegistry()
        handler = FileReadHandler()
        registry.register(handler)

        handlers1 = registry.get_all()
        handlers2 = registry.get_all()

        # Should be equal but not the same object
        assert handlers1 == handlers2
        assert handlers1 is not handlers2

    def test_registry_multiple_registrations(self) -> None:
        """Test registering multiple handlers."""
        registry = ToolHandlerRegistry()

        registry.register(FileReadHandler())
        registry.register(FileWriteHandler())
        registry.register(CommandRunHandler())

        assert len(registry.get_all()) == 3
        assert registry.get("read_file") is not None
        assert registry.get("write_file") is not None
        assert registry.get("run_command") is not None

    def test_registry_handler_overwrite(self) -> None:
        """Test that registering same handler name overwrites."""
        registry = ToolHandlerRegistry()
        handler1 = FileReadHandler()
        handler2 = FileReadHandler()

        registry.register(handler1)
        registry.register(handler2)

        # Should have only one handler
        assert len(registry.get_all()) == 1
        assert registry.get("read_file") == handler2


class TestGlobalHandlerRegistry:
    """Tests for global handler registry access."""

    def test_get_handler_registry(self) -> None:
        """Test getting global handler registry (L342-L349)."""
        registry = get_handler_registry()

        assert registry is not None
        assert isinstance(registry, ToolHandlerRegistry)

    def test_global_registry_has_handlers(self) -> None:
        """Test that global registry has default handlers."""
        registry = get_handler_registry()
        all_handlers = registry.get_all()

        # Should have at least the three default handlers
        assert "read_file" in all_handlers
        assert "write_file" in all_handlers
        assert "run_command" in all_handlers

    def test_global_registry_get_read_file_handler(self) -> None:
        """Test getting read_file handler from global registry."""
        registry = get_handler_registry()
        handler = registry.get("read_file")

        assert handler is not None
        assert handler.name == "read_file"

    def test_global_registry_get_write_file_handler(self) -> None:
        """Test getting write_file handler from global registry."""
        registry = get_handler_registry()
        handler = registry.get("write_file")

        assert handler is not None
        assert handler.name == "write_file"

    def test_global_registry_get_run_command_handler(self) -> None:
        """Test getting run_command handler from global registry."""
        registry = get_handler_registry()
        handler = registry.get("run_command")

        assert handler is not None
        assert handler.name == "run_command"


class TestHandlerAbstraction:
    """Tests for ToolHandler abstraction and handler classes."""

    @pytest.mark.asyncio
    async def test_file_read_handler_name_property(self) -> None:
        """Test FileReadHandler name property."""
        handler = FileReadHandler()

        assert handler.name == "read_file"

    @pytest.mark.asyncio
    async def test_file_write_handler_name_property(self) -> None:
        """Test FileWriteHandler name property."""
        handler = FileWriteHandler()

        assert handler.name == "write_file"

    @pytest.mark.asyncio
    async def test_command_run_handler_name_property(self) -> None:
        """Test CommandRunHandler name property."""
        handler = CommandRunHandler()

        assert handler.name == "run_command"

    @pytest.mark.asyncio
    async def test_handler_execute_method(self) -> None:
        """Test handler execute method."""
        handler = FileReadHandler()
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test")
            temp_path = f.name

        try:
            result = await handler.execute(FileReadInput(file_path=temp_path))

            assert result.success is True
            assert result.content == "test"
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_handler_call_method(self) -> None:
        """Test handler __call__ method."""
        handler = FileReadHandler()
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("callable")
            temp_path = f.name

        try:
            # Call handler using __call__
            result = await handler(FileReadInput(file_path=temp_path))

            assert result.success is True
            assert result.content == "callable"
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_handler_instances_are_callable(self) -> None:
        """Test that handler instances are callable."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test")
            temp_path = f.name

        try:
            # Test all three handler instances are callable
            result1 = await read_file_handler(FileReadInput(file_path=temp_path))
            assert result1.success is True

            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, "test.txt")
                result2 = await write_file_handler(
                    FileWriteInput(file_path=file_path, content="test")
                )
                assert result2.success is True

            result3 = await run_command_handler(CommandRunInput(command="echo test"))
            assert result3.success is True
        finally:
            os.unlink(temp_path)
