"""Tool handlers for AI agent capabilities.

This module implements handlers for file operations and command execution using
an abstraction-based, object-oriented approach for better maintainability and extensibility.
"""

import asyncio
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from gearmeshing_ai.core.logging_config import get_logger

from .definitions import (
    CommandRunInput,
    CommandRunOutput,
    FileReadInput,
    FileReadOutput,
    FileWriteInput,
    FileWriteOutput,
)

logger = get_logger(__name__)

# Type variables for generic handler
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class ToolHandler(ABC, Generic[InputType, OutputType]):
    """Abstract base class for tool handlers.

    Provides a common interface for all tool handlers with logging and error handling.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the tool handler name."""

    @abstractmethod
    async def execute(self, input_data: InputType) -> OutputType:
        """Execute the tool operation.

        Args:
            input_data: Input data for the tool

        Returns:
            Output data from the tool execution
        """

    async def __call__(self, input_data: InputType) -> OutputType:
        """Make handler callable.

        Args:
            input_data: Input data for the tool

        Returns:
            Output data from the tool execution
        """
        return await self.execute(input_data)


class FileReadHandler(ToolHandler[FileReadInput, FileReadOutput]):
    """Handler for file read operations."""

    @property
    def name(self) -> str:
        """Get handler name."""
        return "read_file"

    async def execute(self, input_data: FileReadInput) -> FileReadOutput:
        """Execute file read operation.

        Args:
            input_data: FileReadInput with file_path and encoding

        Returns:
            FileReadOutput with success status and file content or error
        """
        try:
            file_path = Path(input_data.file_path)

            # Validate file exists
            if not file_path.exists():
                return FileReadOutput(
                    success=False,
                    file_path=str(file_path),
                    error=f"File not found: {file_path}",
                )

            # Validate it's a file (not directory)
            if not file_path.is_file():
                return FileReadOutput(
                    success=False,
                    file_path=str(file_path),
                    error=f"Path is not a file: {file_path}",
                )

            # Read file content
            content = file_path.read_text(encoding=input_data.encoding)
            size_bytes = file_path.stat().st_size

            logger.info(f"Successfully read file: {file_path} ({size_bytes} bytes)")

            return FileReadOutput(
                success=True,
                content=content,
                file_path=str(file_path),
                size_bytes=size_bytes,
            )

        except UnicodeDecodeError as e:
            error_msg = f"Encoding error reading {input_data.file_path}: {str(e)}"
            logger.error(error_msg)
            return FileReadOutput(
                success=False,
                file_path=input_data.file_path,
                error=error_msg,
            )
        except Exception as e:
            error_msg = f"Error reading file {input_data.file_path}: {str(e)}"
            logger.error(error_msg)
            return FileReadOutput(
                success=False,
                file_path=input_data.file_path,
                error=error_msg,
            )


class FileWriteHandler(ToolHandler[FileWriteInput, FileWriteOutput]):
    """Handler for file write operations."""

    @property
    def name(self) -> str:
        """Get handler name."""
        return "write_file"

    async def execute(self, input_data: FileWriteInput) -> FileWriteOutput:
        """Execute file write operation.

        Args:
            input_data: FileWriteInput with file_path, content, and options

        Returns:
            FileWriteOutput with success status and bytes written or error
        """
        try:
            file_path = Path(input_data.file_path)

            # Create parent directories if requested
            if input_data.create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created parent directories for {file_path}")

            # Write file content
            bytes_written = file_path.write_text(input_data.content, encoding=input_data.encoding)

            logger.info(f"Successfully wrote file: {file_path} ({bytes_written} bytes)")

            return FileWriteOutput(
                success=True,
                file_path=str(file_path),
                bytes_written=bytes_written,
            )

        except PermissionError as e:
            error_msg = f"Permission denied writing to {input_data.file_path}: {str(e)}"
            logger.error(error_msg)
            return FileWriteOutput(
                success=False,
                file_path=input_data.file_path,
                error=error_msg,
            )
        except Exception as e:
            error_msg = f"Error writing file {input_data.file_path}: {str(e)}"
            logger.error(error_msg)
            return FileWriteOutput(
                success=False,
                file_path=input_data.file_path,
                error=error_msg,
            )


class CommandRunHandler(ToolHandler[CommandRunInput, CommandRunOutput]):
    """Handler for command execution operations."""

    @property
    def name(self) -> str:
        """Get handler name."""
        return "run_command"

    async def execute(self, input_data: CommandRunInput) -> CommandRunOutput:
        """Execute command operation.

        Args:
            input_data: CommandRunInput with command and options

        Returns:
            CommandRunOutput with exit code, stdout, stderr, or error
        """
        start_time = time.time()

        try:
            # Prepare command
            cmd = input_data.command
            cwd = input_data.cwd or os.getcwd()

            # Validate working directory exists
            if not os.path.isdir(cwd):
                error_msg = f"Working directory not found: {cwd}"
                logger.error(error_msg)
                return CommandRunOutput(
                    success=False,
                    command=cmd,
                    error=error_msg,
                )

            logger.info(f"Executing command: {cmd} (cwd={cwd})")

            # Execute command
            try:
                process = await asyncio.wait_for(
                    asyncio.create_subprocess_shell(
                        cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=cwd,
                        shell=True,
                    ),
                    timeout=input_data.timeout,
                )

                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=input_data.timeout,
                )

                stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
                stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
                exit_code = process.returncode

                duration = time.time() - start_time

                logger.info(
                    f"Command completed with exit code {exit_code} "
                    f"(duration: {duration:.2f}s, stdout: {len(stdout)} chars, stderr: {len(stderr)} chars)"
                )

                return CommandRunOutput(
                    success=exit_code == 0,
                    exit_code=exit_code,
                    stdout=stdout if stdout else None,
                    stderr=stderr if stderr else None,
                    command=cmd,
                    duration_seconds=duration,
                )

            except asyncio.TimeoutError:
                duration = time.time() - start_time
                error_msg = f"Command execution timeout after {input_data.timeout} seconds"
                logger.error(error_msg)
                return CommandRunOutput(
                    success=False,
                    command=cmd,
                    error=error_msg,
                    duration_seconds=duration,
                )

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Error executing command: {str(e)}"
            logger.error(error_msg)
            return CommandRunOutput(
                success=False,
                command=input_data.command,
                error=error_msg,
                duration_seconds=duration,
            )


class ToolHandlerRegistry:
    """Registry for managing tool handlers."""

    def __init__(self) -> None:
        """Initialize the handler registry."""
        self._handlers: dict[str, ToolHandler[Any, Any]] = {}

    def register(self, handler: ToolHandler[Any, Any]) -> None:
        """Register a tool handler.

        Args:
            handler: ToolHandler instance to register
        """
        self._handlers[handler.name] = handler
        logger.debug(f"Registered tool handler: {handler.name}")

    def get(self, name: str) -> Optional[ToolHandler[Any, Any]]:
        """Get a tool handler by name.

        Args:
            name: Name of the handler

        Returns:
            ToolHandler instance or None if not found
        """
        return self._handlers.get(name)

    def get_all(self) -> dict[str, ToolHandler[Any, Any]]:
        """Get all registered handlers.

        Returns:
            Dictionary of all registered handlers
        """
        return self._handlers.copy()


# Create global registry
_handler_registry = ToolHandlerRegistry()

# Create handler instances
read_file_handler = FileReadHandler()
write_file_handler = FileWriteHandler()
run_command_handler = CommandRunHandler()

# Register handlers
_handler_registry.register(read_file_handler)
_handler_registry.register(write_file_handler)
_handler_registry.register(run_command_handler)


def get_handler_registry() -> ToolHandlerRegistry:
    """Get the global tool handler registry.

    Returns:
        ToolHandlerRegistry instance
    """
    return _handler_registry


# Set handlers on tool definitions
from . import definitions

definitions.read_file_tool.handler = read_file_handler
definitions.write_file_tool.handler = write_file_handler
definitions.run_command_tool.handler = run_command_handler
