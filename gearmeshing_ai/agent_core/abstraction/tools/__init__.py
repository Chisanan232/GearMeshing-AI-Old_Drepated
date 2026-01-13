"""AI Agent Tools Module.

This module provides tool definitions and handlers for extending AI agent capabilities
with file operations, command execution, and other software development tasks.
"""

from .definitions import (
    read_file_tool,
    run_command_tool,
    write_file_tool,
)
from .handlers import (
    CommandRunHandler,
    FileReadHandler,
    FileWriteHandler,
    ToolHandler,
    ToolHandlerRegistry,
    get_handler_registry,
    read_file_handler,
    run_command_handler,
    write_file_handler,
)

__all__ = [
    # Tool definitions
    "read_file_tool",
    "write_file_tool",
    "run_command_tool",
    # Handler abstractions
    "ToolHandler",
    "FileReadHandler",
    "FileWriteHandler",
    "CommandRunHandler",
    "ToolHandlerRegistry",
    # Handler instances
    "read_file_handler",
    "write_file_handler",
    "run_command_handler",
    # Registry access
    "get_handler_registry",
]
