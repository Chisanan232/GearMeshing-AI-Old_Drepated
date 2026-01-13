"""AI Agent Tools Module.

This module provides tool definitions and handlers for extending AI agent capabilities
with file operations, command execution, and other software development tasks.
"""

from .definitions import (
    read_file_tool,
    write_file_tool,
    run_command_tool,
)
from .handlers import (
    ToolHandler,
    FileReadHandler,
    FileWriteHandler,
    CommandRunHandler,
    ToolHandlerRegistry,
    read_file_handler,
    write_file_handler,
    run_command_handler,
    get_handler_registry,
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
