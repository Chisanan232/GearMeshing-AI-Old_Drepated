"""Tool definitions for AI agent capabilities.

This module defines the tools available to AI agents for file operations,
command execution, and other software development tasks.
"""

from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel, Field, ConfigDict


class FileReadInput(BaseModel):
    """Input schema for file read operation."""

    file_path: str = Field(..., description="Absolute path to the file to read")
    encoding: str = Field(default="utf-8", description="File encoding (default: utf-8)")


class FileReadOutput(BaseModel):
    """Output schema for file read operation."""

    success: bool = Field(..., description="Whether the operation succeeded")
    content: Optional[str] = Field(None, description="File content if successful")
    error: Optional[str] = Field(None, description="Error message if failed")
    file_path: str = Field(..., description="Path of the file read")
    size_bytes: Optional[int] = Field(None, description="Size of file in bytes")


class FileWriteInput(BaseModel):
    """Input schema for file write operation."""

    file_path: str = Field(..., description="Absolute path to the file to write")
    content: str = Field(..., description="Content to write to the file")
    encoding: str = Field(default="utf-8", description="File encoding (default: utf-8)")
    create_dirs: bool = Field(
        default=True,
        description="Create parent directories if they don't exist",
    )


class FileWriteOutput(BaseModel):
    """Output schema for file write operation."""

    success: bool = Field(..., description="Whether the operation succeeded")
    file_path: str = Field(..., description="Path of the file written")
    bytes_written: Optional[int] = Field(None, description="Number of bytes written")
    error: Optional[str] = Field(None, description="Error message if failed")


class CommandRunInput(BaseModel):
    """Input schema for command execution."""

    command: str = Field(..., description="Command to execute (shell command or script)")
    cwd: Optional[str] = Field(None, description="Working directory for command execution")
    timeout: Optional[float] = Field(
        default=30.0,
        description="Timeout in seconds (default: 30)",
    )
    shell: bool = Field(default=True, description="Execute as shell command")


class CommandRunOutput(BaseModel):
    """Output schema for command execution."""

    success: bool = Field(..., description="Whether the command executed successfully")
    exit_code: Optional[int] = Field(None, description="Command exit code")
    stdout: Optional[str] = Field(None, description="Standard output")
    stderr: Optional[str] = Field(None, description="Standard error")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    command: str = Field(..., description="Command that was executed")
    duration_seconds: Optional[float] = Field(None, description="Execution duration in seconds")


class ToolDefinition(BaseModel):
    """Pydantic model for tool definitions.
    
    Provides a structured, validated, and flexible way to define AI agent tools
    with clear input/output schemas and handler functions.
    """

    name: str = Field(..., description="Unique identifier for the tool")
    description: str = Field(..., description="Human-readable description of what the tool does")
    input_schema: Type[BaseModel] = Field(..., description="Pydantic model class for input validation")
    output_schema: Type[BaseModel] = Field(..., description="Pydantic model class for output")
    handler: Optional[Callable] = Field(
        default=None,
        description="Async handler function that executes the tool (can be set later)"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool definition to dictionary format.

        Returns:
            Dictionary representation of tool definition with JSON schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.model_json_schema(),
        }

    def to_full_dict(self) -> Dict[str, Any]:
        """Convert tool definition to full dictionary format including output schema.

        Returns:
            Complete dictionary representation of tool definition
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.model_json_schema(),
            "output_schema": self.output_schema.model_json_schema(),
        }

    def get_input_schema_json(self) -> Dict[str, Any]:
        """Get the input schema as JSON schema.

        Returns:
            JSON schema for input validation
        """
        return self.input_schema.model_json_schema()

    def get_output_schema_json(self) -> Dict[str, Any]:
        """Get the output schema as JSON schema.

        Returns:
            JSON schema for output validation
        """
        return self.output_schema.model_json_schema()

    def set_handler(self, handler: Callable) -> None:
        """Set the handler function for this tool.

        Args:
            handler: Async handler function to execute the tool
        """
        self.handler = handler

    def has_handler(self) -> bool:
        """Check if a handler has been set for this tool.

        Returns:
            True if handler is set, False otherwise
        """
        return self.handler is not None


# Tool instances
read_file_tool = ToolDefinition(
    name="read_file",
    description="Read the contents of a file from the filesystem. Useful for examining source code, configuration files, or any text-based file.",
    input_schema=FileReadInput,
    output_schema=FileReadOutput,
    handler=None,  # Will be set by handler module
)

write_file_tool = ToolDefinition(
    name="write_file",
    description="Write content to a file on the filesystem. Creates the file if it doesn't exist, or overwrites it if it does. Can create parent directories automatically.",
    input_schema=FileWriteInput,
    output_schema=FileWriteOutput,
    handler=None,  # Will be set by handler module
)

run_command_tool = ToolDefinition(
    name="run_command",
    description="Execute a shell command or script and capture its output. Useful for running build commands, tests, or other system operations.",
    input_schema=CommandRunInput,
    output_schema=CommandRunOutput,
    handler=None,  # Will be set by handler module
)
