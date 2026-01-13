"""Pydantic AI Framework Adapter.

This module provides an adapter that implements the AIAgentBase abstraction
for the Pydantic AI framework, enabling seamless integration with the unified
agent system. Includes support for file operations and command execution tools.
"""

from typing import Any, Dict, Optional

from pydantic_ai import RunContext

from gearmeshing_ai.agent_core.abstraction.tools import (
    read_file_handler,
    run_command_handler,
    write_file_handler,
)
from gearmeshing_ai.agent_core.abstraction.tools.definitions import (
    CommandRunInput,
    FileReadInput,
    FileWriteInput,
)
from gearmeshing_ai.core.logging_config import get_logger

from ..base import AIAgentBase, AIAgentConfig, AIAgentResponse

logger = get_logger(__name__)


class PydanticAIAgent(AIAgentBase):
    """Adapter for Pydantic AI framework.

    This class wraps Pydantic AI agents to provide a unified interface
    compatible with the AIAgentBase abstraction.

    Attributes:
        _agent: The underlying Pydantic AI agent instance
        _model: The LLM model instance
    """

    def __init__(self, config: AIAgentConfig) -> None:
        """Initialize the Pydantic AI agent adapter.

        Args:
            config: AIAgentConfig with agent parameters
        """
        super().__init__(config)
        self._agent = None
        self._model = None
        self._enable_tools = True
        self._capability_event_repo = None

    def build_init_kwargs(self) -> Dict[str, Any]:
        """Build Pydantic AI-specific initialization kwargs.

        Constructs the kwargs dictionary for the Pydantic AI Agent constructor
        based on the agent configuration.

        Returns:
            Dictionary of kwargs for Agent constructor

        Raises:
            ValueError: If required parameters are missing
        """
        kwargs: Dict[str, Any] = {
            "model": self._config.model,
        }

        # Add system prompt if provided
        if self._config.system_prompt:
            kwargs["system_prompt"] = self._config.system_prompt

        # Build model settings from config
        model_settings: Dict[str, Any] = {}

        if self._config.temperature is not None:
            model_settings["temperature"] = self._config.temperature

        if self._config.max_tokens is not None:
            model_settings["max_tokens"] = self._config.max_tokens

        # Add framework-specific model settings from config.model_settings
        if self._config.model_settings:
            model_settings.update(self._config.model_settings)

        if model_settings:
            kwargs["model_settings"] = model_settings

        # Extract AI agent object parameters from metadata
        # These are parameters for the Agent constructor itself
        if self._config.metadata:
            kwargs.update(self._config.metadata)

        logger.debug(f"Built initialization kwargs for {self._config.name}: {list(kwargs.keys())}")

        return kwargs

    async def initialize(self) -> None:
        """Initialize the Pydantic AI agent.

        This method sets up the Pydantic AI agent using the initialization kwargs
        built from the agent configuration.

        Raises:
            RuntimeError: If initialization fails
        """
        try:
            from pydantic_ai import Agent

            logger.debug(f"Initializing Pydantic AI agent: {self._config.name} with model {self._config.model}")

            # Build initialization kwargs from config
            init_kwargs = self.build_init_kwargs()

            logger.debug(f"Using initialization kwargs for {self._config.name}: {init_kwargs}")

            # Create the agent with built kwargs
            self._agent = Agent(**init_kwargs)

            # Register tools with the agent
            self._register_tools(self._agent)

            self._initialized = True
            logger.debug(f"Pydantic AI agent initialized: {self._config.name}")

        except ImportError as e:
            raise RuntimeError("Pydantic AI is not installed. Install it with: pip install pydantic-ai") from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pydantic AI agent: {e}") from e

    def _register_tools(self, agent: Any) -> None:
        """Register file and command execution tools with the agent.

        Args:
            agent: The Pydantic AI Agent instance
        """
        if not self._enable_tools:
            logger.debug(f"Tools disabled for {self._config.name}")
            return

        try:
            # Define tools as nested functions with decorators
            # First parameter must be RunContext as per Pydantic AI documentation
            @agent.tool
            async def read_file(ctx: RunContext, file_path: str, encoding: str = "utf-8") -> str:
                """Read a file from the filesystem.

                This tool reads the contents of a file at the specified path and returns
                the file contents as a string. It supports various text encodings and is
                useful for retrieving configuration files, logs, or other text-based data.

                Args:
                    file_path: The absolute or relative path to the file to read.
                    encoding: The character encoding to use when reading the file.
                             Defaults to 'utf-8'. Common alternatives include 'ascii',
                             'latin-1', 'utf-16', etc.

                Returns:
                    A JSON string containing:
                    - success: Boolean indicating if the read was successful
                    - content: The file contents as a string
                    - file_path: The path that was read
                    - size_bytes: The size of the file in bytes

                Raises:
                    FileNotFoundError: If the specified file does not exist
                    PermissionError: If the file cannot be read due to permissions
                    UnicodeDecodeError: If the file cannot be decoded with the specified encoding
                """
                input_data = FileReadInput(file_path=file_path, encoding=encoding)
                result = await read_file_handler(input_data)
                return result.model_dump_json()

            @agent.tool
            async def write_file(
                ctx: RunContext,
                file_path: str,
                content: str,
                encoding: str = "utf-8",
                create_dirs: bool = True,
            ) -> str:
                """Write content to a file on the filesystem.

                This tool writes the provided content to a file at the specified path.
                It can create parent directories if they don't exist and supports various
                text encodings. Useful for creating configuration files, logs, or saving
                generated content.

                Args:
                    file_path: The absolute or relative path where the file should be written.
                    content: The text content to write to the file.
                    encoding: The character encoding to use when writing the file.
                             Defaults to 'utf-8'. Common alternatives include 'ascii',
                             'latin-1', 'utf-16', etc.
                    create_dirs: If True (default), creates parent directories if they don't exist.
                                If False, raises an error if parent directories are missing.

                Returns:
                    A JSON string containing:
                    - success: Boolean indicating if the write was successful
                    - file_path: The path where the file was written
                    - bytes_written: The number of bytes written to the file

                Raises:
                    PermissionError: If the file cannot be written due to permissions
                    OSError: If parent directories cannot be created (when create_dirs=True)
                """
                input_data = FileWriteInput(
                    file_path=file_path,
                    content=content,
                    encoding=encoding,
                    create_dirs=create_dirs,
                )
                result = await write_file_handler(input_data)
                return result.model_dump_json()

            @agent.tool
            async def run_command(
                ctx: RunContext,
                command: str,
                cwd: Optional[str] = None,
                timeout: float = 30.0,
                shell: bool = True,
            ) -> str:
                """Execute a shell command and capture output.

                This tool executes a shell command in the system and captures both
                stdout and stderr output. It's useful for running scripts, checking
                system status, building projects, running tests, or any other
                command-line operations.

                Args:
                    command: The shell command to execute. Can be a simple command like
                            'ls' or a complex pipeline like 'cat file.txt | grep pattern'.
                    cwd: The working directory in which to execute the command.
                         If None (default), uses the current working directory.
                    timeout: Maximum time in seconds to wait for the command to complete.
                            Defaults to 30 seconds. Raises TimeoutError if exceeded.
                    shell: If True (default), executes the command through a shell.
                           If False, executes the command directly without shell interpretation.

                Returns:
                    A JSON string containing:
                    - success: Boolean indicating if the command executed successfully
                    - exit_code: The exit code returned by the command (0 = success)
                    - command: The command that was executed
                    - stdout: Standard output from the command
                    - stderr: Standard error output from the command (if any)
                    - duration_seconds: How long the command took to execute

                Raises:
                    TimeoutError: If the command takes longer than the specified timeout
                    OSError: If the command cannot be executed
                """
                input_data = CommandRunInput(
                    command=command,
                    cwd=cwd,
                    timeout=timeout,
                    shell=shell,
                )
                result = await run_command_handler(input_data)
                return result.model_dump_json()

            logger.debug(f"Registered tools for {self._config.name}: read_file, write_file, run_command")

        except Exception as e:
            logger.error(f"Error registering tools for {self._config.name}: {e}")

    async def invoke(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AIAgentResponse:
        """Execute the agent with input.

        Args:
            input_text: The input prompt or query
            context: Optional context dictionary
            **kwargs: Additional parameters (timeout, etc.)

        Returns:
            AIAgentResponse with the agent's response

        Raises:
            RuntimeError: If agent is not initialized or invocation fails
        """
        if not self._initialized or self._agent is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        try:
            timeout = kwargs.get("timeout") or self._config.timeout

            logger.debug(f"Invoking Pydantic AI agent: {self._config.name} with input length {len(input_text)}")

            # Prepare the full prompt with context if provided
            full_prompt = input_text
            if context:
                context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
                full_prompt = f"{context_str}\n\n{input_text}"

            # Invoke the agent
            result = await self._agent.run(
                full_prompt,
                timeout=timeout,
            )

            # Extract response content
            content = result.data if hasattr(result, "data") else str(result)

            # Extract tool calls if any
            tool_calls = []
            if hasattr(result, "tool_calls"):
                tool_calls = result.tool_calls or []

            # Build metadata
            metadata = {
                "model": self._config.model,
                "framework": "pydantic_ai",
            }

            if hasattr(result, "usage"):
                metadata["usage"] = {
                    "input_tokens": getattr(result.usage, "input_tokens", None),
                    "output_tokens": getattr(result.usage, "output_tokens", None),
                }

            logger.debug(f"Pydantic AI agent invocation completed: {self._config.name}")

            return AIAgentResponse(
                content=content,
                tool_calls=tool_calls,
                metadata=metadata,
                success=True,
            )

        except Exception as e:
            logger.error(
                f"Pydantic AI agent invocation failed: {self._config.name}: {e}",
                exc_info=True,
            )
            return AIAgentResponse(
                content=None,
                error=str(e),
                success=False,
            )

    async def stream(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Stream responses from the agent.

        Args:
            input_text: The input prompt or query
            context: Optional context dictionary
            **kwargs: Additional parameters

        Yields:
            Response chunks as they become available
        """
        if not self._initialized or self._agent is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        try:
            timeout = kwargs.get("timeout") or self._config.timeout

            logger.debug(f"Streaming from Pydantic AI agent: {self._config.name}")

            # Prepare the full prompt with context if provided
            full_prompt = input_text
            if context:
                context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
                full_prompt = f"{context_str}\n\n{input_text}"

            # Stream from the agent
            async with self._agent.run_stream(
                full_prompt,
                timeout=timeout,
            ) as stream:
                async for chunk in stream:
                    if chunk:
                        yield chunk

        except Exception as e:
            logger.error(
                f"Pydantic AI agent streaming failed: {self._config.name}: {e}",
                exc_info=True,
            )
            raise

    async def cleanup(self) -> None:
        """Clean up agent resources.

        For Pydantic AI, this mainly involves clearing references.
        """
        try:
            logger.debug(f"Cleaning up Pydantic AI agent: {self._config.name}")

            if self._agent is not None:
                self._agent = None

            if self._model is not None:
                self._model = None

            self._initialized = False

        except Exception as e:
            logger.error(
                f"Error during Pydantic AI agent cleanup: {self._config.name}: {e}",
                exc_info=True,
            )
