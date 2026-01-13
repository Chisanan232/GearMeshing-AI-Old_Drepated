"""Pydantic AI Framework Adapter.

This module provides an adapter that implements the AIAgentBase abstraction
for the Pydantic AI framework, enabling seamless integration with the unified
agent system. Includes support for file operations and command execution tools.
"""

from typing import Any, Dict, Optional

from gearmeshing_ai.core.logging_config import get_logger
from gearmeshing_ai.agent_core.abstraction.tools import (
    read_file_handler,
    write_file_handler,
    run_command_handler,
    read_file_tool,
    write_file_tool,
    run_command_tool,
)
from gearmeshing_ai.agent_core.abstraction.tools.definitions import (
    FileReadInput,
    FileWriteInput,
    CommandRunInput,
)

from ..base import AIAgentBase, AIAgentConfig, AIAgentResponse

logger = get_logger(__name__)


# Module-level tool functions for Pydantic AI
# These must be defined at module level for proper schema generation
async def read_file(file_path: str, encoding: str = "utf-8") -> str:
    """Read a file from the filesystem."""
    input_data = FileReadInput(file_path=file_path, encoding=encoding)
    result = await read_file_handler(input_data)
    return result.model_dump_json()


async def write_file(
    file_path: str,
    content: str,
    encoding: str = "utf-8",
    create_dirs: bool = True,
) -> str:
    """Write content to a file on the filesystem."""
    input_data = FileWriteInput(
        file_path=file_path,
        content=content,
        encoding=encoding,
        create_dirs=create_dirs,
    )
    result = await write_file_handler(input_data)
    return result.model_dump_json()


async def run_command(
    command: str,
    cwd: Optional[str] = None,
    timeout: float = 30.0,
    shell: bool = True,
) -> str:
    """Execute a shell command and capture output."""
    input_data = CommandRunInput(
        command=command,
        cwd=cwd,
        timeout=timeout,
        shell=shell,
    )
    result = await run_command_handler(input_data)
    return result.model_dump_json()


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
            # Register module-level tools using Pydantic AI's decorator style
            # Using the decorator syntax to properly register tools
            agent.tool(read_file, name="read_file")
            agent.tool(write_file, name="write_file")
            agent.tool(run_command, name="run_command")

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
