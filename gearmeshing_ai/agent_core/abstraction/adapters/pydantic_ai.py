"""Pydantic AI Framework Adapter.

This module provides an adapter that implements the AIAgentBase abstraction
for the Pydantic AI framework, enabling seamless integration with the unified
agent system.
"""

from typing import Any, Dict, Optional

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

            self._initialized = True
            logger.debug(f"Pydantic AI agent initialized: {self._config.name}")

        except ImportError as e:
            raise RuntimeError("Pydantic AI is not installed. Install it with: pip install pydantic-ai") from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pydantic AI agent: {e}") from e

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
