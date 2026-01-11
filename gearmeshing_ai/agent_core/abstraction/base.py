"""Base abstraction for AI agents.

This module defines the core interfaces that all AI agent implementations
must adhere to, enabling framework-agnostic agent usage throughout the project.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AIAgentConfig:
    """Configuration for initializing an AI agent.

    Attributes:
        name: Unique identifier for the agent instance
        framework: The AI framework being used (e.g., 'pydantic_ai', 'langchain')
        model: The LLM model identifier (e.g., 'gpt-4o', 'claude-3-opus')
        system_prompt: System prompt/instructions for the agent
        tools: List of tool definitions available to the agent
        temperature: Model temperature for response generation
        max_tokens: Maximum tokens for response generation
        timeout: Request timeout in seconds
        metadata: Additional framework-specific configuration
    """

    name: str
    framework: str
    model: str
    system_prompt: Optional[str] = None
    tools: List[Dict[str, Any]] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "framework": self.framework,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "metadata": self.metadata,
        }


@dataclass
class AIAgentResponse:
    """Response from an AI agent.

    Attributes:
        content: The main response content (text, structured data, etc.)
        tool_calls: List of tool calls made by the agent
        metadata: Additional response metadata (tokens used, latency, etc.)
        error: Error message if the request failed
        success: Whether the request was successful
    """

    content: Any
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "content": self.content,
            "tool_calls": self.tool_calls,
            "metadata": self.metadata,
            "error": self.error,
            "success": self.success,
        }


class AIAgentBase(ABC):
    """Abstract base class for all AI agent implementations.

    This class defines the interface that all AI agent frameworks must implement
    to be used within the GearMeshing-AI system. Implementations should handle
    framework-specific details while exposing a unified API.

    Subclasses must implement:
    - initialize(): Set up the agent with configuration
    - invoke(): Execute the agent with input
    - stream(): Stream responses from the agent
    - cleanup(): Clean up resources
    """

    def __init__(self, config: AIAgentConfig) -> None:
        """Initialize the agent with configuration.

        Args:
            config: AIAgentConfig instance with agent parameters
        """
        self._config = config
        self._initialized = False

    @property
    def config(self) -> AIAgentConfig:
        """Get the agent configuration."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Check if the agent is initialized."""
        return self._initialized

    @property
    def framework(self) -> str:
        """Get the framework name."""
        return self._config.framework

    @property
    def model(self) -> str:
        """Get the model identifier."""
        return self._config.model

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent.

        This method should set up any framework-specific resources,
        validate configuration, and prepare the agent for use.

        Raises:
            RuntimeError: If initialization fails
        """
        pass

    @abstractmethod
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
            **kwargs: Framework-specific parameters

        Returns:
            AIAgentResponse with the agent's response

        Raises:
            RuntimeError: If invocation fails
        """
        pass

    @abstractmethod
    async def stream(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Stream responses from the agent.

        This is an async generator that yields response chunks.

        Args:
            input_text: The input prompt or query
            context: Optional context dictionary
            **kwargs: Framework-specific parameters

        Yields:
            Response chunks as they become available
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up agent resources.

        This method should release any held resources, close connections, etc.
        """
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"name={self._config.name}, "
            f"framework={self._config.framework}, "
            f"model={self._config.model})"
        )
