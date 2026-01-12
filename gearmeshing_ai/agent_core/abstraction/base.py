"""Base abstraction for AI agents.

This module defines the core interfaces that all AI agent implementations
must adhere to, enabling framework-agnostic agent usage throughout the project.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class AIAgentConfig(BaseModel):
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
        metadata: Additional framework-specific configuration (e.g., output_type for Pydantic AI)
    """

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    name: str = Field(..., description="Unique identifier for the agent instance")
    framework: str = Field(..., description="The AI framework being used")
    model: str = Field(..., description="The LLM model identifier")
    system_prompt: Optional[str] = Field(None, description="System prompt/instructions for the agent")
    tools: List[Dict[str, Any]] = Field(default_factory=list, description="List of tool definitions")
    temperature: float = Field(default=0.7, description="Model temperature for response generation")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens for response generation")
    timeout: Optional[float] = Field(None, description="Request timeout in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional framework-specific configuration")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()


class AIAgentResponse(BaseModel):
    """Response from an AI agent.

    Attributes:
        content: The main response content (text, structured data, etc.)
        tool_calls: List of tool calls made by the agent
        metadata: Additional response metadata (tokens used, latency, etc.)
        error: Error message if the request failed
        success: Whether the request was successful
    """

    model_config = ConfigDict(frozen=False)

    content: Any = Field(None, description="The main response content")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="List of tool calls made by the agent")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional response metadata")
    error: Optional[str] = Field(None, description="Error message if the request failed")
    success: bool = Field(default=True, description="Whether the request was successful")

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return self.model_dump()


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

    def build_init_kwargs(self) -> Dict[str, Any]:
        """Build framework-specific initialization kwargs.

        This method constructs the actual kwargs dictionary that will be passed
        to the framework's constructor, based on the agent configuration.

        Returns:
            Dictionary of kwargs for the framework constructor

        Raises:
            ValueError: If required parameters cannot be built from config
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement build_init_kwargs()")

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent.

        This method should set up any framework-specific resources,
        validate configuration, and prepare the agent for use.

        Raises:
            RuntimeError: If initialization fails
        """

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

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up agent resources.

        This method should release any held resources, close connections, etc.
        """

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
