"""Configuration system for AI agent abstraction.

This module provides configuration management for the agent abstraction layer,
including environment-based configuration and settings validation.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentAbstractionConfig(BaseSettings):
    """Configuration for the AI agent abstraction layer.

    Attributes:
        framework: Active AI framework (from AI_AGENT_FRAMEWORK env var)
        cache_enabled: Whether to enable agent caching
        cache_max_size: Maximum number of agents to cache (0 = unlimited)
        cache_ttl: Time-to-live for cached agents in seconds
        default_timeout: Default timeout for agent operations
        auto_initialize: Whether to auto-initialize on import

    Environment variables:
    - AI_AGENT_FRAMEWORK: Active framework (e.g., 'pydantic_ai')
    - AI_AGENT_CACHE_ENABLED: Enable caching (true/false)
    - AI_AGENT_CACHE_MAX_SIZE: Max cache size (integer)
    - AI_AGENT_CACHE_TTL: Cache TTL in seconds (float)
    - AI_AGENT_DEFAULT_TIMEOUT: Default timeout (float)
    - AI_AGENT_AUTO_INIT: Auto-initialize (true/false)
    """

    model_config = SettingsConfigDict(
        env_prefix="AI_AGENT_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    framework: Optional[str] = Field(
        default=None,
        description="Active AI framework (e.g., 'pydantic_ai')",
    )
    cache_enabled: bool = Field(
        default=True,
        description="Whether to enable agent caching",
    )
    cache_max_size: int = Field(
        default=10,
        description="Maximum number of agents to cache (0 = unlimited)",
    )
    cache_ttl: Optional[float] = Field(
        default=None,
        description="Time-to-live for cached agents in seconds",
    )
    default_timeout: Optional[float] = Field(
        default=None,
        description="Default timeout for agent operations",
    )
    auto_initialize: bool = Field(
        default=True,
        description="Whether to auto-initialize on import",
    )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return self.model_dump()
