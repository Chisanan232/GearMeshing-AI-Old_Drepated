"""Configuration system for AI agent abstraction.

This module provides configuration management for the agent abstraction layer,
including environment-based configuration and settings validation.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentAbstractionConfig:
    """Configuration for the AI agent abstraction layer.

    Attributes:
        framework: Active AI framework (from AI_AGENT_FRAMEWORK env var)
        cache_enabled: Whether to enable agent caching
        cache_max_size: Maximum number of agents to cache (0 = unlimited)
        cache_ttl: Time-to-live for cached agents in seconds
        default_timeout: Default timeout for agent operations
        auto_initialize: Whether to auto-initialize on import
    """

    framework: Optional[str] = None
    cache_enabled: bool = True
    cache_max_size: int = 10
    cache_ttl: Optional[float] = None
    default_timeout: Optional[float] = None
    auto_initialize: bool = True

    @classmethod
    def from_env(cls) -> "AgentAbstractionConfig":
        """Load configuration from environment variables.

        Environment variables:
        - AI_AGENT_FRAMEWORK: Active framework (e.g., 'pydantic_ai')
        - AI_AGENT_CACHE_ENABLED: Enable caching (true/false)
        - AI_AGENT_CACHE_MAX_SIZE: Max cache size (integer)
        - AI_AGENT_CACHE_TTL: Cache TTL in seconds (float)
        - AI_AGENT_DEFAULT_TIMEOUT: Default timeout (float)
        - AI_AGENT_AUTO_INIT: Auto-initialize (true/false)

        Returns:
            AgentAbstractionConfig instance with environment values
        """
        def parse_bool(value: Optional[str], default: bool) -> bool:
            if value is None:
                return default
            return value.lower() in ("true", "1", "yes", "on")

        def parse_int(value: Optional[str], default: int) -> int:
            if value is None:
                return default
            try:
                return int(value)
            except ValueError:
                return default

        def parse_float(value: Optional[str], default: Optional[float]) -> Optional[float]:
            if value is None:
                return default
            try:
                return float(value)
            except ValueError:
                return default

        return cls(
            framework=os.getenv("AI_AGENT_FRAMEWORK"),
            cache_enabled=parse_bool(os.getenv("AI_AGENT_CACHE_ENABLED"), True),
            cache_max_size=parse_int(os.getenv("AI_AGENT_CACHE_MAX_SIZE"), 10),
            cache_ttl=parse_float(os.getenv("AI_AGENT_CACHE_TTL"), None),
            default_timeout=parse_float(os.getenv("AI_AGENT_DEFAULT_TIMEOUT"), None),
            auto_initialize=parse_bool(os.getenv("AI_AGENT_AUTO_INIT"), True),
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "framework": self.framework,
            "cache_enabled": self.cache_enabled,
            "cache_max_size": self.cache_max_size,
            "cache_ttl": self.cache_ttl,
            "default_timeout": self.default_timeout,
            "auto_initialize": self.auto_initialize,
        }
