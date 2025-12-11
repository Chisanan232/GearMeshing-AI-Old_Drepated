"""Schemas for the MCP client.

Defines base and domain Pydantic models used by strategies and clients,
including configuration types. This package centralizes validation and
serialization for the client layer. Re-exports `MCPConfig` for convenience.
"""

from .config import MCPConfig

__all__ = [
    "MCPConfig",
]
