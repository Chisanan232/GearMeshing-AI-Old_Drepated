"""MCP client package.

Provides:
- Domain and DTO schemas under `schemas` and `gateway_api.models`.
- Strategies for direct and gateway-backed MCP access (sync/async).
- High-level client facades (`McpClient`, `AsyncMcpClient`) with policy support.
"""

from .schemas.config import MCPConfig

__all__ = [
    "MCPConfig",
]
