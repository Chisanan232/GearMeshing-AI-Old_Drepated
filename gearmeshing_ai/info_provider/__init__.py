"""MCP info provider facade.

This package exposes the minimal interfaces and concrete facades for
retrieving MCP metadata (endpoints and tools) that AI agents or other
components can consume.

It currently re-exports the implementations from ``gearmeshing_ai.mcp_client``
so callers can gradually migrate imports without breaking existing code.

Public surface:
- MCPInfoProvider / AsyncMCPInfoProvider: protocols describing the read-only
  MCP info contract (endpoints + tools listing, with optional pagination).
- McpClient / AsyncMcpClient: concrete sync/async facades that implement
  these protocols and apply ToolPolicy-based restrictions.
"""

from __future__ import annotations

from gearmeshing_ai.mcp_client.base import AsyncMCPInfoProvider, MCPInfoProvider
from gearmeshing_ai.mcp_client.client_async import AsyncMcpClient
from gearmeshing_ai.mcp_client.client_sync import McpClient

__all__ = [
    "MCPInfoProvider",
    "AsyncMCPInfoProvider",
    "McpClient",
    "AsyncMcpClient",
]
