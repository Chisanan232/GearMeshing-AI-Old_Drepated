from __future__ import annotations


class MCPError(Exception):
    """Base class for MCP client errors."""


class MCPTransportError(MCPError):
    """Transport-level failure (network, stdio, timeouts)."""


class MCPProtocolError(MCPError):
    """JSON-RPC or protocol violation."""


class MCPPermissionError(MCPError):
    """Policy denied an action."""


class MCPConfigError(MCPError):
    """Configuration is invalid or missing required fields."""
