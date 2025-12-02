"""Error types for the MCP client package.

Defines a small hierarchy of exceptions raised by clients and strategies to
signal policy violations, missing servers, and tool invocation failures.
"""

from __future__ import annotations


class McpClientError(Exception):
    """Base error for all MCP client exceptions."""


class ToolAccessDeniedError(McpClientError):
    """Raised when an operation is blocked by policy enforcement."""

    def __init__(self, agent_id: str, server_id: str, tool_name: str) -> None:
        super().__init__(f"Agent '{agent_id}' is not permitted to call tool '{tool_name}' on server '{server_id}'")


class ServerNotFoundError(McpClientError):
    """Raised when no strategy can find or serve the requested server."""

    def __init__(self, server_id: str) -> None:
        super().__init__(f"MCP server not found: '{server_id}'")


class ToolInvocationError(McpClientError):
    """Raised for unsuccessful tool invocations with additional context."""

    def __init__(self, server_id: str, tool_name: str, message: str) -> None:
        super().__init__(f"Tool invocation failed for '{tool_name}' on '{server_id}': {message}")
