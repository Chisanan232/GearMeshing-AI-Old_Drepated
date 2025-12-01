from __future__ import annotations

class McpClientError(Exception):
    pass


class ToolAccessDeniedError(McpClientError):
    def __init__(self, agent_id: str, server_id: str, tool_name: str) -> None:
        super().__init__(
            f"Agent '{agent_id}' is not permitted to call tool '{tool_name}' on server '{server_id}'"
        )


class ServerNotFoundError(McpClientError):
    def __init__(self, server_id: str) -> None:
        super().__init__(f"MCP server not found: '{server_id}'")


class ToolInvocationError(McpClientError):
    def __init__(self, server_id: str, tool_name: str, message: str) -> None:
        super().__init__(f"Tool invocation failed for '{tool_name}' on '{server_id}': {message}")
