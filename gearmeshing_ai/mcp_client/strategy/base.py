from __future__ import annotations
from typing import Any, Iterable, Protocol

from gearmeshing_ai.mcp_client.schemas.core import McpServerRef, McpTool, ToolCallResult


class StrategyBase(Protocol):
    def list_servers(self) -> Iterable[McpServerRef]: ...

    def list_tools(self, server_id: str) -> Iterable[McpTool]: ...

    def call_tool(
        self,
        server_id: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        agent_id: str | None = None,
    ) -> ToolCallResult: ...
