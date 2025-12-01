from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterable, List, Protocol

from gearmeshing_ai.mcp_client.schemas.core import McpServerRef, McpTool, ToolArgument, ToolCallResult


class SyncStrategy(Protocol):
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


class AsyncStrategy(Protocol):
    async def list_tools(self, server_id: str) -> List[McpTool]: ...

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> ToolCallResult: ...

    async def stream_events(self, server_id: str, path: str = "/sse") -> AsyncIterator[str]: ...

    async def stream_events_parsed(self, server_id: str, path: str = "/sse") -> AsyncIterator[Dict[str, Any]]: ...
