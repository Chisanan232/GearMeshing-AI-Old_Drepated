from __future__ import annotations
from typing import Any, Dict, Iterable, List, Sequence
import httpx

from ..models import ToolMetadata, ToolResult
from ..config import MCPConfig


class GatewayStrategy:
    """Gateway transport strategy (e.g., HTTP gateway for MCP)."""

    def __init__(self, config: MCPConfig) -> None:
        self._config = config
        self._client = httpx.Client(timeout=self._config.timeout, follow_redirects=True)

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._config.session_id:
            headers["mcp-session-id"] = self._config.session_id
        return headers

    def list_tools(self) -> Sequence[ToolMetadata]:
        # Skeleton: no real call; return empty list
        return []

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        # Skeleton: no real call; return a placeholder result
        return ToolResult(ok=True, data=None, error=None, raw={"tool": tool_name, "args": arguments})

    def stream_call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Iterable[ToolResult]:
        # Skeleton: single-yield placeholder
        yield ToolResult(ok=True, data=None, error=None, raw={"tool": tool_name, "args": arguments, "stream": True})
