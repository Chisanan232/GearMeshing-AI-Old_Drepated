from __future__ import annotations

from typing import Any, Dict, Sequence

from ..config import MCPConfig
from ..models import ToolMetadata, ToolResult


class DirectStrategy:
    """Direct-to-server transport strategy (placeholder skeleton)."""

    def __init__(self, config: MCPConfig) -> None:
        self._config = config

    def list_tools(self) -> Sequence[ToolMetadata]:
        return []

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, data=None, error=None, raw={"direct": True, "tool": tool_name, "args": arguments})

    def stream_call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        yield ToolResult(
            ok=True, data=None, error=None, raw={"direct": True, "tool": tool_name, "args": arguments, "stream": True}
        )
