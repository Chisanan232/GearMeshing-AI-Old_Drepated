from __future__ import annotations
from typing import Any, Dict, Iterable, List, Sequence, Optional
import httpx

from ..models import ToolMetadata, ToolResult
from ..config import MCPConfig


class GatewayStrategy:
    """Gateway transport strategy (HTTP gateway for MCP)."""

    def __init__(self, config: MCPConfig, client: Optional[httpx.Client] = None) -> None:
        self._config = config
        self._client = client or httpx.Client(timeout=self._config.timeout, follow_redirects=True)

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._config.session_id:
            headers["mcp-session-id"] = self._config.session_id
        return headers

    def list_tools(self) -> Sequence[ToolMetadata]:
        base = (self._config.base_url or "http://localhost:3004").rstrip("/")
        try:
            resp = self._client.get(f"{base}/mcp/", headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
            tools_raw = data.get("tools", data) if isinstance(data, dict) else data
            tools: List[ToolMetadata] = []
            if isinstance(tools_raw, list):
                for t in tools_raw:
                    if isinstance(t, dict):
                        tools.append(ToolMetadata.model_validate(t))
            return tools
        except httpx.RequestError as e:
            return []

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        base = (self._config.base_url or "http://localhost:3004").rstrip("/")
        try:
            resp = self._client.post(
                f"{base}/mcp/invoke/{tool_name}",
                headers=self._headers(),
                json=arguments,
            )
            resp.raise_for_status()
            payload = resp.json()
            return ToolResult(ok=True, data=payload, raw=payload)
        except httpx.HTTPStatusError as e:
            return ToolResult(ok=False, error=f"HTTP {e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            return ToolResult(ok=False, error=str(e))

    def stream_call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Iterable[ToolResult]:
        # No true streaming in this simple gateway; yield once
        yield self.call_tool(tool_name, arguments)
