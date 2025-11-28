from __future__ import annotations
from typing import Any, Dict, List
import json as _json

import httpx

from gearmeshing_ai.mcp_client.config import MCPConfig
from gearmeshing_ai.mcp_client.models import ToolMetadata, ToolResult
from gearmeshing_ai.mcp_client.strategy.gateway import GatewayStrategy


def _mock_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:  # type: ignore[override]
        if request.method == "GET" and request.url.path.endswith("/mcp/"):
            data: Dict[str, Any] = {
                "tools": [
                    {"name": "echo", "description": "Echo tool", "parameters": {"text": {"type": "string"}}}
                ]
            }
            return httpx.Response(200, json=data)
        if request.method == "POST" and request.url.path.endswith("/mcp/invoke/echo"):
            body: Dict[str, Any] = {}
            if request.content:
                try:
                    body = _json.loads(request.content.decode("utf-8"))
                except Exception:
                    body = {}
            data = {"ok": True, "result": "ok", "echo": body.get("text")}
            return httpx.Response(200, json=data)
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


def test_gateway_list_tools_and_call_tool() -> None:
    transport: httpx.MockTransport = _mock_transport()
    client = httpx.Client(transport=transport, base_url="http://mock")
    cfg: MCPConfig = MCPConfig(base_url="http://mock", timeout=5)
    strat: GatewayStrategy = GatewayStrategy(cfg, client=client)

    tools: List[ToolMetadata] = list(strat.list_tools())
    assert len(tools) == 1
    assert tools[0].name == "echo"

    res: ToolResult = strat.call_tool("echo", {"text": "hi"})
    assert res.ok is True
    assert isinstance(res.data, dict)
    assert res.data.get("echo") == "hi"
