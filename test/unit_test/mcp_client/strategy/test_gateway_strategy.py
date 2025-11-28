from typing import List, Any, Dict
import json as _json
import httpx

from gearmeshing_ai.mcp_client.strategy.gateway import GatewayStrategy
from gearmeshing_ai.mcp_client.config import MCPConfig
from gearmeshing_ai.mcp_client.models import ToolMetadata, ToolResult


def _mock_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:  # type: ignore[override]
        if request.method == "GET" and request.url.path.endswith("/mcp/"):
            data: Dict[str, Any] = {"tools": []}
            return httpx.Response(200, json=data)
        if request.method == "POST" and request.url.path.endswith("/mcp/invoke/echo"):
            body: Dict[str, Any] = {}
            if request.content:
                try:
                    body = _json.loads(request.content.decode("utf-8"))
                except Exception:
                    body = {}
            return httpx.Response(200, json={"echo": body.get("text")})
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


def test_gateway_strategy_skeleton() -> None:
    transport: httpx.MockTransport = _mock_transport()
    client = httpx.Client(transport=transport, base_url="http://mock")
    s: GatewayStrategy = GatewayStrategy(MCPConfig(base_url="http://mock"), client=client)
    tools: List[ToolMetadata] = list(s.list_tools())
    assert tools == []
    res: ToolResult = s.call_tool("echo", {"text": "hi"})
    assert res.ok is True
