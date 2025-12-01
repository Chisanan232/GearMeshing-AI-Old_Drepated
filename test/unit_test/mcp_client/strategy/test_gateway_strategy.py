from __future__ import annotations

import json as _json
from typing import Any, Dict, List

import httpx

from gearmeshing_ai.mcp_client.gateway_api.client import GatewayApiClient
from gearmeshing_ai.mcp_client.strategy.gateway import GatewayMcpStrategy


def _mock_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        # Management API
        if request.method == "GET" and request.url.path == "/servers":
            data = [
                {
                    "id": "s1",
                    "name": "gateway-s1",
                    "url": "http://underlying/mcp/",
                    "transport": "STREAMABLEHTTP",
                }
            ]
            return httpx.Response(200, json=data)

        # Streamable HTTP endpoints under the Gateway
        if request.method == "GET" and request.url.path == "/servers/s1/mcp/tools":
            tools: List[Dict[str, Any]] = [
                {
                    "name": "echo",
                    "description": "Echo tool",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"text": {"type": "string", "description": "Text to echo"}},
                        "required": ["text"],
                    },
                }
            ]
            return httpx.Response(200, json=tools)

        if request.method == "POST" and request.url.path == "/servers/s1/mcp/a2a/echo/invoke":
            body: Dict[str, Any] = {}
            if request.content:
                try:
                    body = _json.loads(request.content.decode("utf-8"))
                except Exception:
                    body = {}
            params = (body or {}).get("parameters") or {}
            return httpx.Response(200, json={"ok": True, "echo": params.get("text")})

        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


def test_gateway_strategy_list_and_call() -> None:
    transport = _mock_transport()

    mgmt_client = httpx.Client(transport=transport, base_url="http://mock")
    http_client = httpx.Client(transport=transport, base_url="http://mock")

    gateway = GatewayApiClient("http://mock", client=mgmt_client)
    strategy = GatewayMcpStrategy(gateway, client=http_client)

    servers = list(strategy.list_servers())
    assert len(servers) == 1
    assert servers[0].id == "s1"
    assert str(servers[0].endpoint_url).endswith("/servers/s1/mcp/")

    tools = list(strategy.list_tools("s1"))
    assert len(tools) == 1
    assert tools[0].name == "echo"
    assert tools[0].raw_parameters_schema["properties"]["text"]["type"] == "string"

    res = strategy.call_tool("s1", "echo", {"text": "hi"})
    assert res.ok is True
    assert res.data.get("echo") == "hi"
