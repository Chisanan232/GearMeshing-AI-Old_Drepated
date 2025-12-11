from __future__ import annotations

import json as _json
from typing import Any, Dict, List

import httpx

from gearmeshing_ai.info_provider.mcp.gateway_api.client import GatewayApiClient
from gearmeshing_ai.info_provider.mcp.strategy.gateway import GatewayMcpStrategy


def _mock_transport(state: dict) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        # Management API: list servers
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

        # Streamable HTTP endpoints: tools
        if request.method == "GET" and request.url.path == "/servers/s1/mcp/tools":
            state["tools_get_count"] = state.get("tools_get_count", 0) + 1
            # Check Authorization header propagation
            assert request.headers.get("Authorization") == state.get("expected_auth")
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

        # Streamable HTTP endpoints: invoke
        if request.method == "POST" and request.url.path == "/servers/s1/mcp/a2a/echo/invoke":
            # Check Authorization header propagation
            assert request.headers.get("Authorization") == state.get("expected_auth")
            try:
                body = _json.loads(request.content.decode("utf-8")) if request.content else {}
            except Exception:
                body = {}
            params = (body or {}).get("parameters") or {}
            return httpx.Response(200, json={"ok": True, "echo": params.get("text")})

        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


def test_gateway_strategy_cache_and_auth() -> None:
    state: dict = {"expected_auth": "Bearer abc"}
    transport = _mock_transport(state)

    mgmt_client = httpx.Client(transport=transport, base_url="http://mock")
    http_client = httpx.Client(transport=transport, base_url="http://mock")

    gateway = GatewayApiClient("http://mock", auth_token=state["expected_auth"], client=mgmt_client)
    strategy = GatewayMcpStrategy(gateway, client=http_client, ttl_seconds=60.0)

    # First list_tools should hit the network
    tools1 = list(strategy.list_tools("s1"))
    assert len(tools1) == 1 and tools1[0].name == "echo"
    assert state.get("tools_get_count", 0) == 1

    # Second list_tools should be served from cache (no additional GET)
    tools2 = list(strategy.list_tools("s1"))
    assert len(tools2) == 1 and tools2[0].name == "echo"
    assert state.get("tools_get_count", 0) == 1

    # Call tool should include Authorization header
    res = strategy.call_tool("s1", "echo", {"text": "hi"})
    assert res.ok is True
    assert res.data.get("echo") == "hi"
