from __future__ import annotations

from typing import Any, Dict, List
import json as _json

import httpx
import pytest

from gearmeshing_ai.mcp_client.gateway_api.client import GatewayApiClient
from gearmeshing_ai.mcp_client.strategy.gateway_async import AsyncGatewayMcpStrategy


def _mock_transport(state: dict) -> httpx.MockTransport:
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
            state["tools_get_count"] = state.get("tools_get_count", 0) + 1
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

        if request.method == "POST" and request.url.path == "/servers/s1/mcp/a2a/echo/invoke":
            assert request.headers.get("Authorization") == state.get("expected_auth")
            try:
                body = _json.loads(request.content.decode("utf-8")) if request.content else {}
            except Exception:
                body = {}
            params = (body or {}).get("parameters") or {}
            return httpx.Response(200, json={"ok": True, "echo": params.get("text")})

        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_async_gateway_strategy_cache_and_auth() -> None:
    state: dict = {"expected_auth": "Bearer xyz"}
    transport = _mock_transport(state)

    mgmt_client = httpx.Client(transport=transport, base_url="http://mock")
    http_client = httpx.AsyncClient(transport=transport, base_url="http://mock")

    gateway = GatewayApiClient("http://mock", auth_token=state["expected_auth"], client=mgmt_client)
    strategy = AsyncGatewayMcpStrategy(gateway, client=http_client, ttl_seconds=60.0)

    tools1 = await strategy.list_tools("s1")
    assert len(tools1) == 1 and tools1[0].name == "echo"
    assert state.get("tools_get_count", 0) == 1

    tools2 = await strategy.list_tools("s1")
    assert len(tools2) == 1 and tools2[0].name == "echo"
    assert state.get("tools_get_count", 0) == 1

    res = await strategy.call_tool("s1", "echo", {"text": "hi"})
    assert res.ok is True
    assert res.data.get("echo") == "hi"

    await http_client.aclose()
    mgmt_client.close()


@pytest.mark.asyncio
async def test_async_gateway_strategy_ttl_zero_no_cache() -> None:
    state: dict = {"expected_auth": "Bearer xyz"}
    transport = _mock_transport(state)

    mgmt_client = httpx.Client(transport=transport, base_url="http://mock")
    http_client = httpx.AsyncClient(transport=transport, base_url="http://mock")

    gateway = GatewayApiClient("http://mock", auth_token=state["expected_auth"], client=mgmt_client)
    strategy = AsyncGatewayMcpStrategy(gateway, client=http_client, ttl_seconds=0.0)

    tools1 = await strategy.list_tools("s1")
    assert len(tools1) == 1 and tools1[0].name == "echo"
    assert state.get("tools_get_count", 0) == 1

    tools2 = await strategy.list_tools("s1")
    assert len(tools2) == 1 and tools2[0].name == "echo"
    assert state.get("tools_get_count", 0) == 2

    await http_client.aclose()
    mgmt_client.close()
