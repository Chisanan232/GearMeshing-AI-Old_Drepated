from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict

import httpx
import pytest

from gearmeshing_ai.info_provider.mcp.gateway_api.client import GatewayApiClient
from gearmeshing_ai.info_provider.mcp.strategy.gateway_async import (
    AsyncGatewayMcpStrategy,
)
from gearmeshing_ai.info_provider.mcp.transport.mcp import AsyncMCPTransport


def _mock_transport(state: dict) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        # Admin list gateways
        if request.method == "GET" and request.url.path == "/admin/gateways":
            return httpx.Response(200, json=[{"id": "g1", "name": "gw", "url": "http://mock/mcp"}])
        # Admin get gateway
        if request.method == "GET" and request.url.path == "/admin/gateways/g1":
            return httpx.Response(200, json={"id": "g1", "name": "gw", "url": "http://mock/mcp"})
        # Admin tools list
        if request.method == "GET" and request.url.path == "/admin/tools":
            state["tools_get_count"] = state.get("tools_get_count", 0) + 1
            assert request.headers.get("Authorization") == state.get("expected_auth")
            data = {
                "data": [
                    {
                        "id": "t1",
                        "originalName": "workspace.list",
                        "name": "clickup-workspace-list",
                        "gatewaySlug": "gw",
                        "customName": "workspace.list",
                        "customNameSlug": "workspace-list",
                        "requestType": "SSE",
                        "integrationType": "MCP",
                        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
                        "createdAt": "2024-01-01T00:00:00Z",
                        "updatedAt": "2024-01-01T00:00:00Z",
                        "enabled": True,
                        "reachable": True,
                        "executionCount": 0,
                        "metrics": {
                            "totalExecutions": 0,
                            "successfulExecutions": 0,
                            "failedExecutions": 0,
                            "failureRate": 0.0,
                        },
                        "gatewayId": "g1",
                    }
                ]
            }
            return httpx.Response(200, json=data)

        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_async_gateway_strategy_cache_and_auth() -> None:
    state: dict = {"expected_auth": "Bearer xyz"}
    transport = _mock_transport(state)

    mgmt_client = httpx.Client(transport=transport, base_url="http://mock")
    http_client = httpx.AsyncClient(transport=transport, base_url="http://mock")

    gateway = GatewayApiClient("http://mock", auth_token=state["expected_auth"], client=mgmt_client)

    # Fake MCP transport to simulate direct-style invocation
    captured: Dict[str, Any] = {}

    class FakeTransport(AsyncMCPTransport):
        def session(self, endpoint_url: str):
            @asynccontextmanager
            async def _cm():
                class _S:
                    async def call_tool(self, name: str, arguments: Dict[str, Any]):
                        captured["endpoint"] = endpoint_url
                        captured["name"] = name
                        captured["arguments"] = arguments
                        return {"ok": True, "echo": arguments.get("text")}

                yield _S()

            return _cm()

    strategy = AsyncGatewayMcpStrategy(gateway, client=http_client, ttl_seconds=60.0, mcp_transport=FakeTransport())

    tools1 = await strategy.list_tools("g1")
    assert len(tools1) == 1 and tools1[0].name == "workspace.list"
    assert state.get("tools_get_count", 0) == 1

    tools2 = await strategy.list_tools("g1")
    assert len(tools2) == 1 and tools2[0].name == "workspace.list"
    assert state.get("tools_get_count", 0) == 1

    res = await strategy.call_tool("g1", "echo", {"text": "hi"})
    assert res.ok is True
    assert res.data.get("echo") == "hi"
    assert captured["endpoint"] == "http://mock/mcp"

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

    tools1 = await strategy.list_tools("g1")
    assert len(tools1) == 1 and tools1[0].name == "workspace.list"
    assert state.get("tools_get_count", 0) == 1

    tools2 = await strategy.list_tools("g1")
    assert len(tools2) == 1 and tools2[0].name == "workspace.list"
    assert state.get("tools_get_count", 0) == 2

    await http_client.aclose()
    mgmt_client.close()
