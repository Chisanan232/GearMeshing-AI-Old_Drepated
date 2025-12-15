from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict

import httpx

from gearmeshing_ai.info_provider.mcp.gateway_api.client import GatewayApiClient
from gearmeshing_ai.info_provider.mcp.strategy.gateway import GatewayMcpStrategy
from gearmeshing_ai.info_provider.mcp.transport import AsyncMCPTransport


def _mock_transport(state: dict) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        # Management API: list gateways
        if request.method == "GET" and request.url.path == "/admin/gateways":
            return httpx.Response(200, json=[{"id": "g1", "name": "gw", "url": "http://mock/mcp"}])
        # Management API: get gateway
        if request.method == "GET" and request.url.path == "/admin/gateways/g1":
            return httpx.Response(200, json={"id": "g1", "name": "gw", "url": "http://mock/mcp"})
        # Admin tools list
        if request.method == "GET" and request.url.path == "/admin/tools":
            state["tools_get_count"] = state.get("tools_get_count", 0) + 1
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


def test_gateway_strategy_cache_and_auth() -> None:
    state: dict = {"expected_auth": "Bearer abc"}
    transport = _mock_transport(state)

    mgmt_client = httpx.Client(transport=transport, base_url="http://mock")
    http_client = httpx.Client(transport=transport, base_url="http://mock")

    gateway = GatewayApiClient("http://mock", auth_token=state["expected_auth"], client=mgmt_client)

    # Fake MCP transport that checks endpoint and returns echo
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

    strategy = GatewayMcpStrategy(gateway, client=http_client, ttl_seconds=60.0, mcp_transport=FakeTransport())

    # First list_tools should hit the network
    tools1 = list(strategy.list_tools("g1"))
    assert len(tools1) == 1 and tools1[0].name == "workspace.list"
    assert state.get("tools_get_count", 0) == 1

    # Second list_tools should be served from cache (no additional GET)
    tools2 = list(strategy.list_tools("g1"))
    assert len(tools2) == 1 and tools2[0].name == "workspace.list"
    assert state.get("tools_get_count", 0) == 1

    # Call tool should route via MCP transport
    res = strategy.call_tool("g1", "echo", {"text": "hi"})
    assert res.ok is True
    assert res.data.get("echo") == "hi"
    assert captured["endpoint"] == "http://mock/mcp"
