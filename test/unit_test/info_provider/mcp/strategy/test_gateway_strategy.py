from __future__ import annotations

import json as _json
from typing import Any, Dict, List
from contextlib import asynccontextmanager

import httpx

from gearmeshing_ai.info_provider.mcp.gateway_api.client import GatewayApiClient
from gearmeshing_ai.info_provider.mcp.strategy.gateway import GatewayMcpStrategy
from gearmeshing_ai.info_provider.mcp.transport.mcp import AsyncMCPTransport


def _mock_transport(state: dict) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        # Management API: list gateways
        if request.method == "GET" and request.url.path == "/admin/gateways":
            return httpx.Response(200, json=[{"id": "g1", "name": "gw", "url": "http://mock/mcp"}])
        # Management API: get gateway
        if request.method == "GET" and request.url.path == "/admin/gateways/g1":
            return httpx.Response(200, json={"id": "g1", "name": "gw", "url": "http://mock/mcp"})
        # Management API: tools list
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
                        "metrics": {"totalExecutions": 0, "successfulExecutions": 0, "failedExecutions": 0, "failureRate": 0.0},
                        "gatewayId": "g1",
                    }
                ]
            }
            return httpx.Response(200, json=data)

        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


def test_gateway_strategy_list_and_call() -> None:
    state: dict = {}
    transport = _mock_transport(state)

    mgmt_client = httpx.Client(transport=transport, base_url="http://mock")
    http_client = httpx.Client(transport=transport, base_url="http://mock")

    # Fake MCP transport to capture calls
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

    gateway = GatewayApiClient("http://mock", client=mgmt_client)
    strategy = GatewayMcpStrategy(gateway, client=http_client, mcp_transport=FakeTransport())

    servers = list(strategy.list_servers())
    assert len(servers) == 1
    assert servers[0].id == "g1"
    assert str(servers[0].endpoint_url) == "http://mock/mcp"

    tools = list(strategy.list_tools("g1"))
    assert len(tools) == 1
    assert tools[0].name == "workspace.list"
    assert tools[0].raw_parameters_schema["properties"]["text"]["type"] == "string"

    res = strategy.call_tool("g1", "echo", {"text": "hi"})
    assert res.ok is True
    assert res.data.get("echo") == "hi"
    assert captured["endpoint"] == "http://mock/mcp"
