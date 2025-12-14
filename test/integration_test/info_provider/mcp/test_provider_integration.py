from __future__ import annotations

from typing import Any, Dict, List

import httpx

from gearmeshing_ai.info_provider.mcp.gateway_api.client import GatewayApiClient
from gearmeshing_ai.info_provider.mcp.provider import MCPInfoProvider
from gearmeshing_ai.info_provider.mcp.schemas.config import (
    GatewayConfig,
    McpClientConfig,
    ServerConfig,
)
from gearmeshing_ai.info_provider.mcp.strategy import (
    DirectMcpStrategy,
    GatewayMcpStrategy,
)


def _mock_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        # Gateway management API
        if request.method == "GET" and request.url.path == "/admin/tools":
            data = {
                "data": [
                    {
                        "id": "t_g_echo",
                        "originalName": "g_echo",
                        "name": "g-echo",
                        "gatewaySlug": "gw",
                        "customName": "g_echo",
                        "customNameSlug": "g-echo",
                        "requestType": "SSE",
                        "integrationType": "MCP",
                        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
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

        # Gateway streamable HTTP endpoints
        if request.method == "GET" and request.url.path == "/servers/s1/mcp/tools":
            tools: List[Dict[str, Any]] = [
                {
                    "name": "g_echo",
                    "description": "Gateway Echo",
                    "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
                }
            ]
            return httpx.Response(200, json=tools)
        if request.method == "POST" and request.url.path == "/servers/s1/mcp/a2a/g_echo/invoke":
            return httpx.Response(200, json={"ok": True, "source": "gateway", "echo": True})

        # Direct server endpoints (expect endpoint_url http://mock/mcp)
        if request.method == "GET" and request.url.path == "/mcp/tools":
            tools: List[Dict[str, Any]] = [  # type: ignore[no-redef]
                {
                    "name": "d_echo",
                    "description": "Direct Echo",
                    "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
                }
            ]
            return httpx.Response(200, json=tools)
        if request.method == "POST" and request.url.path == "/mcp/a2a/d_echo/invoke":
            return httpx.Response(200, json={"ok": True, "source": "direct", "echo": True})

        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


def test_mcp_client_composed_strategies() -> None:
    transport = _mock_transport()
    # Reuse one transport for both mgmt and data endpoints for simplicity
    mgmt_client = httpx.Client(transport=transport, base_url="http://mock")
    http_client = httpx.Client(transport=transport, base_url="http://mock")

    cfg = McpClientConfig(
        gateway=GatewayConfig(base_url="http://mock"),
        servers=[ServerConfig(name="direct1", endpoint_url="http://mock/mcp")],
    )

    # Build Gateway strategy as before
    gw = GatewayApiClient("http://mock", client=mgmt_client)
    gw_strategy = GatewayMcpStrategy(gw, client=http_client)

    # Build Direct strategy with a fake MCP transport (no httpx)
    from contextlib import asynccontextmanager

    class _FakeTool:
        def __init__(self, name: str, description: str | None, input_schema: Dict[str, Any]) -> None:
            self.name = name
            self.description = description
            self.inputSchema = input_schema

    class _FakeListToolsResp:
        def __init__(self, tools: List[_FakeTool]) -> None:
            self.tools = tools
            self.next_cursor = None

    class _FakeSession:
        async def list_tools(self, cursor: str | None = None, limit: int | None = None):  # noqa: ARG002
            return _FakeListToolsResp(
                [_FakeTool("d_echo", "Direct Echo", {"type": "object", "properties": {"text": {"type": "string"}}})]
            )

        async def call_tool(self, name: str, arguments: Dict[str, Any] | None = None):  # noqa: ARG002
            return {"ok": True, "source": "direct", "echo": True}

    class _FakeMCPTransport:
        def session(self, endpoint_url: str):  # noqa: ARG002
            @asynccontextmanager
            async def _cm():
                yield _FakeSession()

            return _cm()

    direct_strategy = DirectMcpStrategy(
        [ServerConfig(name="direct1", endpoint_url="http://mock/mcp")],
        mcp_transport=_FakeMCPTransport(),
    )

    client = MCPInfoProvider(strategies=[direct_strategy, gw_strategy])

    tools_direct = {t.name for t in client.list_tools("direct1")}
    tools_gateway = {t.name for t in client.list_tools("s1")}
    assert tools_direct == {"d_echo"}
    assert tools_gateway == {"g_echo"}
