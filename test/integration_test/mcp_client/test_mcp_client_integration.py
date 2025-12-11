from __future__ import annotations

from typing import Any, Dict, List

import httpx

from gearmeshing_ai.info_provider.client_sync import McpClient
from gearmeshing_ai.info_provider.schemas.config import (
    GatewayConfig,
    McpClientConfig,
    ServerConfig,
)


def _mock_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        # Gateway management API
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

    client = McpClient.from_config(
        cfg,
        direct_http_client=http_client,
        gateway_mgmt_client=mgmt_client,
        gateway_http_client=http_client,
    )

    servers = {s.id for s in client.list_servers()}
    assert servers == {"direct1", "s1"}

    tools_direct = {t.name for t in client.list_tools("direct1")}
    tools_gateway = {t.name for t in client.list_tools("s1")}
    assert tools_direct == {"d_echo"}
    assert tools_gateway == {"g_echo"}

    res_d = client.call_tool("direct1", "d_echo", {"text": "x"})
    res_g = client.call_tool("s1", "g_echo", {"text": "y"})
    assert res_d.ok is True and res_d.data.get("source") == "direct"
    assert res_g.ok is True and res_g.data.get("source") == "gateway"
