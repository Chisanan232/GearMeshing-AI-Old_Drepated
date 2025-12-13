from __future__ import annotations

import httpx
import pytest

from gearmeshing_ai.info_provider.mcp.gateway_api.client import GatewayApiClient


def _mock_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/admin/mcp-registry/servers":
            data = {
                "items": [
                    {
                        "id": "s1",
                        "name": "srv1",
                        "url": "http://underlying/mcp/",
                        "transport": "STREAMABLEHTTP",
                    },
                    {
                        "id": "s2",
                        "name": "srv2",
                        "url": "http://underlying2/mcp/",
                        "transport": "SSE",
                    },
                ]
            }
            return httpx.Response(200, json=data)
        if request.method == "POST" and request.url.path == "/admin/mcp-registry/s1/register":
            return httpx.Response(200, json={"ok": True, "id": "s1"})
        if request.method == "GET" and request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


def test_admin_registry_list_register_and_health() -> None:
    transport = _mock_transport()
    client = httpx.Client(transport=transport, base_url="http://mock")

    gw = GatewayApiClient("http://mock", client=client)

    # List catalog servers via admin namespace
    payload = gw.admin.mcp_registry.list()
    assert isinstance(payload, dict)
    assert [item["id"] for item in payload.get("items", [])] == ["s1", "s2"]

    # Register one catalog server
    res = gw.admin.mcp_registry.register("s1")
    assert res.get("ok") is True and res.get("id") == "s1"

    # Health check
    h = gw.health()
    assert h.get("status") == "ok"
