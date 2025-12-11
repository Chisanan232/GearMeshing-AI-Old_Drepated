from __future__ import annotations

import json as _json

import httpx
import pytest

from gearmeshing_ai.info_provider.mcp.gateway_api.client import GatewayApiClient
from gearmeshing_ai.info_provider.mcp.gateway_api.errors import GatewayServerNotFoundError
from gearmeshing_ai.info_provider.mcp.gateway_api.models.domain import (
    GatewayServerCreate,
    GatewayTransport,
)


def _mock_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/servers":
            data = [
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
            return httpx.Response(200, json=data)
        if request.method == "GET" and request.url.path == "/servers/s1":
            return httpx.Response(
                200,
                json={
                    "id": "s1",
                    "name": "srv1",
                    "url": "http://underlying/mcp/",
                    "transport": "STREAMABLEHTTP",
                },
            )
        if request.method == "GET" and request.url.path == "/servers/missing":
            return httpx.Response(404, json={"error": "not found"})
        if request.method == "POST" and request.url.path == "/servers":
            try:
                body = _json.loads(request.content.decode("utf-8")) if request.content else {}
            except Exception:
                body = {}
            created = {
                "id": "created-123",
                "name": body.get("name", "new"),
                "url": body.get("url", "http://x/mcp/"),
                "transport": body.get("transport", "STREAMABLEHTTP"),
            }
            return httpx.Response(200, json=created)
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


def test_list_and_get_and_create_server() -> None:
    transport = _mock_transport()
    client = httpx.Client(transport=transport, base_url="http://mock")

    gw = GatewayApiClient("http://mock", client=client)

    servers = gw.list_servers()
    assert [s.id for s in servers] == ["s1", "s2"]
    assert servers[0].transport == GatewayTransport.STREAMABLE_HTTP

    s1 = gw.get_server("s1")
    assert s1.name == "srv1"

    with pytest.raises(GatewayServerNotFoundError):
        gw.get_server("missing")

    created = gw.create_server(
        GatewayServerCreate(name="new-srv", url="http://new/mcp/", transport=GatewayTransport.STREAMABLE_HTTP)
    )
    assert created.id == "created-123"
    assert created.name == "new-srv"
    assert str(created.url).endswith("/mcp/")
