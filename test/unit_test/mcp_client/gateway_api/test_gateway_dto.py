from __future__ import annotations

import httpx

from gearmeshing_ai.mcp_client.gateway_api.client import GatewayApiClient
from gearmeshing_ai.mcp_client.gateway_api.models import GatewayTransport


def _mock_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/servers":
            data = [
                {
                    "id": "s1",
                    "name": "srv1",
                    "url": "http://underlying/mcp/",
                    "transport": "STREAMABLEHTTP",
                    "description": "Test server",
                    "tags": ["prod", "search"],
                    "visibility": "team",
                    "teamId": "team-123",
                    "isActive": True,
                    "metrics": {"uptime": 12345},
                }
            ]
            return httpx.Response(200, json=data)
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


def test_gateway_dto_extended_fields() -> None:
    transport = _mock_transport()
    client = httpx.Client(transport=transport, base_url="http://mock")

    gw = GatewayApiClient("http://mock", client=client)
    servers = gw.list_servers()
    assert len(servers) == 1
    s = servers[0]
    assert s.id == "s1"
    assert s.transport == GatewayTransport.STREAMABLE_HTTP
    assert s.description == "Test server"
    assert s.tags == ["prod", "search"]
    assert s.visibility == "team"
    assert s.team_id == "team-123"
    assert s.is_active is True
    assert isinstance(s.metrics, dict) and s.metrics.get("uptime") == 12345
