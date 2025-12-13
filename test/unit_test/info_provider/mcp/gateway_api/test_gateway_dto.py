from __future__ import annotations

import httpx

from gearmeshing_ai.info_provider.mcp.gateway_api.client import GatewayApiClient
from gearmeshing_ai.info_provider.mcp.gateway_api.models.dto import ServersListPayloadDTO


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
                        "description": "Test server",
                        "tags": ["prod", "search"],
                        "visibility": "team",
                        "teamId": "team-123",
                        "isActive": True,
                        "metrics": {"uptime": 12345},
                    }
                ]
            }
            return httpx.Response(200, json=data)
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


def test_gateway_dto_extended_fields() -> None:
    transport = _mock_transport()
    client = httpx.Client(transport=transport, base_url="http://mock")

    gw = GatewayApiClient("http://mock", client=client)
    payload = gw.admin.mcp_registry.list()
    servers_list = ServersListPayloadDTO.model_validate(payload)
    assert len(servers_list.items) == 1
    dto = servers_list.items[0]
    assert dto.id == "s1"
    assert dto.transport == "STREAMABLEHTTP"
    assert dto.description == "Test server"
    assert dto.tags == ["prod", "search"]
    assert dto.visibility == "team"
    assert dto.team_id == "team-123"
    assert dto.is_active is True
    assert isinstance(dto.metrics, dict) and dto.metrics.get("uptime") == 12345
