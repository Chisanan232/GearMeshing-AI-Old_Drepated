from __future__ import annotations

import httpx

from gearmeshing_ai.info_provider.mcp.gateway_api.client import GatewayApiClient
from gearmeshing_ai.info_provider.mcp.gateway_api.models.dto import CatalogListResponseDTO


def _mock_transport(expected_path: str, expected_query: dict[str, str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == expected_path
        # Verify query params
        for k, v in expected_query.items():
            assert request.url.params.get(k) == v
        # Return empty catalog payload with required fields
        return httpx.Response(
            200,
            json={
                "servers": [],
                "total": 0,
                "categories": [],
                "auth_types": [],
                "providers": [],
                "all_tags": [],
            },
        )

    return httpx.MockTransport(handler)


def test_list_servers_with_query_params() -> None:
    expected_query = {
        "include_inactive": "true",
        "tags": "a,b",
        "team_id": "team-1",
        "visibility": "team",
    }
    transport = _mock_transport("/admin/mcp-registry/servers", expected_query)
    client = httpx.Client(transport=transport, base_url="http://mock")

    gw = GatewayApiClient("http://mock", client=client)
    payload = gw.admin.mcp_registry.list(
        include_inactive=True, tags="a,b", team_id="team-1", visibility="team"
    )
    assert isinstance(payload, CatalogListResponseDTO)
    assert payload.servers == [] and payload.total == 0
