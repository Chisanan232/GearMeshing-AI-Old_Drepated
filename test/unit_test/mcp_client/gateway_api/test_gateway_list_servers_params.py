from __future__ import annotations
import httpx

from gearmeshing_ai.mcp_client.gateway_api.client import GatewayApiClient


def _mock_transport(expected_path: str, expected_query: dict[str, str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:  # type: ignore[override]
        assert request.method == "GET"
        assert request.url.path == expected_path
        # Verify query params
        for k, v in expected_query.items():
            assert request.url.params.get(k) == v
        # Return empty list
        return httpx.Response(200, json=[])

    return httpx.MockTransport(handler)


def test_list_servers_with_query_params() -> None:
    expected_query = {
        "include_inactive": "true",
        "tags": "a,b",
        "team_id": "team-1",
        "visibility": "team",
    }
    transport = _mock_transport("/servers", expected_query)
    client = httpx.Client(transport=transport, base_url="http://mock")

    gw = GatewayApiClient("http://mock", client=client)
    res = gw.list_servers(include_inactive=True, tags="a,b", team_id="team-1", visibility="team")
    assert res == []
