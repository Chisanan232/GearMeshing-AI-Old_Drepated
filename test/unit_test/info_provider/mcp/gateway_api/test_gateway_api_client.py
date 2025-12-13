from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import httpx
import pytest

from gearmeshing_ai.info_provider.mcp.gateway_api.client import GatewayApiClient
from gearmeshing_ai.info_provider.mcp.gateway_api.models.dto import (
    AdminToolsListResponseDTO,
    CatalogListResponseDTO,
    CatalogServerRegisterResponseDTO,
    GatewayReadDTO,
    ToolReadDTO,
)

def _sample_tools_payload() -> dict:
    now = datetime.now(timezone.utc).isoformat()
    def tool(tool_id: str, name: str, custom: str) -> dict:
        return {
            "id": tool_id,
            "originalName": custom,
            "requestType": "SSE",
            "integrationType": "MCP",
            "inputSchema": {"type": "object", "properties": {}},
            "createdAt": now,
            "updatedAt": now,
            "enabled": True,
            "reachable": True,
            "executionCount": 0,
            "metrics": {
                "totalExecutions": 0,
                "successfulExecutions": 0,
                "failedExecutions": 0,
                "failureRate": 0.0,
            },
            "name": f"{name}",
            "gatewaySlug": "gw",
            "customName": custom,
            "customNameSlug": custom.replace(".", "-"),
        }
    return {
        "data": [
            tool("t1", "tool-one", "workspace.list"),
            tool("t2", "tool-two", "get_authorized_teams"),
        ],
        "pagination": {"page": 1, "per_page": 50, "total_items": 2, "total_pages": 1},
        "links": {"self": "/admin/tools?page=1&per_page=50", "first": "...", "last": "...", "next": None, "prev": None},
    }


def _mock_transport() -> httpx.MockTransport:
    tools_payload = _sample_tools_payload()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/admin/mcp-registry/servers":
            data = {
                "servers": [
                    {
                        "id": "s1",
                        "name": "srv1",
                        "category": "Utilities",
                        "url": "http://underlying/mcp/",
                        "auth_type": "Open",
                        "provider": "E2E",
                        "description": "desc",
                        "transport": "SSE",
                    },
                    {
                        "id": "s2",
                        "name": "srv2",
                        "category": "Utilities",
                        "url": "http://underlying2/mcp/",
                        "auth_type": "Open",
                        "provider": "E2E",
                        "description": "desc",
                        "transport": "STREAMABLEHTTP",
                    },
                ],
                "total": 2,
                "categories": ["Utilities"],
                "auth_types": ["Open"],
                "providers": ["E2E"],
                "all_tags": ["project-management", "python"],
            }
            return httpx.Response(200, json=data)
        if request.method == "POST" and request.url.path == "/admin/mcp-registry/s1/register":
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "server_id": "s1",
                    "message": "Registered s1",
                    "error": None,
                },
            )
        if request.method == "GET" and request.url.path == "/admin/gateways":
            return httpx.Response(200, json=[{"id": "g1", "name": "gw", "url": "http://mock"}])
        if request.method == "GET" and request.url.path == "/admin/gateways/g1":
            return httpx.Response(200, json={"id": "g1", "name": "gw", "url": "http://mock"})
        if request.method == "GET" and request.url.path == "/admin/tools":
            return httpx.Response(200, json=tools_payload)
        if request.method == "GET" and request.url.path.startswith("/admin/tools/"):
            tool_id = request.url.path.rsplit("/", 1)[-1]
            for item in tools_payload.get("data", []):
                if item.get("id") == tool_id:
                    return httpx.Response(200, json=item)
            return httpx.Response(404, json={"error": "not found"})
        if request.method == "GET" and request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)

@pytest.fixture()
def gw_client() -> GatewayApiClient:
    transport = _mock_transport()
    client = httpx.Client(transport=transport, base_url="http://mock")
    return GatewayApiClient("http://mock", client=client)


def test_admin_mcp_registry_list_returns_catalog_dto(gw_client: GatewayApiClient) -> None:
    catalog: CatalogListResponseDTO = gw_client.admin.mcp_registry.list()
    assert catalog.total == 2
    assert [s.id for s in catalog.servers] == ["s1", "s2"]


def test_admin_mcp_registry_register_returns_dto(gw_client: GatewayApiClient) -> None:
    reg: CatalogServerRegisterResponseDTO = gw_client.admin.mcp_registry.register("s1")
    assert reg.success is True and reg.server_id == "s1"


def test_admin_gateway_list_and_get(gw_client: GatewayApiClient) -> None:
    gws = gw_client.admin.gateway.list()
    assert isinstance(gws, list) and isinstance(gws[0], GatewayReadDTO)
    gw1 = gw_client.admin.gateway.get("g1")
    assert isinstance(gw1, GatewayReadDTO) and gw1.id == "g1"


def test_admin_tools_list_and_get(gw_client: GatewayApiClient) -> None:
    tools_list = gw_client.admin.tools.list()
    assert isinstance(tools_list, AdminToolsListResponseDTO)
    assert tools_list.data and isinstance(tools_list.data[0], ToolReadDTO)
    first_id = tools_list.data[0].id
    tool = gw_client.admin.tools.get(first_id)
    assert isinstance(tool, ToolReadDTO) and tool.id == first_id


def test_gateway_health_ok(gw_client: GatewayApiClient) -> None:
    h = gw_client.health()
    assert h.get("status") == "ok"


def test_gateway_list_accepts_list_or_items_and_include_inactive_query() -> None:
    # First response as list
    def handler1(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET" and request.url.path == "/admin/gateways"
        assert request.url.params.get("include_inactive") == "true"
        return httpx.Response(200, json=[{"id": "g1", "name": "gw", "url": "http://mock"}])

    client1 = httpx.Client(transport=httpx.MockTransport(handler1), base_url="http://mock")
    gw1 = GatewayApiClient("http://mock", client=client1)
    lst1 = gw1.admin.gateway.list(include_inactive=True)
    assert isinstance(lst1[0], GatewayReadDTO)

    # Second response as {items: [...]}
    def handler2(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET" and request.url.path == "/admin/gateways"
        assert request.url.params.get("include_inactive") == "false"
        return httpx.Response(200, json={"items": [{"id": "g2", "name": "gw2", "url": "http://mock2"}]})

    client2 = httpx.Client(transport=httpx.MockTransport(handler2), base_url="http://mock")
    gw2 = GatewayApiClient("http://mock", client=client2)
    lst2 = gw2.admin.gateway.list(include_inactive=False)
    assert isinstance(lst2[0], GatewayReadDTO) and lst2[0].id == "g2"


def test_tools_list_include_inactive_query_sets_boolean() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET" and request.url.path == "/admin/tools"
        assert request.url.params.get("include_inactive") == "true"
        return httpx.Response(200, json={"data": _sample_tools_payload()["data"]})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock")
    gw = GatewayApiClient("http://mock", client=client)
    res = gw.admin.tools.list(include_inactive=True)
    assert isinstance(res, AdminToolsListResponseDTO)


def test_health_error_raises_gateway_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(500, json={"error": "down"})
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock")
    gw = GatewayApiClient("http://mock", client=client)
    import pytest as _pytest
    from gearmeshing_ai.info_provider.mcp.gateway_api.errors import GatewayApiError

    with _pytest.raises(GatewayApiError) as ei:
        gw.health()
    assert ei.value.status_code == 500


def test_health_text_fallback_when_json_decode_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            # Return plain text body; json() will fail
            return httpx.Response(200, content=b"ok-text", headers={"Content-Type": "text/plain"})
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock")
    gw = GatewayApiClient("http://mock", client=client)
    resp = gw.health()
    assert resp == {"status": "ok-text"}
