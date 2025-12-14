from __future__ import annotations

import httpx

from gearmeshing_ai.info_provider.mcp.gateway_api.client import GatewayApiClient
from gearmeshing_ai.info_provider.mcp.strategy.gateway import GatewayMcpStrategy


def _mock_transport_paginated() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        # Strategy now calls admin.tools.list with offset/limit pagination
        if request.method == "GET" and request.url.path == "/admin/tools":
            # Map cursor to offset
            offset = int(request.url.params.get("offset", "0"))
            limit = int(request.url.params.get("limit", "50"))
            all_tools = [
                {
                    "id": "t_a",
                    "originalName": "tool_a",
                    "name": "tool-a",
                    "customName": "tool_a",
                    "customNameSlug": "tool-a",
                    "gatewaySlug": "gw",
                    "requestType": "SSE",
                    "integrationType": "MCP",
                    "inputSchema": {"type": "object"},
                    "createdAt": "2024-01-01T00:00:00Z",
                    "updatedAt": "2024-01-01T00:00:00Z",
                    "enabled": True,
                    "reachable": True,
                    "executionCount": 0,
                    "metrics": {"totalExecutions": 0, "successfulExecutions": 0, "failedExecutions": 0, "failureRate": 0.0},
                },
                {
                    "id": "t_b",
                    "originalName": "tool_b",
                    "name": "tool-b",
                    "customName": "tool_b",
                    "customNameSlug": "tool-b",
                    "gatewaySlug": "gw",
                    "requestType": "SSE",
                    "integrationType": "MCP",
                    "inputSchema": {"type": "object"},
                    "createdAt": "2024-01-01T00:00:00Z",
                    "updatedAt": "2024-01-01T00:00:00Z",
                    "enabled": True,
                    "reachable": True,
                    "executionCount": 0,
                    "metrics": {"totalExecutions": 0, "successfulExecutions": 0, "failedExecutions": 0, "failureRate": 0.0},
                },
                {
                    "id": "t_c",
                    "originalName": "tool_c",
                    "name": "tool-c",
                    "customName": "tool_c",
                    "customNameSlug": "tool-c",
                    "gatewaySlug": "gw",
                    "requestType": "SSE",
                    "integrationType": "MCP",
                    "inputSchema": {"type": "object"},
                    "createdAt": "2024-01-01T00:00:00Z",
                    "updatedAt": "2024-01-01T00:00:00Z",
                    "enabled": True,
                    "reachable": True,
                    "executionCount": 0,
                    "metrics": {"totalExecutions": 0, "successfulExecutions": 0, "failedExecutions": 0, "failureRate": 0.0},
                },
            ]
            page_items = all_tools[offset: offset + limit]
            page_num = (offset // limit) + 1 if limit else 1
            total = len(all_tools)
            data = {"data": page_items, "pagination": {"page": page_num, "perPage": limit, "total": total}}
            return httpx.Response(200, json=data)
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


def test_gateway_strategy_list_tools_page_two_pages() -> None:
    transport = _mock_transport_paginated()
    http_client = httpx.Client(transport=transport, base_url="http://mock")

    gw = GatewayApiClient("http://mock", client=http_client)
    strat = GatewayMcpStrategy(gw, client=http_client)

    page1 = strat.list_tools_page("s1", limit=2)
    assert [t.name for t in page1.items] == ["tool_a", "tool_b"]
    assert page1.next_cursor == "2"

    page2 = strat.list_tools_page("s1", cursor=page1.next_cursor, limit=2)
    assert [t.name for t in page2.items] == ["tool_c"]
    assert page2.next_cursor is None
