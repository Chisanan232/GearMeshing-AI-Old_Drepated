from __future__ import annotations

import httpx

from gearmeshing_ai.info_provider.gateway_api.client import GatewayApiClient
from gearmeshing_ai.info_provider.strategy.gateway import GatewayMcpStrategy


def _mock_transport_paginated() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        # Expect paths like /servers/s1/mcp/tools with optional query params
        if request.method == "GET" and request.url.path == "/servers/s1/mcp/tools":
            cursor = request.url.params.get("cursor")
            if cursor is None:
                # first page
                data = {
                    "tools": [
                        {"name": "tool_a", "description": "A", "inputSchema": {"type": "object"}},
                        {"name": "tool_b", "description": "B", "inputSchema": {"type": "object"}},
                    ],
                    "nextCursor": "c1",
                }
                return httpx.Response(200, json=data)
            if cursor == "c1":
                data = {
                    "tools": [
                        {"name": "tool_c", "description": "C", "inputSchema": {"type": "object"}},
                    ]
                }
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
    assert page1.next_cursor == "c1"

    page2 = strat.list_tools_page("s1", cursor=page1.next_cursor, limit=2)
    assert [t.name for t in page2.items] == ["tool_c"]
    assert page2.next_cursor is None
