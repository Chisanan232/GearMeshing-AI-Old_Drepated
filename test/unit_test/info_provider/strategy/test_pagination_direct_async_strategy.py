from __future__ import annotations

from typing import Any, Dict, List

import httpx
import pytest

from gearmeshing_ai.info_provider.mcp.schemas.config import ServerConfig
from gearmeshing_ai.info_provider.mcp.strategy.direct_async import AsyncDirectMcpStrategy


def _mock_transport(state: dict) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        # First page (no cursor param present)
        if request.method == "GET" and request.url.path == "/tools" and request.url.params.get("cursor") is None:
            tools: List[Dict[str, Any]] = [
                {"name": "t1", "description": "one", "inputSchema": {"type": "object"}},
                {"name": "t2", "description": "two", "inputSchema": {"type": "object"}},
            ]
            return httpx.Response(200, json={"tools": tools, "nextCursor": "cursor-2"})

        # Second page
        if request.method == "GET" and request.url.path == "/tools" and request.url.params.get("cursor") == "cursor-2":
            tools2: List[Dict[str, Any]] = [
                {"name": "t3", "description": "three", "inputSchema": {"type": "object"}},
            ]
            return httpx.Response(200, json={"tools": tools2, "nextCursor": None})

        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_async_direct_strategy_list_tools_page_pagination() -> None:
    state: dict = {}
    transport = _mock_transport(state)

    http_client = httpx.AsyncClient(transport=transport, base_url="http://mock")

    servers = [ServerConfig(name="s1", endpoint_url="http://mock", auth_token=None)]
    strategy = AsyncDirectMcpStrategy(servers, client=http_client, ttl_seconds=60.0)

    page1 = await strategy.list_tools_page("s1", limit=2)
    assert [t.name for t in page1.items] == ["t1", "t2"]
    assert page1.next_cursor == "cursor-2"

    page2 = await strategy.list_tools_page("s1", cursor=page1.next_cursor, limit=2)
    assert [t.name for t in page2.items] == ["t3"]
    assert page2.next_cursor is None

    await http_client.aclose()
