from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict

import httpx
import pytest

from gearmeshing_ai.info_provider.mcp.gateway_api.client import GatewayApiClient
from gearmeshing_ai.info_provider.mcp.strategy.gateway_async import (
    AsyncGatewayMcpStrategy,
)
from gearmeshing_ai.info_provider.mcp.transport import AsyncMCPTransport


def _mock_transport(state: dict) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        # Admin list gateways
        if request.method == "GET" and request.url.path == "/admin/gateways":
            return httpx.Response(200, json=[{"id": "g1", "name": "gw", "url": "http://mock/mcp"}])
        # Admin get gateway
        if request.method == "GET" and request.url.path == "/admin/gateways/g1":
            return httpx.Response(200, json={"id": "g1", "name": "gw", "url": "http://mock/mcp"})
        # Admin tools list
        if request.method == "GET" and request.url.path == "/admin/tools":
            state["tools_get_count"] = state.get("tools_get_count", 0) + 1
            assert request.headers.get("Authorization") == state.get("expected_auth")
            data = {
                "data": [
                    {
                        "id": "t1",
                        "originalName": "workspace.list",
                        "name": "clickup-workspace-list",
                        "gatewaySlug": "gw",
                        "customName": "workspace.list",
                        "customNameSlug": "workspace-list",
                        "requestType": "SSE",
                        "integrationType": "MCP",
                        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
                        "createdAt": "2024-01-01T00:00:00Z",
                        "updatedAt": "2024-01-01T00:00:00Z",
                        "enabled": True,
                        "reachable": True,
                        "executionCount": 0,
                        "metrics": {
                            "totalExecutions": 0,
                            "successfulExecutions": 0,
                            "failedExecutions": 0,
                            "failureRate": 0.0,
                        },
                        "gatewayId": "g1",
                    }
                ]
            }
            return httpx.Response(200, json=data)

        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


def _mock_transport_paginated(state: dict | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        # Strategy calls admin.tools.list with offset/limit pagination
        if request.method == "GET" and request.url.path == "/admin/tools":
            offset = int(request.url.params.get("offset", "0"))
            limit = int(request.url.params.get("limit", "50"))
            if state is not None:
                offsets: list[int] = state.setdefault("offsets", [])  # type: ignore[assignment]
                offsets.append(offset)
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
                    "metrics": {
                        "totalExecutions": 0,
                        "successfulExecutions": 0,
                        "failedExecutions": 0,
                        "failureRate": 0.0,
                    },
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
                    "metrics": {
                        "totalExecutions": 0,
                        "successfulExecutions": 0,
                        "failedExecutions": 0,
                        "failureRate": 0.0,
                    },
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
                    "metrics": {
                        "totalExecutions": 0,
                        "successfulExecutions": 0,
                        "failedExecutions": 0,
                        "failureRate": 0.0,
                    },
                },
            ]
            page_items = all_tools[offset : offset + limit]
            page_num = (offset // limit) + 1 if limit else 1
            total = len(all_tools)
            data = {"data": page_items, "pagination": {"page": page_num, "perPage": limit, "total": total}}
            return httpx.Response(200, json=data)
        # Gateway lookup for call_tool path
        if request.method == "GET" and request.url.path == "/admin/gateways/g1":
            return httpx.Response(200, json={"id": "g1", "name": "gw", "url": "http://mock/mcp"})
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_async_gateway_strategy_cache_and_auth() -> None:
    state: dict = {"expected_auth": "Bearer xyz"}
    transport = _mock_transport(state)

    mgmt_client = httpx.Client(transport=transport, base_url="http://mock")
    http_client = httpx.AsyncClient(transport=transport, base_url="http://mock")

    gateway = GatewayApiClient("http://mock", auth_token=state["expected_auth"], client=mgmt_client)

    # Fake MCP transport to simulate direct-style invocation
    captured: Dict[str, Any] = {}

    class FakeTransport(AsyncMCPTransport):
        def session(self, endpoint_url: str):
            @asynccontextmanager
            async def _cm():
                class _S:
                    async def call_tool(self, name: str, arguments: Dict[str, Any]):
                        captured["endpoint"] = endpoint_url
                        captured["name"] = name
                        captured["arguments"] = arguments
                        return {"ok": True, "echo": arguments.get("text")}

                yield _S()

            return _cm()

    strategy = AsyncGatewayMcpStrategy(gateway, client=http_client, ttl_seconds=60.0, mcp_transport=FakeTransport())

    tools1 = await strategy.list_tools("g1")
    assert len(tools1) == 1 and tools1[0].name == "workspace.list"
    assert state.get("tools_get_count", 0) == 1

    tools2 = await strategy.list_tools("g1")
    assert len(tools2) == 1 and tools2[0].name == "workspace.list"
    assert state.get("tools_get_count", 0) == 1

    res = await strategy.call_tool("g1", "echo", {"text": "hi"})
    assert res.ok is True
    assert res.data.get("echo") == "hi"
    assert captured["endpoint"] == "http://mock/mcp"

    await http_client.aclose()
    mgmt_client.close()


@pytest.mark.asyncio
async def test_async_gateway_strategy_ttl_zero_no_cache() -> None:
    state: dict = {"expected_auth": "Bearer xyz"}
    transport = _mock_transport(state)

    mgmt_client = httpx.Client(transport=transport, base_url="http://mock")
    http_client = httpx.AsyncClient(transport=transport, base_url="http://mock")

    gateway = GatewayApiClient("http://mock", auth_token=state["expected_auth"], client=mgmt_client)
    strategy = AsyncGatewayMcpStrategy(gateway, client=http_client, ttl_seconds=0.0)

    tools1 = await strategy.list_tools("g1")
    assert len(tools1) == 1 and tools1[0].name == "workspace.list"
    assert state.get("tools_get_count", 0) == 1

    tools2 = await strategy.list_tools("g1")
    assert len(tools2) == 1 and tools2[0].name == "workspace.list"
    assert state.get("tools_get_count", 0) == 2

    await http_client.aclose()
    mgmt_client.close()


@pytest.mark.asyncio
async def test_async_gateway_strategy_call_tool_handles_model_dump_result() -> None:
    state: dict = {"expected_auth": "Bearer xyz"}
    transport = _mock_transport(state)

    mgmt_client = httpx.Client(transport=transport, base_url="http://mock")
    http_client = httpx.AsyncClient(transport=transport, base_url="http://mock")

    gateway = GatewayApiClient("http://mock", auth_token=state["expected_auth"], client=mgmt_client)

    captured: Dict[str, Any] = {}

    class FakeResult:
        def __init__(self, payload: Dict[str, Any]) -> None:
            self._payload = payload

        def model_dump(self) -> Dict[str, Any]:
            return self._payload

    class FakeTransport(AsyncMCPTransport):
        def session(self, endpoint_url: str):  # type: ignore[override]
            @asynccontextmanager
            async def _cm():
                class _S:
                    async def call_tool(self, name: str, arguments: Dict[str, Any]):
                        captured["endpoint"] = endpoint_url
                        captured["name"] = name
                        captured["arguments"] = arguments
                        return FakeResult({"ok": True, "value": arguments.get("x")})

                yield _S()

            return _cm()

    strategy = AsyncGatewayMcpStrategy(
        gateway,
        client=http_client,
        ttl_seconds=60.0,
        mcp_transport=FakeTransport(),
    )

    res = await strategy.call_tool("g1", "compute", {"x": 42})
    assert res.ok is True
    assert res.data == {"ok": True, "value": 42}
    assert captured["endpoint"] == "http://mock/mcp"

    await http_client.aclose()
    mgmt_client.close()


@pytest.mark.asyncio
async def test_async_gateway_strategy_call_tool_wraps_non_dict_result() -> None:
    state: dict = {"expected_auth": "Bearer xyz"}
    transport = _mock_transport(state)

    mgmt_client = httpx.Client(transport=transport, base_url="http://mock")
    http_client = httpx.AsyncClient(transport=transport, base_url="http://mock")

    gateway = GatewayApiClient("http://mock", auth_token=state["expected_auth"], client=mgmt_client)

    captured: Dict[str, Any] = {}

    class FakeTransport(AsyncMCPTransport):
        def session(self, endpoint_url: str):  # type: ignore[override]
            @asynccontextmanager
            async def _cm():
                class _S:
                    async def call_tool(self, name: str, arguments: Dict[str, Any]):
                        captured["endpoint"] = endpoint_url
                        captured["name"] = name
                        captured["arguments"] = arguments
                        return "raw-result"

                yield _S()

            return _cm()

    strategy = AsyncGatewayMcpStrategy(
        gateway,
        client=http_client,
        ttl_seconds=60.0,
        mcp_transport=FakeTransport(),
    )

    res = await strategy.call_tool("g1", "noop", {"x": 1})
    assert res.ok is True
    assert res.data == {"ok": "raw-result"}
    assert captured["endpoint"] == "http://mock/mcp"

    await http_client.aclose()
    mgmt_client.close()


@pytest.mark.asyncio
async def test_async_gateway_strategy_list_tools_page_two_pages() -> None:
    state: dict = {}
    transport = _mock_transport_paginated(state)
    mgmt_client = httpx.Client(transport=transport, base_url="http://mock")
    http_client = httpx.AsyncClient(transport=transport, base_url="http://mock")

    gw = GatewayApiClient("http://mock", client=mgmt_client)
    strat = AsyncGatewayMcpStrategy(gw, client=http_client)

    page1 = await strat.list_tools_page("g1", limit=2)
    assert [t.name for t in page1.items] == ["tool_a", "tool_b"]
    assert page1.next_cursor == "2"

    page2 = await strat.list_tools_page("g1", cursor=page1.next_cursor, limit=2)
    assert [t.name for t in page2.items] == ["tool_c"]
    assert page2.next_cursor is None

    await http_client.aclose()
    mgmt_client.close()


@pytest.mark.asyncio
async def test_async_gateway_strategy_list_tools_page_invalid_cursor_uses_zero_offset() -> None:
    state: dict = {}
    transport = _mock_transport_paginated(state)
    mgmt_client = httpx.Client(transport=transport, base_url="http://mock")
    http_client = httpx.AsyncClient(transport=transport, base_url="http://mock")

    gw = GatewayApiClient("http://mock", client=mgmt_client)
    strat = AsyncGatewayMcpStrategy(gw, client=http_client)

    page = await strat.list_tools_page("g1", cursor="not-an-int", limit=2)
    assert [t.name for t in page.items] == ["tool_a", "tool_b"]
    assert state.get("offsets") == [0]

    await http_client.aclose()
    mgmt_client.close()


@pytest.mark.asyncio
async def test_async_gateway_strategy_list_tools_page_unpaginated_updates_cache() -> None:
    state: dict = {}
    transport = _mock_transport_paginated(state)
    mgmt_client = httpx.Client(transport=transport, base_url="http://mock")
    http_client = httpx.AsyncClient(transport=transport, base_url="http://mock")

    gw = GatewayApiClient("http://mock", client=mgmt_client)
    strat = AsyncGatewayMcpStrategy(gw, client=http_client, ttl_seconds=60.0)

    assert "g1" not in strat._tools_cache
    page = await strat.list_tools_page("g1")
    assert [t.name for t in page.items] == ["tool_a", "tool_b", "tool_c"]
    assert "g1" in strat._tools_cache
    cached_tools, _expires = strat._tools_cache["g1"]
    assert [t.name for t in cached_tools] == ["tool_a", "tool_b", "tool_c"]

    await http_client.aclose()
    mgmt_client.close()


@pytest.mark.asyncio
async def test_async_gateway_strategy_list_tools_page_paginated_does_not_update_cache() -> None:
    state: dict = {}
    transport = _mock_transport_paginated(state)
    mgmt_client = httpx.Client(transport=transport, base_url="http://mock")
    http_client = httpx.AsyncClient(transport=transport, base_url="http://mock")

    gw = GatewayApiClient("http://mock", client=mgmt_client)
    strat = AsyncGatewayMcpStrategy(gw, client=http_client, ttl_seconds=60.0)

    assert "g1" not in strat._tools_cache
    _page = await strat.list_tools_page("g1", limit=2)
    assert "g1" not in strat._tools_cache

    await http_client.aclose()
    mgmt_client.close()
