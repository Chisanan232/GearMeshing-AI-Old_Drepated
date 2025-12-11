from __future__ import annotations

from typing import Any, Dict, List

import httpx
import pytest

from gearmeshing_ai.info_provider.mcp.policy import ToolPolicy
from gearmeshing_ai.info_provider.mcp.provider import AsyncMCPInfoProvider, MCPInfoProvider
from gearmeshing_ai.info_provider.mcp.schemas.config import (
    GatewayConfig,
    McpClientConfig,
    ServerConfig,
)
from gearmeshing_ai.info_provider.mcp.strategy.direct_async import AsyncDirectMcpStrategy
from gearmeshing_ai.info_provider.mcp.strategy.gateway_async import AsyncGatewayMcpStrategy
from gearmeshing_ai.info_provider.mcp.strategy import DirectMcpStrategy, GatewayMcpStrategy


def _mock_transport(state: dict) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        # Streamable HTTP endpoints under the Gateway
        if request.method == "GET" and request.url.path == "/servers/s1/mcp/tools":
            assert request.headers.get("Authorization") == state.get("expected_auth")
            tools: List[Dict[str, Any]] = [
                {
                    "name": "create_issue",
                    "description": "Create",
                    "inputSchema": {"type": "object"},
                },
                {
                    "name": "get_issue",
                    "description": "Read",
                    "inputSchema": {"type": "object"},
                },
            ]
            return httpx.Response(200, json=tools)

        if request.method == "POST" and request.url.path == "/servers/s1/mcp/a2a/get_issue/invoke":
            return httpx.Response(200, json={"ok": True, "issue": {"id": 1}})

        if request.method == "POST" and request.url.path == "/servers/s1/mcp/a2a/create_issue/invoke":
            return httpx.Response(200, json={"ok": True, "created": True})

        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_async_from_config_servers_only_builds_direct_strategy() -> None:
    cfg = McpClientConfig(
        servers=[ServerConfig(name="s1", endpoint_url="http://mock/mcp")],
        tools_cache_ttl_seconds=30.0,
    )

    client = await AsyncMCPInfoProvider.from_config(cfg)

    assert len(client._strategies) == 1  # type: ignore[attr-defined]
    assert isinstance(client._strategies[0], AsyncDirectMcpStrategy)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_async_from_config_gateway_only_builds_gateway_strategy() -> None:
    cfg = McpClientConfig(
        gateway=GatewayConfig(base_url="http://mock", auth_token="token"),
        tools_cache_ttl_seconds=30.0,
    )

    mgmt_client = httpx.Client(base_url="http://mock")
    http_client = httpx.AsyncClient(base_url="http://mock")
    sse_client = httpx.AsyncClient(base_url="http://mock")

    client = await AsyncMCPInfoProvider.from_config(
        cfg,
        gateway_mgmt_client=mgmt_client,
        gateway_http_client=http_client,
        gateway_sse_client=sse_client,
    )

    try:
        assert len(client._strategies) == 1  # type: ignore[attr-defined]
        assert isinstance(client._strategies[0], AsyncGatewayMcpStrategy)  # type: ignore[attr-defined]
    finally:
        await http_client.aclose()
        await sse_client.aclose()
        mgmt_client.close()


@pytest.mark.asyncio
async def test_async_from_config_both_servers_and_gateway_builds_both_strategies() -> None:
    cfg = McpClientConfig(
        servers=[ServerConfig(name="s1", endpoint_url="http://mock/mcp")],
        gateway=GatewayConfig(base_url="http://mock", auth_token="token"),
        tools_cache_ttl_seconds=30.0,
    )

    mgmt_client = httpx.Client(base_url="http://mock")
    http_client = httpx.AsyncClient(base_url="http://mock")
    sse_client = httpx.AsyncClient(base_url="http://mock")

    client = await AsyncMCPInfoProvider.from_config(
        cfg,
        gateway_mgmt_client=mgmt_client,
        gateway_http_client=http_client,
        gateway_sse_client=sse_client,
    )

    try:
        types = {type(s) for s in client._strategies}  # type: ignore[attr-defined]
        assert AsyncDirectMcpStrategy in types
        assert AsyncGatewayMcpStrategy in types
    finally:
        await http_client.aclose()
        await sse_client.aclose()
        mgmt_client.close()


@pytest.mark.asyncio
async def test_async_from_config_empty_config_has_no_strategies() -> None:
    cfg = McpClientConfig()

    client = await AsyncMCPInfoProvider.from_config(cfg)

    assert getattr(client, "_strategies") == []


def test_sync_from_config_servers_only_builds_direct_strategy() -> None:
    cfg = McpClientConfig(servers=[ServerConfig(name="direct1", endpoint_url="http://mock/mcp")])

    client = MCPInfoProvider.from_config(cfg)

    assert len(client._strategies) == 1  # type: ignore[attr-defined]
    assert isinstance(client._strategies[0], DirectMcpStrategy)  # type: ignore[attr-defined]


def test_sync_from_config_gateway_only_builds_gateway_strategy() -> None:
    cfg = McpClientConfig(
        gateway=GatewayConfig(base_url="http://mock", auth_token="token"),
    )

    mgmt_client = httpx.Client(base_url="http://mock")
    http_client = httpx.Client(base_url="http://mock")

    client = MCPInfoProvider.from_config(
        cfg,
        gateway_mgmt_client=mgmt_client,
        gateway_http_client=http_client,
    )

    try:
        assert len(client._strategies) == 1  # type: ignore[attr-defined]
        assert isinstance(client._strategies[0], GatewayMcpStrategy)  # type: ignore[attr-defined]
    finally:
        mgmt_client.close()
        http_client.close()


def test_sync_from_config_both_servers_and_gateway_builds_both_strategies() -> None:
    cfg = McpClientConfig(
        servers=[ServerConfig(name="direct1", endpoint_url="http://mock/mcp")],
        gateway=GatewayConfig(base_url="http://mock", auth_token="token"),
    )

    mgmt_client = httpx.Client(base_url="http://mock")
    http_client = httpx.Client(base_url="http://mock")

    client = MCPInfoProvider.from_config(
        cfg,
        gateway_mgmt_client=mgmt_client,
        gateway_http_client=http_client,
    )

    try:
        types = {type(s) for s in client._strategies}  # type: ignore[attr-defined]
        assert DirectMcpStrategy in types
        assert GatewayMcpStrategy in types
    finally:
        mgmt_client.close()
        http_client.close()


def test_sync_from_config_empty_config_has_no_strategies() -> None:
    cfg = McpClientConfig()

    client = MCPInfoProvider.from_config(cfg)

    assert getattr(client, "_strategies") == []


def test_sync_provider_list_and_call_with_policy() -> None:
    state = {"expected_auth": "Bearer aaa"}
    transport = _mock_transport(state)

    http_client = httpx.Client(transport=transport, base_url="http://mock")

    cfg = McpClientConfig(
        gateway=GatewayConfig(base_url="http://mock", auth_token=state["expected_auth"]),
        tools_cache_ttl_seconds=60.0,
    )
    policies = {"agent": ToolPolicy(allowed_servers={"s1"}, read_only=False)}

    client = MCPInfoProvider.from_config(
        cfg,
        agent_policies=policies,
        gateway_http_client=http_client,
        gateway_mgmt_client=http_client,
    )

    tools = client.list_tools("s1", agent_id="agent")
    assert {t.name for t in tools} == {"create_issue", "get_issue"}

    res = client.call_tool("s1", "get_issue", {}, agent_id="agent")
    assert res.ok is True and res.data.get("issue", {}).get("id") == 1

    http_client.close()


def test_sync_provider_read_only_blocks_mutations() -> None:
    state = {"expected_auth": "Bearer bbb"}
    transport = _mock_transport(state)
    http_client = httpx.Client(transport=transport, base_url="http://mock")

    cfg = McpClientConfig(
        gateway=GatewayConfig(base_url="http://mock", auth_token=state["expected_auth"]),
        tools_cache_ttl_seconds=60.0,
    )
    policies = {"agent": ToolPolicy(allowed_servers={"s1"}, read_only=True)}

    client = MCPInfoProvider.from_config(
        cfg,
        agent_policies=policies,
        gateway_http_client=http_client,
        gateway_mgmt_client=http_client,
    )

    tools = client.list_tools("s1", agent_id="agent")
    # read_only should filter out mutating tools when listing
    assert [t.name for t in tools] == ["get_issue"]

    from gearmeshing_ai.info_provider.mcp.errors import ToolAccessDeniedError

    with pytest.raises(ToolAccessDeniedError):
        client.call_tool("s1", "create_issue", {}, agent_id="agent")

    http_client.close()


@pytest.mark.asyncio
async def test_async_provider_list_and_call_with_policy() -> None:
    state = {"expected_auth": "Bearer aaa"}
    transport = _mock_transport(state)

    # Async HTTP client for streamable endpoints
    http_client = httpx.AsyncClient(transport=transport, base_url="http://mock")

    cfg = McpClientConfig(
        gateway=GatewayConfig(base_url="http://mock", auth_token=state["expected_auth"]),
        tools_cache_ttl_seconds=60.0,
    )
    policies = {"agent": ToolPolicy(allowed_servers={"s1"}, read_only=False)}

    client = await AsyncMCPInfoProvider.from_config(cfg, agent_policies=policies, gateway_http_client=http_client)

    tools = await client.list_tools("s1", agent_id="agent")
    assert {t.name for t in tools} == {"create_issue", "get_issue"}

    res = await client.call_tool("s1", "get_issue", {}, agent_id="agent")
    assert res.ok is True and res.data.get("issue", {}).get("id") == 1

    await http_client.aclose()


@pytest.mark.asyncio
async def test_async_provider_read_only_blocks_mutations() -> None:
    state = {"expected_auth": "Bearer bbb"}
    transport = _mock_transport(state)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://mock")

    cfg = McpClientConfig(
        gateway=GatewayConfig(base_url="http://mock", auth_token=state["expected_auth"]),
        tools_cache_ttl_seconds=60.0,
    )
    policies = {"agent": ToolPolicy(allowed_servers={"s1"}, read_only=True)}

    client = await AsyncMCPInfoProvider.from_config(cfg, agent_policies=policies, gateway_http_client=http_client)

    tools = await client.list_tools("s1", agent_id="agent")
    assert [t.name for t in tools] == ["get_issue"]

    from gearmeshing_ai.info_provider.mcp.errors import ToolAccessDeniedError

    with pytest.raises(ToolAccessDeniedError):
        await client.call_tool("s1", "create_issue", {}, agent_id="agent")

    await http_client.aclose()
