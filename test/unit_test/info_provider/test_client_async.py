from __future__ import annotations

from typing import Any, Dict, List

import httpx
import pytest

from gearmeshing_ai.info_provider.mcp.client_async import AsyncMcpClient
from gearmeshing_ai.info_provider.mcp.policy import ToolPolicy
from gearmeshing_ai.info_provider.mcp.schemas.config import GatewayConfig, McpClientConfig


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
async def test_async_client_list_and_call_with_policy() -> None:
    state = {"expected_auth": "Bearer aaa"}
    transport = _mock_transport(state)

    # Async HTTP client for streamable endpoints
    http_client = httpx.AsyncClient(transport=transport, base_url="http://mock")

    cfg = McpClientConfig(
        gateway=GatewayConfig(base_url="http://mock", auth_token=state["expected_auth"]),
        tools_cache_ttl_seconds=60.0,
    )
    policies = {"agent": ToolPolicy(allowed_servers={"s1"}, read_only=False)}

    client = await AsyncMcpClient.from_config(cfg, agent_policies=policies, gateway_http_client=http_client)

    tools = await client.list_tools("s1", agent_id="agent")
    assert {t.name for t in tools} == {"create_issue", "get_issue"}

    res = await client.call_tool("s1", "get_issue", {}, agent_id="agent")
    assert res.ok is True and res.data.get("issue", {}).get("id") == 1

    await http_client.aclose()


@pytest.mark.asyncio
async def test_async_client_read_only_blocks_mutations() -> None:
    state = {"expected_auth": "Bearer bbb"}
    transport = _mock_transport(state)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://mock")

    cfg = McpClientConfig(
        gateway=GatewayConfig(base_url="http://mock", auth_token=state["expected_auth"]),
        tools_cache_ttl_seconds=60.0,
    )
    policies = {"agent": ToolPolicy(allowed_servers={"s1"}, read_only=True)}

    client = await AsyncMcpClient.from_config(cfg, agent_policies=policies, gateway_http_client=http_client)

    tools = await client.list_tools("s1", agent_id="agent")
    assert [t.name for t in tools] == ["get_issue"]

    from gearmeshing_ai.info_provider.mcp.errors import ToolAccessDeniedError

    with pytest.raises(ToolAccessDeniedError):
        await client.call_tool("s1", "create_issue", {}, agent_id="agent")

    await http_client.aclose()
