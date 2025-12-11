from __future__ import annotations

from typing import Any, Dict, List

import httpx
import pytest

from gearmeshing_ai.info_provider.mcp.provider import MCPInfoProvider
from gearmeshing_ai.info_provider.mcp.errors import ToolAccessDeniedError
from gearmeshing_ai.info_provider.mcp.policy import ToolPolicy
from gearmeshing_ai.info_provider.mcp.schemas.config import (
    McpClientConfig,
    ServerConfig,
)


def _mock_direct_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        # Direct server expected at http://mock/mcp
        if request.method == "GET" and request.url.path == "/mcp/tools":
            data: List[Dict[str, Any]] = [
                {
                    "name": "echo",
                    "description": "Echo tool",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"text": {"type": "string", "description": "Text to echo"}},
                        "required": ["text"],
                    },
                },
                {
                    "name": "other",
                    "description": "Another tool",
                    "inputSchema": {"type": "object", "properties": {}},
                },
            ]
            return httpx.Response(200, json=data)
        if request.method == "POST" and request.url.path == "/mcp/a2a/echo/invoke":
            return httpx.Response(200, json={"ok": True, "result": "ok"})
        if request.method == "POST" and request.url.path == "/mcp/a2a/other/invoke":
            return httpx.Response(200, json={"ok": True, "result": "ok"})
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


def test_client_from_config_list_servers_and_tools_with_policy() -> None:
    transport = _mock_direct_transport()
    http_client = httpx.Client(transport=transport, base_url="http://mock")

    cfg = McpClientConfig(servers=[ServerConfig(name="direct1", endpoint_url="http://mock/mcp")])

    policies = {"agent": ToolPolicy(allowed_servers={"direct1"}, allowed_tools={"echo"})}

    client = MCPInfoProvider.from_config(
        cfg,
        agent_policies=policies,
        direct_http_client=http_client,
    )

    servers = client.list_servers(agent_id="agent")
    assert [s.id for s in servers] == ["direct1"]

    tools = client.list_tools("direct1", agent_id="agent")
    assert [t.name for t in tools] == ["echo"]

    res = client.call_tool("direct1", "echo", {"text": "hi"}, agent_id="agent")
    assert res.ok is True


def test_client_policy_denies_server_and_tool() -> None:
    transport = _mock_direct_transport()
    http_client = httpx.Client(transport=transport, base_url="http://mock")

    cfg = McpClientConfig(servers=[ServerConfig(name="direct1", endpoint_url="http://mock/mcp")])

    policies = {"agent": ToolPolicy(allowed_servers={"direct1"}, allowed_tools={"echo"})}

    client = MCPInfoProvider.from_config(
        cfg,
        agent_policies=policies,
        direct_http_client=http_client,
    )

    with pytest.raises(ToolAccessDeniedError):
        client.list_tools("forbidden", agent_id="agent")

    with pytest.raises(ToolAccessDeniedError):
        client.call_tool("direct1", "other", {}, agent_id="agent")
