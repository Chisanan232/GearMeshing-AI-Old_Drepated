from __future__ import annotations

from typing import Any, Dict, List

import httpx

from gearmeshing_ai.info_provider.mcp.policy import ToolPolicy
from gearmeshing_ai.info_provider.mcp.provider import MCPInfoProvider
from gearmeshing_ai.info_provider.mcp.schemas.config import (
    McpClientConfig,
    ServerConfig,
)


def _mock_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        # Direct server expected at http://mock/mcp
        if request.method == "GET" and request.url.path == "/mcp/tools":
            data: List[Dict[str, Any]] = [
                {"name": "create_issue", "description": "Create", "inputSchema": {"type": "object"}},
                {"name": "get_issue", "description": "Read", "inputSchema": {"type": "object"}},
            ]
            return httpx.Response(200, json=data)
        if request.method == "POST" and request.url.path == "/mcp/a2a/get_issue/invoke":
            return httpx.Response(200, json={"ok": True, "issue": {"id": 1}})
        if request.method == "POST" and request.url.path == "/mcp/a2a/create_issue/invoke":
            return httpx.Response(200, json={"ok": True, "created": True})
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


def test_read_only_policy_filters_and_blocks_mutations() -> None:
    transport = _mock_transport()
    http_client = httpx.Client(transport=transport, base_url="http://mock")

    cfg = McpClientConfig(servers=[ServerConfig(name="direct1", endpoint_url="http://mock/mcp")])

    policies = {"agent": ToolPolicy(allowed_servers={"direct1"}, read_only=True)}

    client = MCPInfoProvider.from_config(
        cfg,
        agent_policies=policies,
        direct_http_client=http_client,
    )

    # List tools should only return non-mutating tool when read_only is set
    tools = client.list_tools("direct1", agent_id="agent")
    assert [t.name for t in tools] == ["get_issue"]
