from __future__ import annotations

from typing import Any, Dict, List

from gearmeshing_ai.info_provider.mcp.policy import ToolPolicy
from gearmeshing_ai.info_provider.mcp.provider import MCPInfoProvider
from gearmeshing_ai.info_provider.mcp.schemas.config import (
    McpClientConfig,
    ServerConfig,
)
from gearmeshing_ai.info_provider.mcp.strategy import DirectMcpStrategy


def _fake_mcp_transport_for_readonly():
    from contextlib import asynccontextmanager

    class _FakeTool:
        def __init__(self, name: str, description: str | None, input_schema: Dict[str, Any]) -> None:
            self.name = name
            self.description = description
            self.inputSchema = input_schema

    class _FakeListToolsResp:
        def __init__(self, tools: List[_FakeTool]) -> None:
            self.tools = tools
            self.next_cursor = None

    class _FakeSession:
        async def list_tools(self, cursor: str | None = None, limit: int | None = None):  # noqa: ARG002
            return _FakeListToolsResp([
                _FakeTool("create_issue", "Create", {"type": "object"}),
                _FakeTool("get_issue", "Read", {"type": "object"}),
            ])

        async def call_tool(self, name: str, arguments: Dict[str, Any] | None = None):  # noqa: ARG002
            return {"ok": True}

    class _FakeMCPTransport:
        def session(self, endpoint_url: str):  # noqa: ARG002
            @asynccontextmanager
            async def _cm():
                yield _FakeSession()

            return _cm()

    return _FakeMCPTransport()


def test_read_only_policy_filters_and_blocks_mutations() -> None:
    cfg = McpClientConfig(servers=[ServerConfig(name="direct1", endpoint_url="http://mock/mcp")])

    policies = {"agent": ToolPolicy(allowed_servers={"direct1"}, read_only=True)}

    # Build provider using DirectMcpStrategy with a fake MCP transport
    strat = DirectMcpStrategy(cfg.servers, mcp_transport=_fake_mcp_transport_for_readonly())
    client = MCPInfoProvider(strategies=[strat], agent_policies=policies)

    tools = client.list_tools("direct1", agent_id="agent")
    assert [t.name for t in tools] == ["get_issue"]
