from __future__ import annotations

from typing import Any, Dict, List

from contextlib import asynccontextmanager

from gearmeshing_ai.info_provider.mcp.schemas.config import ServerConfig
from gearmeshing_ai.info_provider.mcp.strategy.direct import DirectMcpStrategy


class _FakeTool:
    def __init__(self, name: str, description: str | None, input_schema: Dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema


class _FakeListToolsResp:
    def __init__(self, tools: List[_FakeTool]) -> None:
        self.tools = tools
        self.next_cursor = None


class _FakeSession:
    async def list_tools(self, cursor: str | None = None, limit: int | None = None):  # noqa: ARG002
        return _FakeListToolsResp([
            _FakeTool(
                "echo",
                "Echo tool",
                {
                    "type": "object",
                    "properties": {"text": {"type": "string", "description": "Text to echo"}},
                    "required": ["text"],
                },
            )
        ])

    async def call_tool(self, name: str, arguments: Dict[str, Any] | None = None):  # noqa: ARG002
        args = dict(arguments or {})
        return {"ok": True, "echo": args.get("text")}


class _FakeMCPTransport:
    def session(self, endpoint_url: str):  # noqa: ARG002
        @asynccontextmanager
        async def _cm():
            yield _FakeSession()

        return _cm()


def test_direct_strategy_list_and_call() -> None:
    servers = [ServerConfig(name="direct1", endpoint_url="http://mock/mcp")]

    strategy = DirectMcpStrategy(servers, mcp_transport=_FakeMCPTransport())

    servers_list = list(strategy.list_servers())
    assert len(servers_list) == 1
    assert servers_list[0].id == "direct1"

    tools = list(strategy.list_tools("direct1"))
    assert len(tools) == 1
    assert tools[0].name == "echo"
    assert tools[0].raw_parameters_schema["properties"]["text"]["type"] == "string"

    res = strategy.call_tool("direct1", "echo", {"text": "hi"})
    assert res.ok is True
    assert res.data.get("echo") == "hi"
