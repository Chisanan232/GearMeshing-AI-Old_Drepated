from __future__ import annotations

from typing import Any, Dict, List
from contextlib import asynccontextmanager

import pytest

from gearmeshing_ai.info_provider.mcp.schemas.config import ServerConfig
from gearmeshing_ai.info_provider.mcp.strategy.direct_async import (
    AsyncDirectMcpStrategy,
)


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
    def __init__(self, state: dict) -> None:
        self._state = state

    async def list_tools(self, cursor: str | None = None, limit: int | None = None):  # noqa: ARG002
        self._state["tools_get_count"] = self._state.get("tools_get_count", 0) + 1
        tool = _FakeTool(
            "echo",
            "Echo tool",
            {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Text to echo"}},
                "required": ["text"],
            },
        )
        return _FakeListToolsResp([tool])

    async def call_tool(self, name: str, arguments: Dict[str, Any] | None = None):  # noqa: ARG002
        args = dict(arguments or {})
        return {"ok": True, "echo": args.get("text")}


class _FakeMCPTransport:
    def __init__(self, state: dict) -> None:
        self._state = state

    def session(self, endpoint_url: str):  # noqa: ARG002
        @asynccontextmanager
        async def _cm():
            yield _FakeSession(self._state)

        return _cm()


@pytest.mark.asyncio
async def test_async_direct_strategy_cache_and_auth() -> None:
    state: dict = {}

    servers = [ServerConfig(name="s1", endpoint_url="http://mock", auth_token=None)]
    strategy = AsyncDirectMcpStrategy(servers, ttl_seconds=60.0, mcp_transport=_FakeMCPTransport(state))

    tools1 = await strategy.list_tools("s1")
    assert len(tools1) == 1 and tools1[0].name == "echo"
    assert state.get("tools_get_count", 0) == 1

    tools2 = await strategy.list_tools("s1")
    assert len(tools2) == 1 and tools2[0].name == "echo"
    assert state.get("tools_get_count", 0) == 1

    res = await strategy.call_tool("s1", "echo", {"text": "hi"})
    assert res.ok is True
    assert res.data.get("echo") == "hi"


@pytest.mark.asyncio
async def test_async_direct_strategy_ttl_zero_no_cache() -> None:
    state: dict = {}

    servers = [ServerConfig(name="s1", endpoint_url="http://mock", auth_token=None)]
    strategy = AsyncDirectMcpStrategy(servers, ttl_seconds=0.0, mcp_transport=_FakeMCPTransport(state))

    tools1 = await strategy.list_tools("s1")
    assert len(tools1) == 1 and tools1[0].name == "echo"
    assert state.get("tools_get_count", 0) == 1

    tools2 = await strategy.list_tools("s1")
    assert len(tools2) == 1 and tools2[0].name == "echo"
    assert state.get("tools_get_count", 0) == 2
