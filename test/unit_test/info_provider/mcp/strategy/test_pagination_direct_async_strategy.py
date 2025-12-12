from __future__ import annotations

from typing import Any, Dict, List
from contextlib import asynccontextmanager

import pytest

from gearmeshing_ai.info_provider.mcp.schemas.config import ServerConfig
from gearmeshing_ai.info_provider.mcp.strategy.direct_async import AsyncDirectMcpStrategy


class _FakeTool:
    def __init__(self, name: str, description: str | None, input_schema: Dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema


class _FakeListToolsResp:
    def __init__(self, tools: List[_FakeTool], next_cursor: str | None) -> None:
        self.tools = tools
        self.next_cursor = next_cursor


class _FakeSession:
    async def list_tools(self, cursor: str | None = None, limit: int | None = None):  # noqa: ARG002
        if cursor is None:
            return _FakeListToolsResp([
                _FakeTool("t1", "one", {"type": "object"}),
                _FakeTool("t2", "two", {"type": "object"}),
            ], next_cursor="cursor-2")
        if cursor == "cursor-2":
            return _FakeListToolsResp([
                _FakeTool("t3", "three", {"type": "object"}),
            ], next_cursor=None)
        return _FakeListToolsResp([], next_cursor=None)


class _FakeMCPTransport:
    def session(self, endpoint_url: str):  # noqa: ARG002
        @asynccontextmanager
        async def _cm():
            yield _FakeSession()
        return _cm()


@pytest.mark.asyncio
async def test_async_direct_strategy_list_tools_page_pagination() -> None:
    servers = [ServerConfig(name="s1", endpoint_url="http://mock", auth_token=None)]
    strategy = AsyncDirectMcpStrategy(servers, ttl_seconds=60.0, mcp_transport=_FakeMCPTransport())

    page1 = await strategy.list_tools_page("s1", limit=2)
    assert [t.name for t in page1.items] == ["t1", "t2"]
    assert page1.next_cursor == "cursor-2"

    page2 = await strategy.list_tools_page("s1", cursor=page1.next_cursor, limit=2)
    assert [t.name for t in page2.items] == ["t3"]
    assert page2.next_cursor is None
