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


@pytest.mark.asyncio
async def test_async_direct_strategy_unknown_server_raises() -> None:
    strategy = AsyncDirectMcpStrategy([ServerConfig(name="s1", endpoint_url="http://mock")])
    with pytest.raises(ValueError):
        await strategy.list_tools("unknown")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "variant",
    [
        "typed",
        "model_dump",
        "dict",
        "raw",
    ],
)
async def test_async_direct_strategy_call_tool_result_variants(monkeypatch: pytest.MonkeyPatch, variant: str) -> None:
    from contextlib import asynccontextmanager

    class _ModelDumpOnly:
        def __init__(self) -> None:
            pass

        def model_dump(self) -> Dict[str, Any]:
            return {"ok": True, "kind": "model_dump"}

    class _Typed:
        def model_dump(self) -> Dict[str, Any]:
            return {"ok": True, "kind": "typed"}

    # Patch the strategy's MCPCallToolResult to our local _Typed to exercise the isinstance branch
    from gearmeshing_ai.info_provider.mcp.strategy import direct_async as da_mod

    monkeypatch.setattr(da_mod, "MCPCallToolResult", _Typed, raising=True)

    class _VaryingSession:
        async def list_tools(self, cursor: str | None = None, limit: int | None = None):  # noqa: ARG002
            return _FakeListToolsResp([_FakeTool("echo", None, {"type": "object"})])

        async def call_tool(self, name: str, arguments: Dict[str, Any] | None = None):  # noqa: ARG002
            if variant == "typed":
                return _Typed()
            if variant == "model_dump":
                return _ModelDumpOnly()
            if variant == "dict":
                return {"ok": True, "kind": "dict"}
            return "ok"

    class _VaryingTransport:
        def session(self, endpoint_url: str):  # noqa: ARG002
            @asynccontextmanager
            async def _cm():
                yield _VaryingSession()

            return _cm()

    strat = AsyncDirectMcpStrategy([ServerConfig(name="s1", endpoint_url="http://mock")], mcp_transport=_VaryingTransport())
    res = await strat.call_tool("s1", "echo", {})
    assert res.ok is True
    assert isinstance(res.data, dict)
    assert "ok" in res.data


@pytest.mark.asyncio
async def test_async_direct_strategy_mutating_call_invalidates_cache() -> None:
    # Setup strategy and prime the cache
    state: dict = {}
    strat = AsyncDirectMcpStrategy([ServerConfig(name="s1", endpoint_url="http://mock")], mcp_transport=_FakeMCPTransport(state))
    await strat.list_tools("s1")
    assert "s1" in strat._tools_cache
    # Mutating tool name should invalidate cache on success
    res = await strat.call_tool("s1", "create_issue", {})
    assert res.ok is True
    assert "s1" not in strat._tools_cache


@pytest.mark.asyncio
async def test_async_direct_strategy_pagination_and_cache_behavior() -> None:
    from contextlib import asynccontextmanager

    class _PageSession:
        def __init__(self, state: dict) -> None:
            self._state = state

        async def list_tools(self, cursor: str | None = None, limit: int | None = None):  # noqa: ARG002
            self._state["calls"] = self._state.get("calls", 0) + 1
            tool = _FakeTool("echo", None, {"type": "object"})
            resp = _FakeListToolsResp([tool])
            resp.next_cursor = None if cursor else "next"
            return resp

    class _PageTransport:
        def __init__(self, state: dict) -> None:
            self._state = state

        def session(self, endpoint_url: str):  # noqa: ARG002
            @asynccontextmanager
            async def _cm():
                yield _PageSession(self._state)

            return _cm()

    state: dict = {}
    strat = AsyncDirectMcpStrategy([ServerConfig(name="s1", endpoint_url="http://mock")], mcp_transport=_PageTransport(state))

    # First page (no cursor/limit) should update cache
    page1 = await strat.list_tools_page("s1")
    assert [t.name for t in page1.items] == ["echo"]
    assert page1.next_cursor == "next"
    assert "s1" in strat._tools_cache

    # Next page with cursor should NOT update cache
    page2 = await strat.list_tools_page("s1", cursor=page1.next_cursor, limit=1)
    assert [t.name for t in page2.items] == ["echo"]
    assert state.get("calls", 0) >= 2
