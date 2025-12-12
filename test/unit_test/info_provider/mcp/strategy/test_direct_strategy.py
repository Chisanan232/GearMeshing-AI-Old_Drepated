from __future__ import annotations

from typing import Any, Dict, List
import pytest

from contextlib import asynccontextmanager

from gearmeshing_ai.info_provider.mcp.schemas.config import ServerConfig
from gearmeshing_ai.info_provider.mcp.strategy.direct import DirectMcpStrategy


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


def test_direct_strategy_cache_feature() -> None:
    # Transport that counts list_tools invocations
    from contextlib import asynccontextmanager

    class _CountingSession:
        def __init__(self, state: dict) -> None:
            self._state = state

        async def list_tools(self, cursor: str | None = None, limit: int | None = None):  # noqa: ARG002
            self._state["calls"] = self._state.get("calls", 0) + 1
            return _FakeListToolsResp([
                _FakeTool("echo", "Echo tool", {"type": "object"})
            ])

        async def call_tool(self, name: str, arguments: Dict[str, Any] | None = None):  # noqa: ARG002
            return {"ok": True}

    class _CountingTransport:
        def __init__(self, state: dict) -> None:
            self._state = state

        def session(self, endpoint_url: str):  # noqa: ARG002
            @asynccontextmanager
            async def _cm():
                yield _CountingSession(self._state)

            return _cm()

    state: dict = {}
    strategy = DirectMcpStrategy([ServerConfig(name="s1", endpoint_url="http://mock")], mcp_transport=_CountingTransport(state))
    # First list populates cache
    tools1 = list(strategy.list_tools("s1"))
    assert [t.name for t in tools1] == ["echo"]
    assert state.get("calls", 0) == 1
    # Second list served from cache
    tools2 = list(strategy.list_tools("s1"))
    assert [t.name for t in tools2] == ["echo"]
    assert state.get("calls", 0) == 1


@pytest.mark.parametrize(
    "variant",
    ["typed", "model_dump", "dict", "raw"],
)
def test_direct_strategy_call_tool_result_variants(monkeypatch: pytest.MonkeyPatch, variant: str) -> None:
    from contextlib import asynccontextmanager
    from gearmeshing_ai.info_provider.mcp.strategy import direct as d_mod

    class _Typed:
        def model_dump(self) -> Dict[str, Any]:
            return {"ok": True, "kind": "typed"}

    class _ModelDumpOnly:
        def model_dump(self) -> Dict[str, Any]:
            return {"ok": True, "kind": "model_dump"}

    # Ensure isinstance(res, MCPCallToolResult) branch is hit
    monkeypatch.setattr(d_mod, "MCPCallToolResult", _Typed, raising=True)

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

    strategy = DirectMcpStrategy([ServerConfig(name="s1", endpoint_url="http://mock")], mcp_transport=_VaryingTransport())
    res = strategy.call_tool("s1", "echo", {})
    assert res.ok is True
    assert isinstance(res.data, dict)
    assert "ok" in res.data


def test_direct_strategy_mutating_call_invalidates_cache() -> None:
    # Prime cache
    strategy = DirectMcpStrategy([ServerConfig(name="s1", endpoint_url="http://mock")], mcp_transport=_FakeMCPTransport())
    list(strategy.list_tools("s1"))
    assert "s1" in strategy._tools_cache
    # Mutating call should clear cache
    res = strategy.call_tool("s1", "create_issue", {})
    assert res.ok is True
    assert "s1" not in strategy._tools_cache
