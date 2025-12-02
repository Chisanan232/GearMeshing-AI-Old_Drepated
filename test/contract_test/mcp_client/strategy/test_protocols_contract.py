from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, Iterable, List

import pytest

from gearmeshing_ai.mcp_client.strategy.base import (
    AsyncStrategy,
    StrategyCommonMixin,
    SyncStrategy,
    is_mutating_tool_name,
)
from gearmeshing_ai.mcp_client.schemas.core import (
    McpServerRef,
    McpTool,
    ServerKind,
    ToolCallResult,
    ToolArgument,
    TransportType,
)


class DummySyncStrategy(StrategyCommonMixin):
    """Minimal concrete class satisfying the SyncStrategy Protocol.

    Provides deterministic, in-memory behavior so implementors can rely on
    the abstract contract without external dependencies.
    """

    def __init__(self) -> None:
        self._schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Issue ID"},
                "count": {"type": "integer"},
            },
            "required": ["id"],
        }

    def list_servers(self) -> Iterable[McpServerRef]:
        yield McpServerRef(
            id="s1",
            display_name="Local Server",
            kind=ServerKind.DIRECT,
            transport=TransportType.STREAMABLE_HTTP,
            endpoint_url="http://mock/mcp/",
        )

    def list_tools(self, server_id: str) -> Iterable[McpTool]:  # noqa: ARG002
        args = self._infer_arguments(self._schema)
        yield McpTool(
            name="get_issue",
            description="Fetch an issue",
            mutating=False,
            arguments=args,
            raw_parameters_schema=self._schema,
        )

    def call_tool(
        self,
        server_id: str,  # noqa: ARG002
        tool_name: str,
        args: dict[str, Any],
        *,
        agent_id: str | None = None,  # noqa: ARG002
    ) -> ToolCallResult:
        return ToolCallResult(ok=True, data={"tool": tool_name, "parameters": dict(args)})


class DummyAsyncStrategy(StrategyCommonMixin):
    """Minimal concrete class satisfying the AsyncStrategy Protocol."""

    def __init__(self) -> None:
        self._schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Issue ID"},
            },
            "required": ["id"],
        }

    async def list_tools(self, server_id: str) -> List[McpTool]:  # noqa: ARG002
        return [
            McpTool(
                name="get_issue",
                description="Fetch an issue",
                mutating=False,
                arguments=self._infer_arguments(self._schema),
                raw_parameters_schema=self._schema,
            )
        ]

    async def call_tool(self, server_id: str, tool_name: str, args: dict[str, Any]) -> ToolCallResult:  # noqa: ARG002
        return ToolCallResult(ok=True, data={"tool": tool_name, "parameters": dict(args)})

    def stream_events(
        self,
        server_id: str,  # noqa: ARG002
        path: str = "/sse",  # noqa: ARG002
        *,
        reconnect: bool = False,  # noqa: ARG002
        max_retries: int = 3,  # noqa: ARG002
        backoff_initial: float = 0.5,  # noqa: ARG002
        backoff_factor: float = 2.0,  # noqa: ARG002
        backoff_max: float = 8.0,  # noqa: ARG002
        idle_timeout: float | None = None,  # noqa: ARG002
        max_total_seconds: float | None = None,  # noqa: ARG002
    ) -> AsyncIterator[str]:
        async def _gen() -> AsyncIterator[str]:
            yield "id: 1"
            await asyncio.sleep(0)
            yield ""
            await asyncio.sleep(0)
            yield "data: done"
        return _gen()

    def stream_events_parsed(
        self,
        server_id: str,  # noqa: ARG002
        path: str = "/sse",  # noqa: ARG002
        *,
        reconnect: bool = False,  # noqa: ARG002
        max_retries: int = 3,  # noqa: ARG002
        backoff_initial: float = 0.5,  # noqa: ARG002
        backoff_factor: float = 2.0,  # noqa: ARG002
        backoff_max: float = 8.0,  # noqa: ARG002
        idle_timeout: float | None = None,  # noqa: ARG002
        max_total_seconds: float | None = None,  # noqa: ARG002
    ) -> AsyncIterator[Dict[str, Any]]:
        async def _gen() -> AsyncIterator[Dict[str, Any]]:
            yield {"id": "1", "event": "message", "data": "hello"}
        return _gen()


# ------------------------------
# SyncStrategy contracts
# ------------------------------

def test_sync_strategy_runtime_protocol_conformance() -> None:
    s = DummySyncStrategy()
    assert isinstance(s, SyncStrategy), "DummySyncStrategy should satisfy SyncStrategy runtime protocol"


def test_sync_strategy_basic_contract() -> None:
    s = DummySyncStrategy()
    servers = list(s.list_servers())
    assert len(servers) == 1 and servers[0].id == "s1"

    tools = list(s.list_tools("s1"))
    assert tools and tools[0].name == "get_issue"
    # arguments inferred from schema
    arg_names = {a.name for a in tools[0].arguments}
    assert {"id", "count"}.issubset(arg_names)
    required_map = {a.name: a.required for a in tools[0].arguments}
    assert required_map.get("id") is True

    res = s.call_tool("s1", "get_issue", {"id": "X"})
    assert isinstance(res, ToolCallResult) and res.ok is True and res.data["tool"] == "get_issue"


# ------------------------------
# AsyncStrategy contracts
# ------------------------------

def test_async_strategy_runtime_protocol_conformance() -> None:
    s = DummyAsyncStrategy()
    assert isinstance(s, AsyncStrategy), "DummyAsyncStrategy should satisfy AsyncStrategy runtime protocol"


@pytest.mark.asyncio
async def test_async_strategy_basic_contract() -> None:
    s = DummyAsyncStrategy()
    tools = await s.list_tools("s1")
    assert tools and tools[0].name == "get_issue"

    res = await s.call_tool("s1", "get_issue", {"id": "Y"})
    assert isinstance(res, ToolCallResult) and res.ok is True and res.data["parameters"]["id"] == "Y"

    # stream raw
    lines: List[str] = []
    async for ln in s.stream_events("s1"):
        lines.append(ln)
        if len(lines) >= 3:
            break
    assert any(l.startswith("id:") for l in lines)

    # stream parsed
    events: List[Dict[str, Any]] = []
    async for evt in s.stream_events_parsed("s1"):
        events.append(evt)
        break
    assert events and events[0]["event"] == "message"


# ------------------------------
# StrategyCommonMixin helpers
# ------------------------------

def test_mutating_heuristic_global_and_mixin() -> None:
    mixin = StrategyCommonMixin()
    cases = {
        "create_issue": True,
        "updateThing": True,
        "delete": True,
        "post_message": True,
        "write_file": True,
        "set_status": True,
        "get": False,
        "search": False,
        "list": False,
    }
    for name, expected in cases.items():
        assert is_mutating_tool_name(name) is expected
        assert mixin._is_mutating_tool_name(name) is expected


def test_infer_arguments_maps_json_schema() -> None:
    mixin = StrategyCommonMixin()
    schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Primary ID"},
            "count": {"type": "integer"},
        },
        "required": ["id"],
    }
    args: List[ToolArgument] = mixin._infer_arguments(schema)
    names = {a.name for a in args}
    assert names == {"id", "count"}
    req = {a.name: a.required for a in args}
    assert req["id"] is True and req["count"] is False
