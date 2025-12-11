from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest

from gearmeshing_ai.info_provider.mcp.base import (
    BaseAsyncMCPInfoProvider,
    ClientCommonMixin,
    BaseMCPInfoProvider,
)
from gearmeshing_ai.info_provider.mcp.policy import ToolPolicy
from gearmeshing_ai.info_provider.mcp.schemas.config import McpClientConfig
from gearmeshing_ai.info_provider.mcp.schemas.core import (
    McpServerRef,
    McpTool,
    ServerKind,
    ToolArgument,
    ToolCallResult,
    ToolsPage,
    TransportType,
)


class _DummySyncClient(ClientCommonMixin):
    """Minimal concrete impl satisfying MCPInfoProvider for contract testing."""

    def __init__(self) -> None:
        self._servers = [
            McpServerRef(
                id="s1",
                display_name="Local",
                kind=ServerKind.DIRECT,
                transport=TransportType.STREAMABLE_HTTP,
                endpoint_url="http://mock/mcp/",
            )
        ]
        self._tools = [
            McpTool(
                name="get_issue",
                description="Fetch issue",
                mutating=False,
                arguments=[ToolArgument(name="id", type="string", required=True)],
                raw_parameters_schema={"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
            ),
            McpTool(
                name="create_issue",
                description="Create",
                mutating=True,
                arguments=[ToolArgument(name="title", type="string", required=True)],
                raw_parameters_schema={
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                },
            ),
        ]

    # Factory
    @classmethod
    def from_config(
        cls,
        config: McpClientConfig,  # noqa: ARG002
        *,
        agent_policies: Optional[Dict[str, ToolPolicy]] = None,  # noqa: ARG002
        direct_http_client=None,  # noqa: ARG002
        gateway_mgmt_client=None,  # noqa: ARG002
        gateway_http_client=None,  # noqa: ARG002
    ) -> "_DummySyncClient":
        return cls()

    # API
    def get_endpoints(self, *, agent_id: str | None = None) -> List[McpServerRef]:  # noqa: ARG002
        return list(self._servers)

    # Back-compat name that matches earlier protocol; used elsewhere in tests.
    def list_servers(self, *, agent_id: str | None = None) -> List[McpServerRef]:  # noqa: ARG002
        return self.get_endpoints(agent_id=agent_id)

    def list_tools(self, server_id: str, *, agent_id: str | None = None) -> List[McpTool]:  # noqa: ARG002
        return list(self._tools)

    def list_tools_page(
        self,
        server_id: str,  # noqa: ARG002
        *,
        cursor: Optional[str] = None,  # noqa: ARG002
        limit: Optional[int] = None,  # noqa: ARG002
        agent_id: str | None = None,  # noqa: ARG002
    ) -> ToolsPage:
        return ToolsPage(items=list(self._tools), next_cursor=None)

    def call_tool(
        self,
        server_id: str,  # noqa: ARG002
        tool_name: str,
        args: dict[str, Any],
        *,
        agent_id: str | None = None,  # noqa: ARG002
    ) -> ToolCallResult:
        return ToolCallResult(ok=True, data={"tool": tool_name, "parameters": dict(args)})


class _DummyAsyncClient(ClientCommonMixin):
    """Minimal concrete impl satisfying AsyncMCPInfoProvider for contract testing."""

    def __init__(self) -> None:
        self._tools = [
            McpTool(
                name="get_issue",
                description="Fetch issue",
                mutating=False,
                arguments=[ToolArgument(name="id", type="string", required=True)],
                raw_parameters_schema={"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
            ),
        ]

    @classmethod
    async def from_config(
        cls,
        config: McpClientConfig,  # noqa: ARG002
        *,
        agent_policies: Optional[Dict[str, ToolPolicy]] = None,  # noqa: ARG002
        gateway_mgmt_client=None,  # noqa: ARG002
        gateway_http_client=None,  # noqa: ARG002
        gateway_sse_client=None,  # noqa: ARG002
    ) -> "_DummyAsyncClient":
        return cls()

    async def get_endpoints(self, *, agent_id: str | None = None) -> List[McpServerRef]:  # noqa: ARG002
        return [
            McpServerRef(
                id="s1",
                display_name="Local",
                kind=ServerKind.DIRECT,
                transport=TransportType.STREAMABLE_HTTP,
                endpoint_url="http://mock/mcp/",
            )
        ]

    # Back-compat alias for older async protocol/tests
    async def list_servers(self, *, agent_id: str | None = None) -> List[McpServerRef]:  # noqa: ARG002
        return await self.get_endpoints(agent_id=agent_id)

    async def list_tools(self, server_id: str, *, agent_id: str | None = None) -> List[McpTool]:  # noqa: ARG002
        return list(self._tools)

    async def list_tools_page(
        self,
        server_id: str,  # noqa: ARG002
        *,
        cursor: Optional[str] = None,  # noqa: ARG002
        limit: Optional[int] = None,  # noqa: ARG002
        agent_id: str | None = None,  # noqa: ARG002
    ) -> ToolsPage:
        return ToolsPage(items=list(self._tools), next_cursor=None)

    async def call_tool(
        self,
        server_id: str,  # noqa: ARG002
        tool_name: str,
        args: dict[str, Any],
        *,
        agent_id: str | None = None,  # noqa: ARG002
    ) -> ToolCallResult:
        return ToolCallResult(ok=True, data={"tool": tool_name, "parameters": dict(args)})

    # Streaming methods (async def returning AsyncIterator per protocol)
    async def stream_events(
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
            yield "data: ok"

        return _gen()

    async def stream_events_parsed(
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
            yield {"id": "1", "event": "message", "data": "ok"}

        return _gen()


# ------------------------------
# Runtime conformance
# ------------------------------


def test_sync_client_runtime_protocol_conformance() -> None:
    c = _DummySyncClient()
    assert isinstance(c, BaseMCPInfoProvider)


@pytest.mark.asyncio
async def test_async_client_runtime_protocol_conformance() -> None:
    c = await _DummyAsyncClient.from_config(McpClientConfig())
    assert isinstance(c, BaseAsyncMCPInfoProvider)


# ------------------------------
# Basic contracts
# ------------------------------


def test_sync_client_basic_contract() -> None:
    c = _DummySyncClient()
    servers = c.list_servers()
    assert servers and servers[0].id == "s1"

    tools = c.list_tools("s1")
    assert tools and {t.name for t in tools} == {"get_issue", "create_issue"}

    page = c.list_tools_page("s1")
    assert isinstance(page, ToolsPage) and page.next_cursor is None and page.items

    result = c.call_tool("s1", "get_issue", {"id": "X"})
    assert isinstance(result, ToolCallResult) and result.ok and result.data["parameters"]["id"] == "X"


@pytest.mark.asyncio
async def test_async_client_basic_contract() -> None:
    c = await _DummyAsyncClient.from_config(McpClientConfig())
    tools = await c.list_tools("s1")
    assert tools and tools[0].name == "get_issue"

    page = await c.list_tools_page("s1")
    assert isinstance(page, ToolsPage) and page.items

    result = await c.call_tool("s1", "get_issue", {"id": "Y"})
    assert isinstance(result, ToolCallResult) and result.ok and result.data["parameters"]["id"] == "Y"

    # streaming
    raw: List[str] = []
    raw_stream = await c.stream_events("s1")
    async for ln in raw_stream:
        raw.append(ln)
        break
    assert raw and raw[0].startswith("id:")

    parsed: List[Dict[str, Any]] = []
    parsed_stream = await c.stream_events_parsed("s1")
    async for evt in parsed_stream:
        parsed.append(evt)
        break
    assert parsed and parsed[0]["event"] == "message"


# ------------------------------
# ClientCommonMixin contracts
# ------------------------------


def test_client_common_mixin_filters_and_blocking_contract() -> None:
    mixin = ClientCommonMixin()

    servers = [
        McpServerRef(
            id="s1",
            display_name="S1",
            kind=ServerKind.DIRECT,
            transport=TransportType.STREAMABLE_HTTP,
            endpoint_url="http://mock/mcp/",
        ),
        McpServerRef(
            id="s2",
            display_name="S2",
            kind=ServerKind.DIRECT,
            transport=TransportType.STREAMABLE_HTTP,
            endpoint_url="http://mock/mcp/",
        ),
    ]

    tools = [
        McpTool(
            name="get_issue",
            description=None,
            mutating=False,
            arguments=[],
            raw_parameters_schema={},
        ),
        McpTool(
            name="create_issue",
            description=None,
            mutating=True,
            arguments=[],
            raw_parameters_schema={},
        ),
    ]

    policy = ToolPolicy(allowed_servers={"s1"}, allowed_tools={"get_issue"}, read_only=True)

    # Servers filtering
    filtered_servers = mixin._filter_servers_by_policy(servers, policy)
    assert [s.id for s in filtered_servers] == ["s1"]

    # Tools filtering (honors allowed_tools and read_only)
    filtered_tools = mixin._filter_tools_by_policy(tools, policy)
    assert [t.name for t in filtered_tools] == ["get_issue"]

    # Read-only blocking based on listed metadata
    listed = {t.name: t for t in tools}
    assert mixin._should_block_read_only(policy, "create_issue", listed) is True
    assert mixin._should_block_read_only(policy, "get_issue", listed) is False

    # Read-only blocking fallback to heuristic when not listed
    assert mixin._should_block_read_only(policy, "update_record", listed_lookup=None) is True
