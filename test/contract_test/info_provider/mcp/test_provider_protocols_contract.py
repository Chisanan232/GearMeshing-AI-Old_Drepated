from __future__ import annotations

from typing import Dict, List, Optional

import pytest

from gearmeshing_ai.info_provider.mcp.base import (
    BaseAsyncMCPInfoProvider,
    BaseMCPInfoProvider,
    ClientCommonMixin,
)
from gearmeshing_ai.info_provider.mcp.policy import ToolPolicy
from gearmeshing_ai.info_provider.mcp.schemas.config import McpClientConfig
from gearmeshing_ai.info_provider.mcp.schemas.core import (
    McpServerRef,
    McpTool,
    ServerKind,
    ToolArgument,
    ToolsPage,
    TransportType,
)


class _DummySyncProvider(ClientCommonMixin):
    """Minimal concrete impl satisfying MCP info-provider contracts (sync)."""

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
    ) -> "_DummySyncProvider":
        return cls()

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


class _DummyAsyncProvider(ClientCommonMixin):
    """Minimal concrete impl satisfying async MCP info-provider contracts."""

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
    ) -> "_DummyAsyncProvider":
        return cls()

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

    # No call_tool or streaming helpers: async provider is tools-only.


# ------------------------------
# Runtime conformance
# ------------------------------


def test_sync_provider_runtime_protocol_conformance() -> None:
    c = _DummySyncProvider()
    assert isinstance(c, BaseMCPInfoProvider)


@pytest.mark.asyncio
async def test_async_provider_runtime_protocol_conformance() -> None:
    c = await _DummyAsyncProvider.from_config(McpClientConfig())
    assert isinstance(c, BaseAsyncMCPInfoProvider)


# ------------------------------
# Basic contracts
# ------------------------------


def test_sync_provider_basic_contract() -> None:
    c = _DummySyncProvider()
    tools = c.list_tools("s1")
    assert tools and {t.name for t in tools} == {"get_issue", "create_issue"}

    page = c.list_tools_page("s1")
    assert isinstance(page, ToolsPage) and page.next_cursor is None and page.items


@pytest.mark.asyncio
async def test_async_provider_basic_contract() -> None:
    c = await _DummyAsyncProvider.from_config(McpClientConfig())
    tools = await c.list_tools("s1")
    assert tools and tools[0].name == "get_issue"

    page = await c.list_tools_page("s1")
    assert isinstance(page, ToolsPage) and page.items


# ------------------------------
# ClientCommonMixin contracts
# ------------------------------


def test_provider_common_mixin_filters_and_blocking_contract() -> None:
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
