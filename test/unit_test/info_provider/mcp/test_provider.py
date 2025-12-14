from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterable, List

import httpx
import pytest

from gearmeshing_ai.info_provider.mcp.errors import (
    ServerNotFoundError,
    ToolAccessDeniedError,
)
from gearmeshing_ai.info_provider.mcp.policy import ToolPolicy
from gearmeshing_ai.info_provider.mcp.provider import (
    AsyncMCPInfoProvider,
    MCPInfoProvider,
)
from gearmeshing_ai.info_provider.mcp.schemas.config import (
    GatewayConfig,
    McpClientConfig,
    ServerConfig,
)
from gearmeshing_ai.info_provider.mcp.schemas.core import (
    McpTool,
    ToolCallResult,
    ToolsPage,
)
from gearmeshing_ai.info_provider.mcp.strategy import (
    DirectMcpStrategy,
    GatewayMcpStrategy,
)
from gearmeshing_ai.info_provider.mcp.strategy.base import AsyncStrategy, SyncStrategy
from gearmeshing_ai.info_provider.mcp.strategy.direct_async import (
    AsyncDirectMcpStrategy,
)
from gearmeshing_ai.info_provider.mcp.strategy.gateway_async import (
    AsyncGatewayMcpStrategy,
)
from gearmeshing_ai.info_provider.mcp.transport.mcp import SseMCPTransport


def _mock_transport(state: dict) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        # Admin tools list used by Gateway strategies
        if request.method == "GET" and request.url.path == "/admin/tools":
            data = {
                "data": [
                    {
                        "id": "t_create",
                        "originalName": "create_issue",
                        "name": "issue-create",
                        "gatewaySlug": "gw",
                        "customName": "create_issue",
                        "customNameSlug": "create-issue",
                        "requestType": "SSE",
                        "integrationType": "MCP",
                        "inputSchema": {"type": "object"},
                        "createdAt": "2024-01-01T00:00:00Z",
                        "updatedAt": "2024-01-01T00:00:00Z",
                        "enabled": True,
                        "reachable": True,
                        "executionCount": 0,
                        "metrics": {
                            "totalExecutions": 0,
                            "successfulExecutions": 0,
                            "failedExecutions": 0,
                            "failureRate": 0.0,
                        },
                        "gatewayId": "g1",
                    },
                    {
                        "id": "t_get",
                        "originalName": "get_issue",
                        "name": "issue-get",
                        "gatewaySlug": "gw",
                        "customName": "get_issue",
                        "customNameSlug": "get-issue",
                        "requestType": "SSE",
                        "integrationType": "MCP",
                        "inputSchema": {"type": "object"},
                        "createdAt": "2024-01-01T00:00:00Z",
                        "updatedAt": "2024-01-01T00:00:00Z",
                        "enabled": True,
                        "reachable": True,
                        "executionCount": 0,
                        "metrics": {
                            "totalExecutions": 0,
                            "successfulExecutions": 0,
                            "failedExecutions": 0,
                            "failureRate": 0.0,
                        },
                        "gatewayId": "g1",
                    },
                ]
            }
            return httpx.Response(200, json=data)

        # Legacy: streamable HTTP endpoints under the Gateway
        if request.method == "GET" and request.url.path == "/servers/s1/mcp/tools":
            assert request.headers.get("Authorization") == state.get("expected_auth")
            tools: List[Dict[str, Any]] = [
                {
                    "name": "create_issue",
                    "description": "Create",
                    "inputSchema": {"type": "object"},
                },
                {
                    "name": "get_issue",
                    "description": "Read",
                    "inputSchema": {"type": "object"},
                },
            ]
            return httpx.Response(200, json=tools)

        if request.method == "POST" and request.url.path == "/servers/s1/mcp/a2a/get_issue/invoke":
            return httpx.Response(200, json={"ok": True, "issue": {"id": 1}})

        if request.method == "POST" and request.url.path == "/servers/s1/mcp/a2a/create_issue/invoke":
            return httpx.Response(200, json={"ok": True, "created": True})

        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


class AsyncDummyBase(AsyncStrategy):
    """Concrete AsyncStrategy test double with no-op implementations.

    Subclasses can override list_tools and/or list_tools_page without having to
    reimplement call_tool or streaming helpers, keeping them MyPy-concrete.
    """

    async def list_tools(self, server_id: str) -> List[McpTool]:  # noqa: ARG002
        return []

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> ToolCallResult:  # noqa: ARG002
        return ToolCallResult(ok=True, data={}, error=None)

    def stream_events(self, server_id: str, path: str = "/sse", **_: Any) -> AsyncIterator[str]:  # noqa: ARG002
        async def _gen() -> AsyncIterator[str]:
            yield "data: ok"

        return _gen()

    def stream_events_parsed(
        self,
        server_id: str,
        path: str = "/sse",
        **_: Any,
    ) -> AsyncIterator[Dict[str, Any]]:  # noqa: ARG002
        async def _gen() -> AsyncIterator[Dict[str, Any]]:
            yield {"data": "ok"}

        return _gen()


class SyncDummyBase(SyncStrategy):
    """Concrete SyncStrategy test double with no-op implementations.

    Subclasses typically override list_tools or list_tools_page; call_tool
    simply returns a successful empty ToolCallResult.
    """

    def list_servers(self) -> Iterable[McpTool]:
        return []

    def list_tools(self, server_id: str) -> Iterable[McpTool]:  # noqa: ARG002
        return []

    def call_tool(
        self,
        server_id: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        agent_id: str | None = None,  # noqa: ARG002
    ) -> ToolCallResult:
        return ToolCallResult(ok=True, data={}, error=None)


class TestAsyncMCPInfoProvider:

    @pytest.mark.asyncio
    async def test_from_config_servers_only_builds_direct_strategy(self) -> None:
        cfg = McpClientConfig(
            servers=[ServerConfig(name="s1", endpoint_url="http://mock/mcp")],
            tools_cache_ttl_seconds=30.0,
        )

        client = await AsyncMCPInfoProvider.from_config(cfg, mcp_transport=SseMCPTransport())

        assert len(client._strategies) == 1
        assert isinstance(client._strategies[0], AsyncDirectMcpStrategy)

    @pytest.mark.asyncio
    async def test_from_config_gateway_only_builds_gateway_strategy(self) -> None:
        cfg = McpClientConfig(
            gateway=GatewayConfig(base_url="http://mock", auth_token="token"),
            tools_cache_ttl_seconds=30.0,
        )

        mgmt_client = httpx.Client(base_url="http://mock")
        http_client = httpx.AsyncClient(base_url="http://mock")
        sse_client = httpx.AsyncClient(base_url="http://mock")

        client = await AsyncMCPInfoProvider.from_config(
            cfg,
            mcp_transport=SseMCPTransport(),
            gateway_mgmt_client=mgmt_client,
            gateway_http_client=http_client,
            gateway_sse_client=sse_client,
        )

        try:
            assert len(client._strategies) == 1
            assert isinstance(client._strategies[0], AsyncGatewayMcpStrategy)
        finally:
            await http_client.aclose()
            await sse_client.aclose()
            mgmt_client.close()

    @pytest.mark.asyncio
    async def test_from_config_both_servers_and_gateway_builds_both_strategies(self) -> None:
        cfg = McpClientConfig(
            servers=[ServerConfig(name="s1", endpoint_url="http://mock/mcp")],
            gateway=GatewayConfig(base_url="http://mock", auth_token="token"),
            tools_cache_ttl_seconds=30.0,
        )

        mgmt_client = httpx.Client(base_url="http://mock")
        http_client = httpx.AsyncClient(base_url="http://mock")
        sse_client = httpx.AsyncClient(base_url="http://mock")

        client = await AsyncMCPInfoProvider.from_config(
            cfg,
            mcp_transport=SseMCPTransport(),
            gateway_mgmt_client=mgmt_client,
            gateway_http_client=http_client,
            gateway_sse_client=sse_client,
        )

        try:
            types = {type(s) for s in client._strategies}
            assert AsyncDirectMcpStrategy in types
            assert AsyncGatewayMcpStrategy in types
        finally:
            await http_client.aclose()
            await sse_client.aclose()
            mgmt_client.close()

    @pytest.mark.asyncio
    async def test_from_config_empty_config_has_no_strategies(self) -> None:
        cfg = McpClientConfig()

        client = await AsyncMCPInfoProvider.from_config(cfg, mcp_transport=SseMCPTransport())

        assert getattr(client, "_strategies") == []

    @pytest.mark.asyncio
    async def test_provider_list_with_policy(self) -> None:
        state = {"expected_auth": "Bearer aaa"}
        transport = _mock_transport(state)

        # Async HTTP client for streamable endpoints
        http_client = httpx.AsyncClient(transport=transport, base_url="http://mock")
        mgmt_client = httpx.Client(transport=transport, base_url="http://mock")

        cfg = McpClientConfig(
            gateway=GatewayConfig(base_url="http://mock", auth_token=state["expected_auth"]),
            tools_cache_ttl_seconds=60.0,
        )
        policies = {"agent": ToolPolicy(allowed_servers={"s1"}, read_only=False)}

        client = await AsyncMCPInfoProvider.from_config(
            cfg,
            mcp_transport=SseMCPTransport(),
            agent_policies=policies,
            gateway_http_client=http_client,
            gateway_mgmt_client=mgmt_client,
        )

        tools = await client.list_tools("s1", agent_id="agent")
        assert {t.name for t in tools} == {"create_issue", "get_issue"}

        await http_client.aclose()
        mgmt_client.close()

    @pytest.mark.asyncio
    async def test_list_tools_applies_allowed_tools_policy(self) -> None:
        state = {"expected_auth": "Bearer aaa"}
        transport = _mock_transport(state)

        http_client = httpx.AsyncClient(transport=transport, base_url="http://mock")
        mgmt_client = httpx.Client(transport=transport, base_url="http://mock")

        cfg = McpClientConfig(
            gateway=GatewayConfig(base_url="http://mock", auth_token=state["expected_auth"]),
            tools_cache_ttl_seconds=60.0,
        )
        policies = {"agent": ToolPolicy(allowed_servers={"s1"}, allowed_tools={"get_issue"})}

        client = await AsyncMCPInfoProvider.from_config(
            cfg,
            mcp_transport=SseMCPTransport(),
            agent_policies=policies,
            gateway_http_client=http_client,
            gateway_mgmt_client=mgmt_client,
        )

        tools = await client.list_tools("s1", agent_id="agent")
        assert [t.name for t in tools] == ["get_issue"]

        await http_client.aclose()

    @pytest.mark.asyncio
    async def test_provider_read_only_filters_mutations_on_list(self) -> None:
        state = {"expected_auth": "Bearer bbb"}
        transport = _mock_transport(state)
        http_client = httpx.AsyncClient(transport=transport, base_url="http://mock")
        mgmt_client = httpx.Client(transport=transport, base_url="http://mock")

        cfg = McpClientConfig(
            gateway=GatewayConfig(base_url="http://mock", auth_token=state["expected_auth"]),
            tools_cache_ttl_seconds=60.0,
        )
        policies = {"agent": ToolPolicy(allowed_servers={"s1"}, read_only=True)}

        client = await AsyncMCPInfoProvider.from_config(
            cfg,
            mcp_transport=SseMCPTransport(),
            agent_policies=policies,
            gateway_http_client=http_client,
            gateway_mgmt_client=mgmt_client,
        )

        tools = await client.list_tools("s1", agent_id="agent")
        assert [t.name for t in tools] == ["get_issue"]

        await http_client.aclose()

    @pytest.mark.asyncio
    async def test_list_tools_page_falls_back_when_strategy_has_no_pagination(self) -> None:
        class DummyAsyncNoPagination(AsyncDummyBase):
            async def list_tools(self, server_id: str) -> List[McpTool]:  # noqa: ARG002
                return [
                    McpTool(
                        name="t1",
                        description="",
                        mutating=False,
                        arguments=[],
                        raw_parameters_schema={},
                    )
                ]

        client = AsyncMCPInfoProvider(strategies=[DummyAsyncNoPagination()])

        page = await client.list_tools_page("s1")
        assert isinstance(page, ToolsPage)
        assert [t.name for t in page.items] == ["t1"]
        assert page.next_cursor is None

    @pytest.mark.asyncio
    async def test_list_tools_page_uses_native_pagination_when_available(self) -> None:
        class DummyAsyncWithPagination(AsyncDummyBase):
            async def list_tools(self, server_id: str) -> List[McpTool]:  # noqa: ARG002
                return []

            async def list_tools_page(
                self, server_id: str, *, cursor: str | None = None, limit: int | None = None
            ) -> ToolsPage:
                items = [
                    McpTool(
                        name="t2",
                        description="",
                        mutating=False,
                        arguments=[],
                        raw_parameters_schema={},
                    )
                ]
                return ToolsPage(items=items, next_cursor="next" if cursor is None else None)

        client = AsyncMCPInfoProvider(strategies=[DummyAsyncWithPagination()])

        page = await client.list_tools_page("s1", cursor=None, limit=1)
        assert [t.name for t in page.items] == ["t2"]
        assert page.next_cursor == "next"

    @pytest.mark.asyncio
    async def test_list_tools_raises_server_not_found_when_no_strategy_succeeds(self) -> None:
        client = AsyncMCPInfoProvider(strategies=[])

        with pytest.raises(ServerNotFoundError):
            await client.list_tools("unknown")

    @pytest.mark.asyncio
    async def test_list_tools_denied_when_server_not_allowed_by_policy(self) -> None:
        # Policy has allowed_servers that does NOT include "s1"; denial happens
        policies = {"agent": ToolPolicy(allowed_servers={"other"})}
        client = AsyncMCPInfoProvider(strategies=[], agent_policies=policies)

        with pytest.raises(ToolAccessDeniedError):
            await client.list_tools("s1", agent_id="agent")

    @pytest.mark.asyncio
    async def test_init_rejects_non_async_strategy(self) -> None:
        class NotAStrategy:
            pass

        with pytest.raises(TypeError):
            AsyncMCPInfoProvider(strategies=[NotAStrategy()])  # type: ignore[list-item]

    @pytest.mark.asyncio
    async def test_list_tools_skips_failing_strategy_and_uses_next(self) -> None:
        class FailingThenWorking(AsyncDummyBase):
            def __init__(self, fail: bool) -> None:
                self._fail = fail

            async def list_tools(self, server_id: str) -> List[McpTool]:  # noqa: ARG002
                if self._fail:
                    raise RuntimeError("boom")
                return [
                    McpTool(
                        name="ok",
                        description="",
                        mutating=False,
                        arguments=[],
                        raw_parameters_schema={},
                    )
                ]

        client = AsyncMCPInfoProvider(strategies=[FailingThenWorking(True), FailingThenWorking(False)])

        tools = await client.list_tools("s1")
        assert [t.name for t in tools] == ["ok"]

    @pytest.mark.asyncio
    async def test_list_tools_page_applies_allowed_tools_and_read_only_policy(self) -> None:
        class Paginated(AsyncDummyBase):
            async def list_tools(self, server_id: str) -> List[McpTool]:  # noqa: ARG002
                return []

            async def list_tools_page(
                self,
                server_id: str,
                *,
                cursor: str | None = None,
                limit: int | None = None,
            ) -> ToolsPage:
                items = [
                    McpTool(
                        name="keep",
                        description="",
                        mutating=False,
                        arguments=[],
                        raw_parameters_schema={},
                    ),
                    McpTool(
                        name="drop_mutating",
                        description="",
                        mutating=True,
                        arguments=[],
                        raw_parameters_schema={},
                    ),
                ]
                return ToolsPage(items=items, next_cursor=None)

        policies = {
            "agent": ToolPolicy(allowed_servers={"s1"}, allowed_tools={"keep", "drop_mutating"}, read_only=True)
        }
        client = AsyncMCPInfoProvider(strategies=[Paginated()], agent_policies=policies)

        page = await client.list_tools_page("s1", agent_id="agent")
        # allowed_tools keeps both by name, read_only then drops the mutating one
        assert [t.name for t in page.items] == ["keep"]

    @pytest.mark.asyncio
    async def test_list_tools_page_denied_when_server_not_allowed_by_policy(self) -> None:
        policies = {"agent": ToolPolicy(allowed_servers={"other"})}
        client = AsyncMCPInfoProvider(strategies=[], agent_policies=policies)

        with pytest.raises(ToolAccessDeniedError):
            await client.list_tools_page("s1", agent_id="agent")

    @pytest.mark.asyncio
    async def test_list_tools_page_falls_back_when_pagination_strategy_raises(self) -> None:
        class RaisingPage(AsyncDummyBase):
            async def list_tools(self, server_id: str) -> List[McpTool]:  # noqa: ARG002
                return []

            async def list_tools_page(
                self,
                server_id: str,
                *,
                cursor: str | None = None,
                limit: int | None = None,
            ) -> ToolsPage:
                raise RuntimeError("pagination not available")

        class ListOnly(AsyncDummyBase):
            async def list_tools(self, server_id: str) -> List[McpTool]:  # noqa: ARG002
                return [
                    McpTool(
                        name="from_list_only",
                        description="",
                        mutating=False,
                        arguments=[],
                        raw_parameters_schema={},
                    )
                ]

        client = AsyncMCPInfoProvider(strategies=[RaisingPage(), ListOnly()])

        page = await client.list_tools_page("s1")
        # first strategy's list_tools_page fails; provider falls back to list_tools
        assert [t.name for t in page.items] == ["from_list_only"]

    # No call_tool/streaming helpers are exposed on AsyncMCPInfoProvider; those
    # remain responsibilities of underlying strategies.


class TestSyncMCPInfoProvider:

    def test_from_config_servers_only_builds_direct_strategy(self) -> None:
        cfg = McpClientConfig(servers=[ServerConfig(name="direct1", endpoint_url="http://mock/mcp")])

        client = MCPInfoProvider.from_config(cfg, mcp_transport=SseMCPTransport())

        assert len(client._strategies) == 1
        assert isinstance(client._strategies[0], DirectMcpStrategy)

    def test_from_config_gateway_only_builds_gateway_strategy(self) -> None:
        cfg = McpClientConfig(
            gateway=GatewayConfig(base_url="http://mock", auth_token="token"),
        )

        mgmt_client = httpx.Client(base_url="http://mock")
        http_client = httpx.Client(base_url="http://mock")

        client = MCPInfoProvider.from_config(
            cfg,
            mcp_transport=SseMCPTransport(),
            gateway_mgmt_client=mgmt_client,
            gateway_http_client=http_client,
        )

        try:
            assert len(client._strategies) == 1
            assert isinstance(client._strategies[0], GatewayMcpStrategy)
        finally:
            mgmt_client.close()
            http_client.close()

    def test_from_config_both_servers_and_gateway_builds_both_strategies(self) -> None:
        cfg = McpClientConfig(
            servers=[ServerConfig(name="direct1", endpoint_url="http://mock/mcp")],
            gateway=GatewayConfig(base_url="http://mock", auth_token="token"),
        )

        mgmt_client = httpx.Client(base_url="http://mock")
        http_client = httpx.Client(base_url="http://mock")

        client = MCPInfoProvider.from_config(
            cfg,
            mcp_transport=SseMCPTransport(),
            gateway_mgmt_client=mgmt_client,
            gateway_http_client=http_client,
        )

        try:
            types = {type(s) for s in client._strategies}
            assert DirectMcpStrategy in types
            assert GatewayMcpStrategy in types
        finally:
            mgmt_client.close()
            http_client.close()

    def test_from_config_empty_config_has_no_strategies(self) -> None:
        cfg = McpClientConfig()

        client = MCPInfoProvider.from_config(cfg, mcp_transport=SseMCPTransport())

        assert getattr(client, "_strategies") == []

    def test_provider_list_with_policy(self) -> None:
        state = {"expected_auth": "Bearer aaa"}
        transport = _mock_transport(state)

        http_client = httpx.Client(transport=transport, base_url="http://mock")

        cfg = McpClientConfig(
            gateway=GatewayConfig(base_url="http://mock", auth_token=state["expected_auth"]),
            tools_cache_ttl_seconds=60.0,
        )
        policies = {"agent": ToolPolicy(allowed_servers={"s1"}, read_only=False)}

        client = MCPInfoProvider.from_config(
            cfg,
            mcp_transport=SseMCPTransport(),
            agent_policies=policies,
            gateway_http_client=http_client,
            gateway_mgmt_client=http_client,
        )

        tools = client.list_tools("s1", agent_id="agent")
        assert {t.name for t in tools} == {"create_issue", "get_issue"}

        http_client.close()

    def test_provider_read_only_filters_mutations_on_list(self) -> None:
        state = {"expected_auth": "Bearer bbb"}
        transport = _mock_transport(state)
        http_client = httpx.Client(transport=transport, base_url="http://mock")

        cfg = McpClientConfig(
            gateway=GatewayConfig(base_url="http://mock", auth_token=state["expected_auth"]),
            tools_cache_ttl_seconds=60.0,
        )
        policies = {"agent": ToolPolicy(allowed_servers={"s1"}, read_only=True)}

        client = MCPInfoProvider.from_config(
            cfg,
            mcp_transport=SseMCPTransport(),
            agent_policies=policies,
            gateway_http_client=http_client,
            gateway_mgmt_client=http_client,
        )

        tools = client.list_tools("s1", agent_id="agent")
        # read_only should filter out mutating tools when listing
        assert [t.name for t in tools] == ["get_issue"]

        http_client.close()

    def test_list_tools_page_falls_back_when_strategy_has_no_pagination(self) -> None:
        class DummySyncNoPagination(SyncDummyBase):
            def list_tools(self, server_id: str) -> Iterable[McpTool]:  # noqa: ARG002
                return [
                    McpTool(
                        name="t1",
                        description="",
                        mutating=False,
                        arguments=[],
                        raw_parameters_schema={},
                    )
                ]

        client = MCPInfoProvider(strategies=[DummySyncNoPagination()])

        page = client.list_tools_page("s1")
        assert isinstance(page, ToolsPage)
        assert [t.name for t in page.items] == ["t1"]
        assert page.next_cursor is None

    def test_list_tools_page_uses_native_pagination_when_available(self) -> None:
        class DummySyncWithPagination(SyncDummyBase):
            def list_tools(self, server_id: str) -> Iterable[McpTool]:  # noqa: ARG002
                return []

            def list_tools_page(
                self, server_id: str, *, cursor: str | None = None, limit: int | None = None
            ) -> ToolsPage:
                items = [
                    McpTool(
                        name="t2",
                        description="",
                        mutating=False,
                        arguments=[],
                        raw_parameters_schema={},
                    )
                ]
                return ToolsPage(items=items, next_cursor="next" if cursor is None else None)

        client = MCPInfoProvider(strategies=[DummySyncWithPagination()])

        page = client.list_tools_page("s1", cursor=None, limit=1)
        assert [t.name for t in page.items] == ["t2"]
        assert page.next_cursor == "next"

    def test_list_tools_raises_server_not_found_when_no_strategy_succeeds(self) -> None:
        client = MCPInfoProvider(strategies=[])

        with pytest.raises(ServerNotFoundError):
            client.list_tools("unknown")

    def test_list_tools_denied_when_server_not_allowed_by_policy(self) -> None:
        policies = {"agent": ToolPolicy(allowed_servers={"other"})}
        client = MCPInfoProvider(strategies=[], agent_policies=policies)

        with pytest.raises(ToolAccessDeniedError):
            client.list_tools("s1", agent_id="agent")

    def test_init_rejects_non_sync_strategy(self) -> None:
        class NotASyncStrategy:
            pass

        with pytest.raises(TypeError):
            MCPInfoProvider(strategies=[NotASyncStrategy()])  # type: ignore[list-item]

    def test_list_tools_skips_failing_strategy_and_uses_next(self) -> None:
        class FailingThenWorking(SyncDummyBase):
            def __init__(self, fail: bool) -> None:
                self._fail = fail

            def list_tools(self, server_id: str) -> Iterable[McpTool]:  # noqa: ARG002
                if self._fail:
                    raise RuntimeError("boom")
                return [
                    McpTool(
                        name="ok",
                        description="",
                        mutating=False,
                        arguments=[],
                        raw_parameters_schema={},
                    )
                ]

        client = MCPInfoProvider(strategies=[FailingThenWorking(True), FailingThenWorking(False)])

        tools = client.list_tools("s1")
        assert [t.name for t in tools] == ["ok"]

    def test_list_tools_page_applies_allowed_tools_and_read_only_policy(self) -> None:
        class Paginated(SyncDummyBase):
            def list_tools(self, server_id: str) -> Iterable[McpTool]:  # noqa: ARG002
                return []

            def list_tools_page(
                self,
                server_id: str,
                *,
                cursor: str | None = None,
                limit: int | None = None,
            ) -> ToolsPage:
                items = [
                    McpTool(
                        name="keep",
                        description="",
                        mutating=False,
                        arguments=[],
                        raw_parameters_schema={},
                    ),
                    McpTool(
                        name="drop_mutating",
                        description="",
                        mutating=True,
                        arguments=[],
                        raw_parameters_schema={},
                    ),
                ]
                return ToolsPage(items=items, next_cursor=None)

        policies = {
            "agent": ToolPolicy(allowed_servers={"s1"}, allowed_tools={"keep", "drop_mutating"}, read_only=True)
        }
        client = MCPInfoProvider(strategies=[Paginated()], agent_policies=policies)

        page = client.list_tools_page("s1", agent_id="agent")
        assert [t.name for t in page.items] == ["keep"]

    def test_list_tools_page_denied_when_server_not_allowed_by_policy(self) -> None:
        policies = {"agent": ToolPolicy(allowed_servers={"other"})}
        client = MCPInfoProvider(strategies=[], agent_policies=policies)

        with pytest.raises(ToolAccessDeniedError):
            client.list_tools_page("s1", agent_id="agent")

    def test_list_tools_page_falls_back_when_pagination_strategy_raises(self) -> None:
        class RaisingPage(SyncDummyBase):
            def list_tools(self, server_id: str) -> Iterable[McpTool]:  # noqa: ARG002
                return []

            def list_tools_page(
                self,
                server_id: str,
                *,
                cursor: str | None = None,
                limit: int | None = None,
            ) -> ToolsPage:
                raise RuntimeError("no pagination")

        class ListOnly(SyncDummyBase):
            def list_tools(self, server_id: str) -> Iterable[McpTool]:  # noqa: ARG002
                return [
                    McpTool(
                        name="from_list_only",
                        description="",
                        mutating=False,
                        arguments=[],
                        raw_parameters_schema={},
                    )
                ]

        client = MCPInfoProvider(strategies=[RaisingPage(), ListOnly()])

        page = client.list_tools_page("s1")
        assert [t.name for t in page.items] == ["from_list_only"]

    # No call_tool helper is exposed on MCPInfoProvider; tool invocation is
    # delegated to lower-level strategy/client layers.
