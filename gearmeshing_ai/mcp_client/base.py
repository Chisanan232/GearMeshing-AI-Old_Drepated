"""Client protocols and shared helpers for the MCP client facade.

Defines the sync/async client protocols that concrete clients must satisfy,
and a `ClientCommonMixin` with helpers for policy-based filtering.

Usage:
- `McpClient` and `AsyncMcpClient` implement these protocols.
- Strategies focus on transport; clients focus on policy and composition.
"""

from __future__ import annotations

from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Protocol,
    Self,
    runtime_checkable,
)

import httpx

from gearmeshing_ai.mcp_client.policy import PolicyMap, ToolPolicy
from gearmeshing_ai.mcp_client.schemas.config import McpClientConfig
from gearmeshing_ai.mcp_client.schemas.core import (
    McpServerRef,
    McpTool,
    ToolCallResult,
    ToolsPage,
)
from gearmeshing_ai.mcp_client.strategy.base import is_mutating_tool_name


@runtime_checkable
class SyncClientProtocol(Protocol):
    """Protocol for synchronous MCP clients.

    Concrete implementations should provide discovery, listing, pagination,
    and tool invocation through one or more underlying `SyncStrategy`s.
    """
    @classmethod
    def from_config(
        cls,
        config: McpClientConfig,
        *,
        agent_policies: Optional[PolicyMap] = None,
        direct_http_client: Optional[httpx.Client] = None,
        gateway_mgmt_client: Optional[httpx.Client] = None,
        gateway_http_client: Optional[httpx.Client] = None,
    ) -> Self: ...

    def list_servers(self, *, agent_id: str | None = None) -> List[McpServerRef]: ...

    def list_tools(self, server_id: str, *, agent_id: str | None = None) -> List[McpTool]: ...

    def list_tools_page(
        self,
        server_id: str,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        agent_id: str | None = None,
    ) -> ToolsPage: ...

    def call_tool(
        self,
        server_id: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        agent_id: str | None = None,
    ) -> ToolCallResult: ...


@runtime_checkable
class AsyncClientProtocol(Protocol):
    """Protocol for asynchronous MCP clients.

    Concrete implementations should provide async listing, pagination, tool
    invocation, and SSE streaming helpers.
    """
    @classmethod
    async def from_config(
        cls,
        config: McpClientConfig,
        *,
        agent_policies: Optional[PolicyMap] = None,
        gateway_mgmt_client: Optional[httpx.Client] = None,
        gateway_http_client: Optional[httpx.AsyncClient] = None,
        gateway_sse_client: Optional[httpx.AsyncClient] = None,
    ) -> Self: ...

    async def list_servers(self, *, agent_id: str | None = None) -> List[McpServerRef]: ...

    async def list_tools(self, server_id: str, *, agent_id: str | None = None) -> List[McpTool]: ...

    async def list_tools_page(
        self,
        server_id: str,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        agent_id: str | None = None,
    ) -> ToolsPage: ...

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        agent_id: str | None = None,
    ) -> ToolCallResult: ...

    # Note: declared as async def returning AsyncIterator since concrete client methods are async wrappers
    # around strategy generators. Strategy Protocol methods are regular def returning AsyncIterator to
    # model async generators per MyPy guidance. Implementations should ensure compatibility.
    async def stream_events(
        self,
        server_id: str,
        path: str = "/sse",
        *,
        reconnect: bool = False,
        max_retries: int = 3,
        backoff_initial: float = 0.5,
        backoff_factor: float = 2.0,
        backoff_max: float = 8.0,
        idle_timeout: Optional[float] = None,
        max_total_seconds: Optional[float] = None,
    ) -> AsyncIterator[str]: ...

    # Note: same rationale as stream_events; returns an AsyncIterator of parsed dict events.
    async def stream_events_parsed(
        self,
        server_id: str,
        path: str = "/sse",
        *,
        reconnect: bool = False,
        max_retries: int = 3,
        backoff_initial: float = 0.5,
        backoff_factor: float = 2.0,
        backoff_max: float = 8.0,
        idle_timeout: Optional[float] = None,
        max_total_seconds: Optional[float] = None,
    ) -> AsyncIterator[Dict[str, Any]]: ...


class ClientCommonMixin:
    """Shared policy helpers for sync/async client facades."""

    def _filter_servers_by_policy(
        self,
        servers: List[McpServerRef],
        policy: Optional[ToolPolicy],
    ) -> List[McpServerRef]:
        """Filter servers by `policy.allowed_servers` if configured.

        Args:
            servers: The list of discovered `McpServerRef`.
            policy: Optional `ToolPolicy` to apply.

        Returns:
            The filtered list of servers. If no policy or no allow-list is set, returns the original list.
        """
        if not policy or policy.allowed_servers is None:
            return servers
        allowed = policy.allowed_servers
        return [s for s in servers if s.id in allowed]

    def _filter_tools_by_policy(
        self,
        tools: List[McpTool],
        policy: Optional[ToolPolicy],
    ) -> List[McpTool]:
        """Apply allowed_tools and read-only filtering to a tool list.

        Args:
            tools: The tool list to filter.
            policy: Optional `ToolPolicy` whose `allowed_tools` and `read_only` fields are honored.

        Returns:
            The filtered tool list.
        """
        if not policy:
            return tools
        res = tools
        if policy.allowed_tools is not None:
            res = [t for t in res if t.name in policy.allowed_tools]
        if policy.read_only:
            res = [t for t in res if not (t.mutating or is_mutating_tool_name(t.name))]
        return res

    def _should_block_read_only(
        self,
        policy: Optional[ToolPolicy],
        tool_name: str,
        listed_lookup: Optional[Dict[str, McpTool]] = None,
    ) -> bool:
        """Return True if a call to `tool_name` should be blocked by read-only policy.

        Args:
            policy: Optional `ToolPolicy` to check.
            tool_name: The tool name being invoked.
            listed_lookup: Optional mapping of tool name to `McpTool` metadata to improve mutating detection.

        Returns:
            True if the call should be blocked due to read-only policy; otherwise False.
        """
        if not policy or not policy.read_only:
            return False
        if listed_lookup is not None:
            t = listed_lookup.get(tool_name)
            if t is not None and t.mutating:
                return True
        return is_mutating_tool_name(tool_name)
