"""Asynchronous MCP client facade.

Provides a high-level async API for listing tools, invoking tools, and
optionally streaming events using one or more underlying async strategies
(direct and/or gateway). Applies optional policy enforcement.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

import httpx

from .base import AsyncClientProtocol, ClientCommonMixin
from .errors import ServerNotFoundError, ToolAccessDeniedError
from .gateway_api.client import GatewayApiClient
from .policy import PolicyMap, enforce_policy
from .schemas.config import McpClientConfig
from .schemas.core import (
    McpServerRef,
    McpTool,
    ServerKind,
    ToolCallResult,
    ToolsPage,
    TransportType,
)
from .strategy.base import AsyncStrategy, is_mutating_tool_name
from .strategy.direct_async import AsyncDirectMcpStrategy
from .strategy.gateway_async import AsyncGatewayMcpStrategy

logger = logging.getLogger(__name__)


class AsyncMcpClient(ClientCommonMixin, AsyncClientProtocol):
    """Async facade for MCP client operations.

    Delegates to one or more `AsyncStrategy` implementations (direct/gateway)
    and enforces optional access policies.
    """

    def __init__(
        self,
        *,
        strategies: Iterable[AsyncStrategy],
        agent_policies: Optional[PolicyMap] = None,
    ) -> None:
        self._strategies: List[AsyncStrategy] = list(strategies)
        for s in self._strategies:
            if not isinstance(s, AsyncStrategy):
                raise TypeError(f"Strategy {type(s).__name__} does not conform to AsyncStrategy protocol")
        self._policies = agent_policies or {}

    @classmethod
    async def from_config(
        cls,
        config: McpClientConfig,
        *,
        agent_policies: Optional[PolicyMap] = None,
        gateway_mgmt_client: Optional[httpx.Client] = None,
        gateway_http_client: Optional[httpx.AsyncClient] = None,
        gateway_sse_client: Optional[httpx.AsyncClient] = None,
    ) -> "AsyncMcpClient":
        """Construct an async client from `McpClientConfig`.

        - Enables AsyncDirect strategy when `servers` are configured.
        - Enables AsyncGateway strategy when `gateway` config is present.
        - Allows providing custom httpx clients for tuning timeouts/proxies.

        Args:
            config: The high-level MCP client configuration.
            agent_policies: Optional per-agent access policies.
            gateway_mgmt_client: Optional sync httpx.Client for Gateway management API.
            gateway_http_client: Optional async httpx.AsyncClient for Gateway HTTP endpoints.
            gateway_sse_client: Optional async httpx.AsyncClient for Gateway SSE endpoints.

        Returns:
            An initialized `AsyncMcpClient`.
        """
        strategies: List[AsyncStrategy] = []
        if config.servers:
            strategies.append(
                AsyncDirectMcpStrategy(
                    list(config.servers),
                    ttl_seconds=config.tools_cache_ttl_seconds,
                )
            )
        if config.gateway is not None:
            gw = GatewayApiClient(
                config.gateway.base_url,
                auth_token=config.gateway.auth_token,
                client=gateway_mgmt_client,
                timeout=config.gateway.request_timeout_seconds,
            )
            strategies.append(
                AsyncGatewayMcpStrategy(
                    gw,
                    client=gateway_http_client,
                    ttl_seconds=config.tools_cache_ttl_seconds,
                    sse_client=gateway_sse_client,
                )
            )
        return cls(strategies=strategies, agent_policies=agent_policies)

    async def list_tools(self, server_id: str, *, agent_id: str | None = None) -> List[McpTool]:
        """List tools for a specific server, applying policy filters.

        Args:
            server_id: Target server identifier.
            agent_id: Optional agent identifier used for policy enforcement.

        Returns:
            A list of `McpTool`.

        Raises:
            ToolAccessDeniedError: If access is blocked by policy.
            ServerNotFoundError: If no strategy can serve the server.
        """
        # Enforce server access policy first
        if agent_id and agent_id in self._policies:
            policy = self._policies[agent_id]
            if policy.allowed_servers is not None and server_id not in policy.allowed_servers:
                raise ToolAccessDeniedError(agent_id, server_id, "<list_tools>")
        for strat in self._strategies:
            try:
                logger.debug("AsyncMcpClient.list_tools: using %s for server_id=%s", type(strat).__name__, server_id)
                tools: List[McpTool] = await strat.list_tools(server_id)
                if tools:
                    if agent_id and agent_id in self._policies:
                        policy = self._policies[agent_id]
                        if policy.allowed_tools is not None:
                            tools = [t for t in tools if t.name in policy.allowed_tools]
                        if policy.read_only:

                            def _is_mut(t: McpTool) -> bool:
                                return bool(getattr(t, "mutating", False) or is_mutating_tool_name(t.name))

                            tools = [t for t in tools if not _is_mut(t)]
                    return tools
            except Exception as e:
                logger.debug("list_tools error from %s: %s", type(strat).__name__, e)
        raise ServerNotFoundError(server_id)

    async def list_servers(self, *, agent_id: str | None = None) -> List[McpServerRef]:
        """Return discovered servers across async strategies, honoring policy.

        Aggregates servers from configured async strategies. For Direct strategies,
        reads configured ServerConfig entries. For Gateway strategies, uses the
        Gateway management client to list servers.
        """
        servers: List[McpServerRef] = []
        for strat in self._strategies:
            try:
                # Direct: introspect configured ServerConfig entries
                if isinstance(strat, AsyncDirectMcpStrategy):
                    for sc in getattr(strat, "_servers", []) or []:
                        servers.append(
                            McpServerRef(
                                id=sc.name,
                                display_name=sc.name,
                                kind=ServerKind.DIRECT,
                                transport=TransportType.STREAMABLE_HTTP,
                                endpoint_url=sc.endpoint_url,
                                auth_token=sc.auth_token,
                            )
                        )
                # Gateway: use GatewayApiClient to discover servers
                elif isinstance(strat, AsyncGatewayMcpStrategy):
                    gw = getattr(strat, "_gateway", None)
                    if gw is not None:
                        for srv in gw.list_servers():
                            servers.append(
                                McpServerRef(
                                    id=srv.id,
                                    display_name=srv.name,
                                    kind=ServerKind.GATEWAY,
                                    transport=TransportType.STREAMABLE_HTTP,
                                    endpoint_url=f"{gw.base_url}/servers/{srv.id}/mcp",
                                    auth_token=gw.auth_token,
                                )
                            )
            except Exception as e:
                logger.debug("list_servers error from %s: %s", type(strat).__name__, e)

        if agent_id and agent_id in self._policies:
            policy = self._policies[agent_id]
            if policy.allowed_servers is not None:
                servers = [s for s in servers if s.id in policy.allowed_servers]

        return servers

    async def list_tools_page(
        self,
        server_id: str,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        agent_id: str | None = None,
    ) -> ToolsPage:
        """List tools with pagination when supported by the strategy/backend.

        Applies filters based on policy when `agent_id` is provided. Falls back
        to non-paginated `list_tools` when pagination is unavailable.

        Args:
            server_id: Target server identifier.
            cursor: Cursor from a previous page.
            limit: Max items per page.
            agent_id: Optional agent identifier used for policy enforcement.

        Returns:
            A `ToolsPage` with `items` and optional `next_cursor`.

        Raises:
            ToolAccessDeniedError: If access is blocked by policy.
            ServerNotFoundError: If pagination unsupported and no tools found.
        """
        # Enforce server access policy first
        if agent_id and agent_id in self._policies:
            policy = self._policies[agent_id]
            if policy.allowed_servers is not None and server_id not in policy.allowed_servers:
                raise ToolAccessDeniedError(agent_id, server_id, "<list_tools_page>")
        for strat in self._strategies:
            try:
                fn = getattr(strat, "list_tools_page", None)
                if fn is not None:
                    page: ToolsPage = await fn(server_id, cursor=cursor, limit=limit)
                    if agent_id and agent_id in self._policies:
                        policy = self._policies[agent_id]
                        items = page.items
                        if policy.allowed_tools is not None:
                            items = [t for t in items if t.name in policy.allowed_tools]
                        if policy.read_only:
                            items = [t for t in items if not (getattr(t, "mutating", False))]
                        return ToolsPage(items=items, next_cursor=page.next_cursor)
                    return page
            except Exception as e:
                logger.debug("list_tools_page error from %s: %s", type(strat).__name__, e)
        tools = await self.list_tools(server_id, agent_id=agent_id)
        return ToolsPage(items=tools, next_cursor=None)

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
    ) -> AsyncIterator[str]:
        """Yield raw SSE lines by delegating to the first strategy that supports it.

        Args:
            server_id: Target server identifier.
            path: SSE path relative to the server base.
            reconnect: Whether to reconnect automatically.
            max_retries: Max reconnection attempts.
            backoff_initial: Initial backoff in seconds.
            backoff_factor: Exponential backoff factor.
            backoff_max: Max backoff in seconds.
            idle_timeout: Optional idle-timeout for the connection.
            max_total_seconds: Optional cap on total streaming time.

        Returns:
            An async iterator of raw SSE lines (decoded strings).

        Raises:
            ServerNotFoundError: If no strategy is available.
        """
        for strat in self._strategies:
            return strat.stream_events(
                server_id,
                path,
                reconnect=reconnect,
                max_retries=max_retries,
                backoff_initial=backoff_initial,
                backoff_factor=backoff_factor,
                backoff_max=backoff_max,
                idle_timeout=idle_timeout,
                max_total_seconds=max_total_seconds,
            )
        raise ServerNotFoundError(server_id)

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
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield parsed SSE events as dictionaries by delegating to a strategy.

        Args:
            server_id: Target server identifier.
            path: SSE path relative to the server base.
            reconnect: Whether to reconnect automatically.
            max_retries: Max reconnection attempts.
            backoff_initial: Initial backoff in seconds.
            backoff_factor: Exponential backoff factor.
            backoff_max: Max backoff in seconds.
            idle_timeout: Optional idle-timeout for the connection.
            max_total_seconds: Optional cap on total streaming time.

        Returns:
            An async iterator of parsed event dictionaries.

        Raises:
            ServerNotFoundError: If no strategy is available.
        """
        for strat in self._strategies:
            return strat.stream_events_parsed(
                server_id,
                path,
                reconnect=reconnect,
                max_retries=max_retries,
                backoff_initial=backoff_initial,
                backoff_factor=backoff_factor,
                backoff_max=backoff_max,
                idle_timeout=idle_timeout,
                max_total_seconds=max_total_seconds,
            )
        raise ServerNotFoundError(server_id)

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        agent_id: str | None = None,
    ) -> ToolCallResult:
        """Invoke a tool through the first async strategy that succeeds.

        Enforces policy checks (read-only, allow-lists).

        Args:
            server_id: Target server identifier.
            tool_name: Tool identifier to invoke.
            args: Tool arguments.
            agent_id: Optional agent identifier used for policy enforcement.

        Returns:
            A `ToolCallResult` describing the outcome.

        Raises:
            ToolAccessDeniedError: If access is blocked by policy.
            ServerNotFoundError: If no strategy can serve the server.
        """
        enforce_policy(agent_id=agent_id, server_id=server_id, tool_name=tool_name, policies=self._policies)
        if agent_id and agent_id in self._policies and self._policies[agent_id].read_only:
            # Prefer metadata from list_tools if available
            try:
                listed = {t.name: t for t in await self.list_tools(server_id, agent_id=agent_id)}
                t = listed.get(tool_name)
                is_mut = bool(getattr(t, "mutating", False)) if t else is_mutating_tool_name(tool_name)
            except Exception:
                is_mut = is_mutating_tool_name(tool_name)
            if is_mut:
                raise ToolAccessDeniedError(agent_id, server_id, tool_name)
        for strat in self._strategies:
            try:
                logger.debug(
                    "AsyncMcpClient.call_tool: strategy=%s server_id=%s tool=%s agent=%s",
                    type(strat).__name__,
                    server_id,
                    tool_name,
                    agent_id,
                )
                return await strat.call_tool(
                    server_id,
                    tool_name,
                    args,
                )
            except Exception as e:
                logger.debug("call_tool error from %s: %s", type(strat).__name__, e)
        raise ServerNotFoundError(server_id)
