"""Synchronous and asynchronous MCP info provider facades.

This module defines the concrete sync/async MCP info provider classes that
implement the :mod:`gearmeshing_ai.info_provider.mcp.base` protocols.

* :class:`MCPInfoProvider` – sync facade for listing MCP endpoints/tools and
  invoking tools over one or more underlying sync strategies (direct/gateway).
* :class:`AsyncMCPInfoProvider` – async counterpart that also supports
  streaming SSE events.

Typical usage:
    cfg = McpClientConfig(...)
    provider = MCPInfoProvider.from_config(cfg)
    tools = provider.list_tools("server-id")
    res = provider.call_tool("server-id", "echo", {"text": "hi"})

Ad an AI agent, in generally, it would initial a MCP connection object with MCP server endpoint, and instantiate the MCP
connection object to build connection with MCP servers and list all the MCP tools. And would use the MCP tools to set to
the AI agent. So as an object for how to provide the necessary info about MCP tools for AI agent. I think this object just
need to provide 2 points: the MCP server endpoints and how to get the MCP tools by the AI agent framework.

Here are the architecture as the relationship of the AI agent and the objects:


+-------------------+
|  Import AI agent  |
+-------------------+                                                                                   Get the MCP info
      |                                              Provide the MCP                                 (directly from config or
      |      +---------------------------------+     info like endpoints      +--------------------+    gateway service)     +---------------+
      +----->|  AI agent MCP tools connection  | <----------------------------|  MCP info provider | ----------------------> |  MCP servers  |
      |      +---------------------------------+                              +--------------------+                         +---------------+
      |           | Use the MCP info
      |           | like tools array   +------------+
      +-----------+------------------> |  AI agent  |
                                       +------------+

From the above architecture and the brief workflow, we could make sure the MCP info provider could be very easily and
simple: provide the MCP info like endpoint and tools is enough to use.

```python
mcp_info_provider = MCPInfoProvider(strategy=StrategyImplementation())

mcp_servers = mcp_info_provider.servers
print(f"MCP servers: {mcp_servers}")    # {"clickup_mcp": {endpoint: "http://clickup-mcp:8082/mcp/strable-http", tools: [{"name": "task.get", }]}}

all_mcp_endpoints = [v.endpoint for k, v in mcp_info_provider.servers]

# Here just demonstrate how a AI agent would be configured some MCP tools
mcp_tools = TestTools(
    endpoints=all_mcp_endpoints,
)
TestAIAgent(
    name="Test AI agent",
    system_prompt="You are a poor software engineer. And you focus on Python."
    tools=mcp_tools.list_tools,
)
```
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

import httpx

from .base import BaseAsyncMCPInfoProvider, BaseMCPInfoProvider, ClientCommonMixin
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
from .strategy import DirectMcpStrategy, GatewayMcpStrategy
from .strategy.base import AsyncStrategy, SyncStrategy, is_mutating_tool_name
from .strategy.direct_async import AsyncDirectMcpStrategy
from .strategy.gateway_async import AsyncGatewayMcpStrategy

logger = logging.getLogger(__name__)


class AsyncMCPInfoProvider(ClientCommonMixin, BaseAsyncMCPInfoProvider):
    """Async MCP info provider facade.

    Delegates to one or more :class:`AsyncStrategy` implementations
    (direct/gateway) and enforces optional access policies.
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
    ) -> "AsyncMCPInfoProvider":
        """Construct an async info provider from ``McpClientConfig``.

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
            An initialized :class:`AsyncMCPInfoProvider`.
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

    async def get_endpoints(self, *, agent_id: str | None = None) -> List[McpServerRef]:
        """Return discovered MCP endpoints across async strategies, honoring policy.

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
                logger.debug("get_endpoints error from %s: %s", type(strat).__name__, e)

        if agent_id and agent_id in self._policies:
            policy = self._policies[agent_id]
            if policy.allowed_servers is not None:
                servers = [s for s in servers if s.id in policy.allowed_servers]

        return servers

    # Back-compat alias used by existing callers/tests; delegates to get_endpoints.
    async def list_servers(
        self, *, agent_id: str | None = None
    ) -> List[McpServerRef]:  # pragma: no cover - thin wrapper
        return await self.get_endpoints(agent_id=agent_id)

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


class MCPInfoProvider(ClientCommonMixin, BaseMCPInfoProvider):
    """High-level MCP info provider facade.

    Delegates operations to one or more :class:`SyncStrategy` implementations
    and applies access policies when provided.
    """

    def __init__(
        self,
        *,
        strategies: Iterable[SyncStrategy],
        agent_policies: Optional[PolicyMap] = None,
    ) -> None:
        self._strategies: List[SyncStrategy] = list(strategies)
        # runtime validation (optional)
        for s in self._strategies:
            if not isinstance(s, SyncStrategy):
                raise TypeError(f"Strategy {type(s).__name__} does not conform to SyncStrategy protocol")
        self._policies = agent_policies or {}

    @classmethod
    def from_config(
        cls,
        config: McpClientConfig,
        *,
        agent_policies: Optional[PolicyMap] = None,
        direct_http_client: Optional[httpx.Client] = None,
        gateway_mgmt_client: Optional[httpx.Client] = None,
        gateway_http_client: Optional[httpx.Client] = None,
    ) -> "MCPInfoProvider":
        """Construct a sync info provider from ``McpClientConfig``.

        - Enables Direct strategy when `servers` are configured.
        - Enables Gateway strategy when `gateway` config is present.
        - Custom httpx clients may be provided for advanced settings.

        Args:
            config: The high-level MCP client configuration.
            agent_policies: Optional per-agent access policies.
            direct_http_client: Optional httpx.Client for direct strategy.
            gateway_mgmt_client: Optional httpx.Client for Gateway management API.
            gateway_http_client: Optional httpx.Client for Gateway HTTP endpoints.

        Returns:
            An initialized :class:`MCPInfoProvider`.
        """
        strategies: List[SyncStrategy] = []
        if config.servers:
            strategies.append(
                DirectMcpStrategy(
                    config.servers,
                    client=direct_http_client,
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
                GatewayMcpStrategy(
                    gw,
                    client=gateway_http_client,
                    ttl_seconds=config.tools_cache_ttl_seconds,
                )
            )
        return cls(strategies=strategies, agent_policies=agent_policies)

    def get_endpoints(self, *, agent_id: str | None = None) -> List[McpServerRef]:
        """Return discovered MCP endpoints across strategies, honoring policy.

        If `agent_id` is provided and policies are configured, filters servers by
        the agent's allowed server list (if present).

        Args:
            agent_id: Optional agent identifier used for policy filtering.

        Returns:
            A list of `McpServerRef` discovered across strategies after filtering.
        """
        servers: List[McpServerRef] = []
        for strat in self._strategies:
            try:
                logger.debug("McpClient.get_endpoints: using %s", type(strat).__name__)
                servers.extend(list(strat.list_servers()))
            except Exception as e:
                logger.debug("get_endpoints error from %s: %s", type(strat).__name__, e)
        if agent_id and agent_id in self._policies:
            policy = self._policies[agent_id]
            before = len(servers)
            servers = self._filter_servers_by_policy(servers, policy)
            if before != len(servers):
                logger.debug(
                    "McpClient.get_endpoints: policy filtered endpoints for agent=%s from %d to %d",
                    agent_id,
                    before,
                    len(servers),
                )
        return servers

    # Back-compat alias used by existing callers/tests; delegates to get_endpoints.
    def list_servers(self, *, agent_id: str | None = None) -> List[McpServerRef]:  # pragma: no cover - thin wrapper
        return self.get_endpoints(agent_id=agent_id)

    def list_tools(self, server_id: str, *, agent_id: str | None = None) -> List[McpTool]:
        """List tools for a specific server, applying policy filters.

        Raises `ToolAccessDeniedError` if the agent is not permitted to access
        the server per policy. If tools are found from a strategy, returns the
        first non-empty result.

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
                logger.debug("McpClient.list_tools: using %s for server_id=%s", type(strat).__name__, server_id)
                tools: List[McpTool] = list(strat.list_tools(server_id))
                if tools:
                    if agent_id and agent_id in self._policies:
                        policy = self._policies[agent_id]
                        before = len(tools)
                        tools = self._filter_tools_by_policy(tools, policy)
                        if before != len(tools):
                            logger.debug(
                                "McpClient.list_tools: policy filtered tools for agent=%s from %d to %d",
                                agent_id,
                                before,
                                len(tools),
                            )
                    return tools
            except Exception as e:
                logger.debug("list_tools error from %s: %s", type(strat).__name__, e)
        raise ServerNotFoundError(server_id)

    def list_tools_page(
        self,
        server_id: str,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        agent_id: str | None = None,
    ) -> ToolsPage:
        """List tools with pagination when supported by the strategy/backend.

        Applies server/tool filters based on policy when `agent_id` is provided.
        Falls back to non-paginated `list_tools` when pagination is unavailable.

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
        if agent_id and agent_id in self._policies:
            policy = self._policies[agent_id]
            if policy.allowed_servers is not None and server_id not in policy.allowed_servers:
                raise ToolAccessDeniedError(agent_id, server_id, "<list_tools_page>")
        for strat in self._strategies:
            try:
                fn = getattr(strat, "list_tools_page", None)
                if fn is not None:
                    page: ToolsPage = fn(server_id, cursor=cursor, limit=limit)
                    if agent_id and agent_id in self._policies:
                        policy = self._policies[agent_id]
                        items = self._filter_tools_by_policy(page.items, policy)
                        return ToolsPage(items=items, next_cursor=page.next_cursor)
                    return page
            except Exception as e:
                logger.debug("list_tools_page error from %s: %s", type(strat).__name__, e)
        tools = self.list_tools(server_id, agent_id=agent_id)
        return ToolsPage(items=tools, next_cursor=None)

    def call_tool(
        self,
        server_id: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        agent_id: str | None = None,
    ) -> ToolCallResult:
        """Invoke a tool through the first responding strategy.

        Enforces policy (server access, tool allow-list, and read-only).

        Args:
            server_id: Target server identifier.
            tool_name: Tool identifier to invoke.
            args: Tool arguments.
            agent_id: Optional agent identifier used for policy enforcement.

        Returns:
            A `ToolCallResult` describing the outcome.

        Raises:
            ToolAccessDeniedError: If access is blocked by policy.
            ServerNotFoundError: If no strategy can handle the server.
        """
        enforce_policy(agent_id=agent_id, server_id=server_id, tool_name=tool_name, policies=self._policies)
        if agent_id and agent_id in self._policies:
            policy = self._policies[agent_id]
            try:
                listed = {t.name: t for t in self.list_tools(server_id, agent_id=agent_id)}
            except Exception:
                listed = None
            if self._should_block_read_only(policy, tool_name, listed):
                raise ToolAccessDeniedError(agent_id, server_id, tool_name)
        for strat in self._strategies:
            try:
                logger.debug(
                    "McpClient.call_tool: strategy=%s server_id=%s tool=%s agent=%s",
                    type(strat).__name__,
                    server_id,
                    tool_name,
                    agent_id,
                )
                return strat.call_tool(
                    server_id,
                    tool_name,
                    args,
                    agent_id=agent_id,
                )
            except Exception as e:
                logger.debug("call_tool error from %s: %s", type(strat).__name__, e)
        raise ServerNotFoundError(server_id)
