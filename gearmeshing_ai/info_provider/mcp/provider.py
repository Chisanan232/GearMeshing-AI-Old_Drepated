"""Synchronous and asynchronous MCP info provider facades.

This module defines the concrete sync/async MCP info provider classes that
implement the :mod:`gearmeshing_ai.info_provider.mcp.base` protocols.

The providers intentionally expose an **info-only** surface focused on tool
discovery for already-known servers. Transport details, tool invocation, and
streaming remain responsibilities of lower-level strategy/client layers.

* :class:`MCPInfoProvider` – sync facade for listing MCP tools (and optional
  paginated tool metadata) over one or more underlying sync strategies
  (direct/gateway).
* :class:`AsyncMCPInfoProvider` – async counterpart for the same listing
  operations.

Typical usage:
    cfg = McpClientConfig(...)
    provider = MCPInfoProvider.from_config(cfg)
    tools = provider.list_tools("server-id")

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
from typing import Iterable, List, Optional

import httpx

from .base import BaseAsyncMCPInfoProvider, BaseMCPInfoProvider, ClientCommonMixin
from .errors import ServerNotFoundError, ToolAccessDeniedError
from .gateway_api.client import GatewayApiClient
from .policy import PolicyMap
from .schemas.config import McpClientConfig
from .schemas.core import (
    McpTool,
    ToolsPage,
)
from .strategy import DirectMcpStrategy, GatewayMcpStrategy
from .strategy.base import AsyncStrategy, SyncStrategy, is_mutating_tool_name
from .strategy.direct_async import AsyncDirectMcpStrategy
from .strategy.gateway_async import AsyncGatewayMcpStrategy
from .transport.mcp import SseMCPTransport

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
                    # TODO: It should extract this property as parameter
                    mcp_transport=SseMCPTransport(),
                ),
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

    # Note: AsyncMCPInfoProvider is intentionally *tools-only*; callers should
    # use lower-level strategies/clients for tool invocation and streaming.


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
                    ttl_seconds=config.tools_cache_ttl_seconds,
                    # TODO: It should extract this property as parameter
                    mcp_transport=SseMCPTransport(),
                ),
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

    # Note: MCPInfoProvider is intentionally *tools-only*; callers should rely
    # on lower-level strategies or higher-level clients for invocation.
