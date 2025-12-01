from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional

import httpx

from .errors import ServerNotFoundError, ToolAccessDeniedError
from .gateway_api.client import GatewayApiClient
from .policy import PolicyMap, enforce_policy
from .schemas.config import McpClientConfig
from .schemas.core import McpTool, ToolCallResult
from .strategy.base import AsyncStrategy
from .strategy.gateway_async import AsyncGatewayMcpStrategy

logger = logging.getLogger(__name__)


class AsyncMcpClient:
    """
    Async facade for MCP client operations, focusing on Gateway-based HTTP usage.

    Provides async list_tools/call_tool via AsyncGatewayMcpStrategy and enforces policies.
    """

    def __init__(
        self,
        *,
        strategies: Iterable[AsyncStrategy],
        agent_policies: Optional[PolicyMap] = None,
    ) -> None:
        self._strategies: List[AsyncStrategy] = list(strategies)
        for s in self._strategies:
            if not isinstance(s, AsyncStrategy):  # type: ignore[arg-type]
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
        strategies: List[object] = []
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
                    ttl_seconds=getattr(config, "tools_cache_ttl_seconds", 10.0),
                    sse_client=gateway_sse_client,
                )
            )
        return cls(strategies=strategies, agent_policies=agent_policies)

    async def list_tools(self, server_id: str, *, agent_id: str | None = None) -> List[McpTool]:
        # Enforce server access policy first
        if agent_id and agent_id in self._policies:
            policy = self._policies[agent_id]
            if policy.allowed_servers is not None and server_id not in policy.allowed_servers:
                raise ToolAccessDeniedError(agent_id, server_id, "<list_tools>")
        for strat in self._strategies:
            if hasattr(strat, "list_tools"):
                try:
                    logger.debug(
                        "AsyncMcpClient.list_tools: using %s for server_id=%s", type(strat).__name__, server_id
                    )
                    tools: List[McpTool] = await strat.list_tools(server_id)
                    if tools:
                        if agent_id and agent_id in self._policies:
                            policy = self._policies[agent_id]
                            if policy.allowed_tools is not None:
                                tools = [t for t in tools if t.name in policy.allowed_tools]
                            if policy.read_only:

                                def _is_mut(t: McpTool) -> bool:
                                    return bool(getattr(t, "mutating", False) or self._is_mutating_tool_name(t.name))

                                tools = [t for t in tools if not _is_mut(t)]
                        return tools
                except Exception as e:
                    logger.debug("list_tools error from %s: %s", type(strat).__name__, e)
        raise ServerNotFoundError(server_id)

    @staticmethod
    def _is_mutating_tool_name(name: str) -> bool:
        n = name.lower()
        prefixes = ("create", "update", "delete", "remove", "post_", "put_", "patch_", "write", "set_")
        return n.startswith(prefixes)

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
        idle_timeout: float | None = None,
        max_total_seconds: float | None = None,
    ):
        for strat in self._strategies:
            if hasattr(strat, "stream_events"):
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
        idle_timeout: float | None = None,
        max_total_seconds: float | None = None,
    ):
        for strat in self._strategies:
            if hasattr(strat, "stream_events_parsed"):
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
        enforce_policy(agent_id=agent_id, server_id=server_id, tool_name=tool_name, policies=self._policies)
        if agent_id and agent_id in self._policies and self._policies[agent_id].read_only:
            # Prefer metadata from list_tools if available
            try:
                listed = {t.name: t for t in await self.list_tools(server_id, agent_id=agent_id)}
                t = listed.get(tool_name)
                is_mut = bool(getattr(t, "mutating", False)) if t else self._is_mutating_tool_name(tool_name)
            except Exception:
                is_mut = self._is_mutating_tool_name(tool_name)
            if is_mut:
                raise ToolAccessDeniedError(agent_id, server_id, tool_name)
        for strat in self._strategies:
            if hasattr(strat, "call_tool"):
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
