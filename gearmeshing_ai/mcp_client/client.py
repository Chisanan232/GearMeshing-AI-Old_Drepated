from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional

import httpx

from .errors import ServerNotFoundError, ToolAccessDeniedError
from .gateway_api.client import GatewayApiClient
from .policy import PolicyMap, enforce_policy
from .schemas.config import McpClientConfig
from .schemas.core import McpServerRef, McpTool, ToolCallResult
from .strategy.base import SyncStrategy
from .strategy.direct import DirectMcpStrategy
from .strategy.gateway import GatewayMcpStrategy

logger = logging.getLogger(__name__)


class McpClient:
    """
    High-level MCP client facade that delegates to one or more strategies and
    applies access policies.
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
            if not isinstance(s, SyncStrategy):  # type: ignore[arg-type]
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
    ) -> "McpClient":
        strategies: List[object] = []
        if config.servers:
            strategies.append(
                DirectMcpStrategy(
                    config.servers,
                    client=direct_http_client,
                    ttl_seconds=getattr(config, "tools_cache_ttl_seconds", 10.0),
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
                    ttl_seconds=getattr(config, "tools_cache_ttl_seconds", 10.0),
                )
            )
        return cls(strategies=strategies, agent_policies=agent_policies)

    def list_servers(self, *, agent_id: str | None = None) -> List[McpServerRef]:
        servers: List[McpServerRef] = []
        for strat in self._strategies:
            if hasattr(strat, "list_servers"):
                try:
                    logger.debug("McpClient.list_servers: using %s", type(strat).__name__)
                    servers.extend(list(strat.list_servers()))
                except Exception as e:
                    logger.debug("list_servers error from %s: %s", type(strat).__name__, e)
        # Apply policy filter by servers if provided
        if agent_id and agent_id in self._policies:
            policy = self._policies[agent_id]
            if policy.allowed_servers is not None:
                before = len(servers)
                servers = [s for s in servers if s.id in policy.allowed_servers]
                logger.debug(
                    "McpClient.list_servers: policy filtered servers for agent=%s from %d to %d",
                    agent_id,
                    before,
                    len(servers),
                )
        return servers

    def list_tools(self, server_id: str, *, agent_id: str | None = None) -> List[McpTool]:
        # Enforce server access policy first
        if agent_id and agent_id in self._policies:
            policy = self._policies[agent_id]
            if policy.allowed_servers is not None and server_id not in policy.allowed_servers:
                raise ToolAccessDeniedError(agent_id, server_id, "<list_tools>")
        for strat in self._strategies:
            if hasattr(strat, "list_tools"):
                try:
                    logger.debug("McpClient.list_tools: using %s for server_id=%s", type(strat).__name__, server_id)
                    tools: List[McpTool] = list(strat.list_tools(server_id))
                    if tools:
                        # Apply tool filter by policy if provided
                        if agent_id and agent_id in self._policies:
                            policy = self._policies[agent_id]
                            if policy.allowed_tools is not None:
                                before = len(tools)
                                tools = [t for t in tools if t.name in policy.allowed_tools]
                                logger.debug(
                                    "McpClient.list_tools: policy filtered tools for agent=%s from %d to %d (allow-list)",
                                    agent_id,
                                    before,
                                    len(tools),
                                )
                            if policy.read_only:
                                before = len(tools)

                                def _is_mut(t: McpTool) -> bool:
                                    # Prefer explicit metadata; fallback to heuristic for older servers
                                    return bool(getattr(t, "mutating", False) or self._is_mutating_tool_name(t.name))

                                tools = [t for t in tools if not _is_mut(t)]
                                logger.debug(
                                    "McpClient.list_tools: policy filtered tools for agent=%s from %d to %d (read-only)",
                                    agent_id,
                                    before,
                                    len(tools),
                                )
                        return tools
                except Exception as e:
                    logger.debug("list_tools error from %s: %s", type(strat).__name__, e)
        raise ServerNotFoundError(server_id)

    def call_tool(
        self,
        server_id: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        agent_id: str | None = None,
    ) -> ToolCallResult:
        enforce_policy(agent_id=agent_id, server_id=server_id, tool_name=tool_name, policies=self._policies)
        # Enforce read-only: block mutating tools
        if agent_id and agent_id in self._policies and self._policies[agent_id].read_only:
            # If we have metadata from listing, prefer it; otherwise fallback to heuristic
            try:
                listed = {t.name: t for t in self.list_tools(server_id, agent_id=agent_id)}
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

    @staticmethod
    def _is_mutating_tool_name(name: str) -> bool:
        n = name.lower()
        prefixes = ("create", "update", "delete", "remove", "post_", "put_", "patch_", "write", "set_")
        return n.startswith(prefixes)
