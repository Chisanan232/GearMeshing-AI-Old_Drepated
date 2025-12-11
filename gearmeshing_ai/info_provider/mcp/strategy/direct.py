"""Direct MCP strategy (sync)

This module implements the synchronous strategy for directly connecting to MCP
servers over HTTP. It provides a simple, cache-aware implementation for
listing tools and invoking tools against a configured set of servers defined
in `ServerConfig`.

Targets and use-cases:
- Applications that already know their MCP server endpoints and want to bypass
  the Gateway management layer.
- Local development scenarios where the MCP server is running on localhost or
  within the same network.

Usage (typical):
    from gearmeshing_ai.mcp_client.schemas.config import ServerConfig
    from gearmeshing_ai.mcp_client.strategy.direct import DirectMcpStrategy

    strat = DirectMcpStrategy([
        ServerConfig(name="github-mcp", endpoint_url="http://localhost:8000/mcp/"),
    ])

    servers = list(strat.list_servers())
    tools = list(strat.list_tools("github-mcp"))
    result = strat.call_tool("github-mcp", "echo", {"text": "hello"})

Guidelines:
- Configure `ttl_seconds` to balance freshness vs performance for tools list.
- Provide an `httpx.Client` instance when you need custom timeouts, proxies, or
  connection pooling settings. Otherwise a sane default client is created.
- The strategy uses simple heuristics to detect mutating tools; if a tool call
  is considered mutating and succeeds, the tools cache for that server is
  invalidated.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

from gearmeshing_ai.info_provider.mcp.schemas.config import ServerConfig
from gearmeshing_ai.info_provider.mcp.schemas.core import (
    McpServerRef,
    McpTool,
    ServerKind,
    ToolCallResult,
    TransportType,
)

from .base import StrategyCommonMixin, SyncStrategy


class DirectMcpStrategy(StrategyCommonMixin, SyncStrategy):
    """Synchronous strategy for direct HTTP access to MCP servers.

    - Discovers servers from the provided `ServerConfig` list only.
    - Fetches tools from the server's `/tools` endpoint.
    - Invokes tools via `/a2a/{tool}/invoke` with a JSON payload `{parameters: ...}`.
    - Maintains a per-server tools cache with TTL to avoid redundant fetches.

    When to use:
    - You control or know the MCP server endpoints and don't need the Gateway
      for discovery or authorization.

    Example:
        strat = DirectMcpStrategy(servers=[ServerConfig(name="s1", endpoint_url="http://s1/mcp/")])
        tools = list(strat.list_tools("s1"))
        res = strat.call_tool("s1", "echo", {"text": "hi"})
    """

    def __init__(
        self,
        servers: Iterable[ServerConfig],
        *,
        client: Optional[httpx.Client] = None,
        ttl_seconds: float = 10.0,
    ) -> None:
        self._servers: List[ServerConfig] = list(servers)
        self._http = client or httpx.Client(timeout=10.0, follow_redirects=True)
        self._logger = logging.getLogger(__name__)
        self._ttl = ttl_seconds
        # cache: server_id -> (tools, expires_at)
        self._tools_cache: Dict[str, Tuple[List[McpTool], float]] = {}

    def list_servers(self) -> Iterable[McpServerRef]:
        """Yield `McpServerRef` entries for each configured server.

        Source of truth is the `ServerConfig` objects provided at construction time.

        Returns:
            Iterable of `McpServerRef` domain references.
        """
        for s in self._servers:
            yield McpServerRef(
                id=s.name,
                display_name=s.name,
                kind=ServerKind.DIRECT,
                transport=TransportType.STREAMABLE_HTTP,
                endpoint_url=s.endpoint_url,
                auth_token=s.auth_token,
            )

    def list_tools(self, server_id: str) -> Iterable[McpTool]:  # noqa: ARG002
        """Return tools for a server, honoring a short-lived cache.

        - Performs GET `<endpoint>/tools` and normalizes results into `McpTool`.
        - Uses `ttl_seconds` to cache results per server.

        Args:
            server_id: The configured `ServerConfig.name` for the server.

        Returns:
            Iterable of `McpTool`.

        Raises:
            httpx.HTTPStatusError: If the HTTP response indicates an error.
            httpx.TransportError: For transport-level HTTP issues.
        """
        # serve from cache if valid
        cached = self._tools_cache.get(server_id)
        now = time.monotonic()
        if cached and cached[1] > now:
            return list(cached[0])

        cfg = self._get_server(server_id)
        endpoint = cfg.endpoint_url.rstrip("/")
        headers = self._headers(cfg)
        self._logger.debug("DirectMcpStrategy.list_tools: GET %s/tools", endpoint)
        r = self._http.get(f"{endpoint}/tools", headers=headers)
        r.raise_for_status()
        data = r.json()
        tools: List[McpTool] = []
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if not isinstance(name, str) or not name:
                    continue
                description = item.get("description") if isinstance(item.get("description"), str) else None
                input_schema: Dict[str, Any] = (
                    item.get("inputSchema") if isinstance(item.get("inputSchema"), dict) else {}
                ) or {}
                is_mut = self._is_mutating_tool_name(name)
                tools.append(
                    McpTool(
                        name=name,
                        description=description,
                        mutating=is_mut,
                        arguments=self._infer_arguments(input_schema),
                        raw_parameters_schema=input_schema,
                    )
                )
        # update cache
        self._tools_cache[server_id] = (tools, now + self._ttl)
        return tools

    def call_tool(
        self,
        server_id: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        agent_id: str | None = None,  # noqa: ARG002
    ) -> ToolCallResult:
        """Invoke a tool on a direct server via `/a2a/{tool}/invoke`.

        - Sends a JSON payload `{ "parameters": args }`.
        - On success, returns `ToolCallResult` with the raw body under `data` and
          a boolean `ok` (defaults to True when not provided by the server).
        - If invocation is detected as mutating and `ok` is True, invalidates the
          per-server tools cache.

        Args:
            server_id: Configured server name.
            tool_name: Tool identifier provided by the server.
            args: Tool parameters as a dictionary.

        Returns:
            `ToolCallResult` describing the outcome.

        Raises:
            httpx.HTTPStatusError: If the HTTP response indicates an error.
            httpx.TransportError: For transport-level HTTP issues.
        """
        cfg = self._get_server(server_id)
        endpoint = cfg.endpoint_url.rstrip("/")
        headers = self._headers(cfg)
        payload: Dict[str, Any] = {"parameters": args or {}}
        self._logger.debug(
            "DirectMcpStrategy.call_tool: POST %s/a2a/%s/invoke args_keys=%s",
            endpoint,
            tool_name,
            list((args or {}).keys()),
        )
        r = self._http.post(f"{endpoint}/a2a/{tool_name}/invoke", headers=headers, json=payload)
        r.raise_for_status()
        body = r.json()
        ok = bool(body.get("ok", True)) if isinstance(body, dict) else True
        data: Dict[str, Any] = body if isinstance(body, dict) else {"result": body}
        # Invalidate cache if mutating tool and call succeeded
        if self._is_mutating_tool_name(tool_name) and ok:
            self._tools_cache.pop(server_id, None)
        return ToolCallResult(ok=ok, data=data)

    def _get_server(self, server_id: str) -> ServerConfig:
        """Lookup a configured server by name.

        Args:
            server_id: The `ServerConfig.name` of the target server.

        Returns:
            The matching `ServerConfig`.

        Raises:
            ValueError: If no configured server matches `server_id`.
        """
        for s in self._servers:
            if s.name == server_id:
                return s
        raise ValueError(f"Unknown server_id: {server_id}")

    def _headers(self, cfg: ServerConfig) -> Dict[str, str]:
        """Build HTTP headers for direct calls.

        - Always sets `Content-Type: application/json`.
        - Adds `Authorization` if `auth_token` is configured on the server.

        Args:
            cfg: The server configuration providing optional `auth_token`.

        Returns:
            A dictionary of headers including `Content-Type: application/json` and
            optional `Authorization` when an auth token is configured.
        """
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if cfg.auth_token:
            headers["Authorization"] = cfg.auth_token
        return headers

    # _infer_arguments and _is_mutating_tool_name inherited from StrategyCommonMixin
