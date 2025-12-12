"""Async Direct MCP strategy

Async strategy for directly connecting to MCP servers over HTTP without going
through the Gateway management API. Supports:

- Listing tools with a per-server TTL cache
- Paginated tools listing when server supports `cursor`/`limit`
- SSE streaming helpers for server-sent events

Typical usage:
    from gearmeshing_ai.mcp_client.schemas.config import ServerConfig
    from gearmeshing_ai.mcp_client.strategy.direct_async import AsyncDirectMcpStrategy

    strat = AsyncDirectMcpStrategy([
        ServerConfig(name="s1", endpoint_url="http://localhost:8000/mcp/"),
    ])

    tools = await strat.list_tools("s1")
    page = await strat.list_tools_page("s1", limit=50)
    res = await strat.call_tool("s1", "echo", {"text": "hi"})
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
from mcp import ClientSession
from ..transport.mcp import AsyncMCPTransport, StreamableHttpMCPTransport

from ..schemas.config import ServerConfig
from ..schemas.core import McpTool, ToolCallResult, ToolsPage
from .base import AsyncStrategy, StrategyCommonMixin
from .models.dto import (
    ToolInvokePayloadDTO,
    ToolInvokeRequestDTO,
    ToolsListPayloadDTO,
    ToolsListQuery,
)


class AsyncDirectMcpStrategy(StrategyCommonMixin, AsyncStrategy):
    """Async strategy for direct HTTP access to MCP servers.

    - Discovers servers from provided `ServerConfig` entries.
    - Uses `httpx.AsyncClient` for HTTP calls and maintains a per-server tools cache.
    - Exposes basic SSE streaming helpers using `BasicSseTransport`.

    Configure `ttl_seconds` to balance cache freshness and performance.
    Provide custom clients when you need custom timeouts/proxies.
    """

    def __init__(
        self,
        servers: List[ServerConfig],
        *,
        client: Optional[httpx.AsyncClient] = None,
        ttl_seconds: float = 10.0,
        sse_client: Optional[httpx.AsyncClient] = None,
        mcp_transport: Optional[AsyncMCPTransport] = None,
    ) -> None:
        self._servers: List[ServerConfig] = list(servers)
        self._http = client or httpx.AsyncClient(timeout=10.0, follow_redirects=True)
        self._logger = logging.getLogger(__name__)
        self._ttl = ttl_seconds
        self._tools_cache: Dict[str, Tuple[List[McpTool], float]] = {}
        self._sse_client = sse_client
        self._mcp_transport: AsyncMCPTransport = mcp_transport or StreamableHttpMCPTransport()

    @asynccontextmanager
    async def _open_session(self, server_id: str) -> AsyncIterator[ClientSession]:
        """Open a short-lived MCP ClientSession for the given server.

        This helper uses the official `mcp` streamable HTTP client to establish
        a bidirectional MCP connection to the configured endpoint. A new
        session is created per call, comparable in cost to the previous
        per-call HTTP usage, and keeps the strategy stateless.
        """
        cfg = self._get_server(server_id)
        base = cfg.endpoint_url.rstrip("/")
        async with self._mcp_transport.session(base) as session:
            yield session

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
        """Build HTTP headers for direct requests.

        Args:
            cfg: The server configuration providing optional `auth_token`.

        Returns:
            A dictionary of headers including `Content-Type: application/json` and
            `Authorization` when an auth token is configured.
        """
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if cfg.auth_token:
            headers["Authorization"] = cfg.auth_token
        return headers

    async def list_tools(self, server_id: str) -> List[McpTool]:
        """Return the list of tools for a server.

        - Honors a TTL cache keyed by server name.
        - Performs GET `<endpoint>/tools` and normalizes via `ToolsListPayloadDTO`.

        Args:
            server_id: The configured `ServerConfig.name` for the server.

        Returns:
            A list of `McpTool`.

        Raises:
            httpx.HTTPStatusError: If the HTTP response indicates an error.
            httpx.TransportError: For transport-level HTTP issues.
            pydantic.ValidationError: If the tools payload fails validation.
        """
        cached = self._tools_cache.get(server_id)
        now = time.monotonic()
        if cached and cached[1] > now:
            return list(cached[0])

        tools: List[McpTool] = []
        async with self._open_session(server_id) as session:
            self._logger.debug(
                "AsyncDirectMcpStrategy.list_tools: using MCP ClientSession for server_id=%s",
                server_id,
            )
            resp = await session.list_tools()
            for tool in getattr(resp, "tools", []) or []:
                name = getattr(tool, "name", None)
                if not isinstance(name, str) or not name:
                    continue
                desc_val = getattr(tool, "description", None)
                description = desc_val if isinstance(desc_val, str) else None
                input_schema = getattr(tool, "input_schema", {}) or {}
                if not isinstance(input_schema, dict):
                    input_schema = {}
                tools.append(
                    McpTool(
                        name=name,
                        description=description,
                        mutating=self._is_mutating_tool_name(name),
                        arguments=self._infer_arguments(input_schema),
                        raw_parameters_schema=input_schema,
                    )
                )
        self._tools_cache[server_id] = (tools, now + self._ttl)
        return tools

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> ToolCallResult:
        """Invoke a tool by POSTing to `/a2a/{tool}/invoke`.

        - Body: `{ "parameters": args }`.
        - Response normalized by `ToolInvokePayloadDTO` to `ToolCallResult`.
        - Invalidates cache on success for mutating tools (heuristic).

        Args:
            server_id: The configured `ServerConfig.name` for the server.
            tool_name: The tool identifier to invoke.
            args: The tool parameters to send.

        Returns:
            A `ToolCallResult` describing the outcome.

        Raises:
            httpx.HTTPStatusError: If the HTTP response indicates an error.
            httpx.TransportError: For transport-level HTTP issues.
            pydantic.ValidationError: If the invocation payload fails validation.
        """
        async with self._open_session(server_id) as session:
            self._logger.debug(
                "AsyncDirectMcpStrategy.call_tool: using MCP ClientSession for server_id=%s tool=%s args_keys=%s",
                server_id,
                tool_name,
                list((args or {}).keys()),
            )
            res = await session.call_tool(name=tool_name, arguments=args or {})
            payload = res.model_dump() if hasattr(res, "model_dump") else res
            ok = True
            data: Dict[str, Any] = payload if isinstance(payload, dict) else {"result": payload}

        if self._is_mutating_tool_name(tool_name) and ok:
            self._tools_cache.pop(server_id, None)
        return ToolCallResult(ok=ok, data=data)

    async def list_tools_page(
        self,
        server_id: str,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> ToolsPage:
        """Return a single page of tools for a server.

        - Uses MCP ClientSession.list_tools(cursor, limit) to request a page.
        - Returns a `ToolsPage` with `items` and an optional `next_cursor`.
        - Updates cache only for non-paginated calls (no cursor/limit).

        Args:
            server_id: The configured `ServerConfig.name` for the server.
            cursor: Cursor returned from a previous page.
            limit: Max number of items to request per page.

        Returns:
            A `ToolsPage` with `items` and optional `next_cursor`.

        Raises:
            httpx.HTTPStatusError: If the HTTP response indicates an error.
            httpx.TransportError: For transport-level HTTP issues.
            pydantic.ValidationError: If the tools payload fails validation.
        """
        tools: List[McpTool] = []
        next_cursor: Optional[str] = None
        async with self._open_session(server_id) as session:
            self._logger.debug(
                "AsyncDirectMcpStrategy.list_tools_page: using MCP ClientSession for server_id=%s cursor=%s limit=%s",
                server_id,
                cursor,
                limit,
            )
            # The MCP client typically supports optional pagination parameters.
            # If unsupported by the backend, it should return all tools with no next_cursor.
            resp = await session.list_tools(cursor=cursor, limit=limit)  # type: ignore[call-arg]
            for tool in getattr(resp, "tools", []) or []:
                name = getattr(tool, "name", None)
                if not isinstance(name, str) or not name:
                    continue
                desc_val = getattr(tool, "description", None)
                description = desc_val if isinstance(desc_val, str) else None
                input_schema = getattr(tool, "input_schema", {}) or {}
                if not isinstance(input_schema, dict):
                    input_schema = {}
                tools.append(
                    McpTool(
                        name=name,
                        description=description,
                        mutating=self._is_mutating_tool_name(name),
                        arguments=self._infer_arguments(input_schema),
                        raw_parameters_schema=input_schema,
                    )
                )
            next_cursor = getattr(resp, "next_cursor", None)
        if cursor is None and limit is None:
            now = time.monotonic()
            self._tools_cache[server_id] = (tools, now + self._ttl)
        return ToolsPage(items=tools, next_cursor=next_cursor)

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
        """Yield raw SSE lines from the server.

        Uses `BasicSseTransport` to connect and handle retries/backoff.
        Yields lines including blank ones (event boundary markers).

        Args:
            server_id: The configured `ServerConfig.name` for the server.
            path: The SSE path relative to the server's endpoint base.
            reconnect: Whether to reconnect automatically on errors.
            max_retries: Max reconnection attempts when `reconnect` is True.
            backoff_initial: Initial backoff delay in seconds.
            backoff_factor: Exponential backoff factor.
            backoff_max: Maximum backoff delay in seconds.
            idle_timeout: Optional idle-timeout for the connection.
            max_total_seconds: Optional max total streaming time.

        Returns:
            An async iterator of raw SSE lines (decoded strings).

        Raises:
            httpx.HTTPStatusError: If initial or subsequent connections fail.
            httpx.TransportError: For transport-level HTTP issues.
        """
        from gearmeshing_ai.info_provider.mcp.transport.sse import BasicSseTransport

        cfg = self._get_server(server_id)
        base = cfg.endpoint_url.rstrip("/")
        sse = BasicSseTransport(
            base,
            client=self._sse_client,
            auth_token=cfg.auth_token,
            include_blank_lines=True,
            reconnect=reconnect,
            max_retries=max_retries,
            backoff_initial=backoff_initial,
            backoff_factor=backoff_factor,
            backoff_max=backoff_max,
            idle_timeout=idle_timeout,
            max_total_seconds=max_total_seconds,
        )
        await sse.connect(path)
        try:
            self._logger.debug("AsyncDirectMcpStrategy.stream_events: start server_id=%s path=%s", server_id, path)
            async for line in sse.aiter():
                yield line
        finally:
            self._logger.debug("AsyncDirectMcpStrategy.stream_events: stop server_id=%s path=%s", server_id, path)
            await sse.close()

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
        """Yield parsed SSE events as dicts: `{id, event, data}`.

        - Multiple data lines are joined with `\n`.
        - Comments (lines starting with `:`) are ignored.
        - Blank line denotes event boundary.

        Args:
            server_id: The configured `ServerConfig.name` for the server.
            path: The SSE path relative to the server's endpoint base.
            reconnect: Whether to reconnect automatically on errors.
            max_retries: Max reconnection attempts when `reconnect` is True.
            backoff_initial: Initial backoff delay in seconds.
            backoff_factor: Exponential backoff factor.
            backoff_max: Maximum backoff delay in seconds.
            idle_timeout: Optional idle-timeout for the connection.
            max_total_seconds: Optional max total streaming time.

        Returns:
            An async iterator of parsed SSE event dictionaries.

        Raises:
            httpx.HTTPStatusError: If initial or subsequent connections fail.
            httpx.TransportError: For transport-level HTTP issues.
        """
        buf_id: Optional[str] = None
        buf_event: Optional[str] = None
        buf_data: List[str] = []
        async for line in self.stream_events(
            server_id,
            path,
            reconnect=reconnect,
            max_retries=max_retries,
            backoff_initial=backoff_initial,
            backoff_factor=backoff_factor,
            backoff_max=backoff_max,
            idle_timeout=idle_timeout,
            max_total_seconds=max_total_seconds,
        ):
            if not line.strip():
                if buf_id is not None or buf_event is not None or buf_data:
                    yield {
                        "id": buf_id,
                        "event": buf_event,
                        "data": "\n".join(buf_data),
                    }
                    buf_id, buf_event, buf_data = None, None, []
                continue
            if line.startswith(":"):
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                val = val.lstrip(" ")
            else:
                key, val = line, ""
            key = key.strip()
            if key == "id":
                buf_id = val
            elif key == "event":
                buf_event = val
            elif key == "data":
                buf_data.append(val)
            else:
                pass
        if buf_id is not None or buf_event is not None or buf_data:
            yield {
                "id": buf_id,
                "event": buf_event,
                "data": "\n".join(buf_data),
            }
