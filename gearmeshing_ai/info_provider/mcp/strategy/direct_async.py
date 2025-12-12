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
from mcp import ClientSession
from mcp.types import ListToolsResult, CallToolResult as MCPCallToolResult
from ..transport.mcp import AsyncMCPTransport, StreamableHttpMCPTransport

from ..schemas.config import ServerConfig
from ..schemas.core import McpTool, ToolCallResult, ToolsPage
from .base import AsyncStrategy, StrategyCommonMixin
 


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
        ttl_seconds: float = 10.0,
        mcp_transport: Optional[AsyncMCPTransport] = None,
    ) -> None:
        self._servers: List[ServerConfig] = list(servers)
        self._logger = logging.getLogger(__name__)
        self._ttl = ttl_seconds
        self._tools_cache: Dict[str, Tuple[List[McpTool], float]] = {}
        self._mcp_transport = mcp_transport or StreamableHttpMCPTransport()

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

    # No HTTP headers builder needed; MCP transport handles wire details.

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
            resp: ListToolsResult = await session.list_tools()
            for tool in resp.tools or []:
                name = tool.name
                description = tool.description
                input_schema = (tool.inputSchema or {})
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
            ok = True
            if isinstance(res, MCPCallToolResult):
                data: Dict[str, Any] = res.model_dump()
            elif hasattr(res, "model_dump"):
                data = res.model_dump()
            elif isinstance(res, dict):
                data = res
            else:
                data = {"ok": res}

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
            resp: ListToolsResult = await session.list_tools(cursor=cursor, limit=limit)  # type: ignore[call-arg]
            for tool in resp.tools or []:
                name = tool.name
                description = tool.description
                input_schema = (tool.inputSchema or {})
                tools.append(
                    McpTool(
                        name=name,
                        description=description,
                        mutating=self._is_mutating_tool_name(name),
                        arguments=self._infer_arguments(input_schema),
                        raw_parameters_schema=input_schema,
                    )
                )
            next_cursor = resp.next_cursor
        if cursor is None and limit is None:
            now = time.monotonic()
            self._tools_cache[server_id] = (tools, now + self._ttl)
        return ToolsPage(items=tools, next_cursor=next_cursor)

    
