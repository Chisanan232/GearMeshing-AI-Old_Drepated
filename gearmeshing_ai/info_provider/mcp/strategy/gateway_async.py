"""Async Gateway MCP strategy

Async strategy that discovers MCP servers via a Gateway management API and
interacts with their streamable HTTP endpoints.

Targets and use-cases:
- Centralized discovery/management via a Gateway service.
- Environments where auth and transport selection are delegated to Gateway.

Highlights:
- Per-server tools caching with TTL.
- Tool invocation through Gateway endpoints.
- Optional pagination for tools listing.
- SSE streaming helpers (raw and parsed events).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..gateway_api.client import GatewayApiClient
from ..schemas.core import (
    McpTool,
    ToolCallResult,
    ToolsPage,
)
from ..transport import AsyncMCPTransport, StreamableHttpMCPTransport
from .base import AsyncStrategy, StrategyCommonMixin


class AsyncGatewayMcpStrategy(StrategyCommonMixin, AsyncStrategy):
    """Async variant of the Gateway strategy for streamable HTTP MCP endpoints.

    - Discovers base via `GatewayApiClient.base_url`.
    - Uses `httpx.AsyncClient` for HTTP operations.
    - Maintains a per-server tools cache with TTL.
    - Propagates Authorization header from `GatewayApiClient`.
    """

    def __init__(
        self,
        gateway: GatewayApiClient,
        *,
        client: Optional[httpx.AsyncClient] = None,
        ttl_seconds: float = 10.0,
        sse_client: Optional[httpx.AsyncClient] = None,
        mcp_transport: Optional[AsyncMCPTransport] = None,
    ) -> None:
        self._gateway = gateway
        self._http = client or httpx.AsyncClient(timeout=10.0, follow_redirects=True)
        self._logger = logging.getLogger(__name__)
        self._ttl = ttl_seconds
        self._tools_cache: Dict[str, Tuple[List[McpTool], float]] = {}
        self._sse_client = sse_client
        self._mcp_transport: AsyncMCPTransport = mcp_transport or StreamableHttpMCPTransport()

    def _base_for(self, server_id: str) -> str:
        """Build the Gateway streamable HTTP base URL for a server.

        Args:
            server_id: The Gateway server identifier.

        Returns:
            The base URL where MCP endpoints for the server are exposed.
        """
        return f"{self._gateway.base_url}/servers/{server_id}/mcp"

    def _headers(self) -> Dict[str, str]:
        """Build JSON headers, propagating Gateway auth if configured.

        Returns:
            A dictionary with `Content-Type` and optional `Authorization` header.
        """
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        token = self._gateway.auth_token
        if token:
            headers["Authorization"] = token
        return headers

    async def list_tools(self, server_id: str) -> List[McpTool]:
        """Return tools for a Gateway server via admin utility, honoring cache.

        Args:
            server_id: The Gateway server identifier.

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

        # Use admin tools listing utility
        from ..gateway_api.models.dto import AdminToolsListResponseDTO

        self._logger.debug("AsyncGatewayMcpStrategy.list_tools: GET admin/tools via client utility")
        dto: AdminToolsListResponseDTO = self._gateway.admin.tools.list()
        tools: List[McpTool] = []
        for t in dto.data or []:
            name = t.customName or t.originalName or t.name
            input_schema = (
                t.inputSchema.model_dump() if getattr(t.inputSchema, "model_dump", None) else (t.inputSchema or {})
            )
            # Ensure input_schema is a dict for _infer_arguments
            schema_dict: Dict[str, Any] = input_schema if isinstance(input_schema, dict) else {}
            tools.append(
                McpTool(
                    name=name,
                    description=t.description,
                    mutating=self._is_mutating_tool_name(name),
                    arguments=self._infer_arguments(schema_dict),
                    raw_parameters_schema=input_schema,
                )
            )
        self._tools_cache[server_id] = (tools, now + self._ttl)
        return tools

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        args: Dict[str, Any],
    ) -> ToolCallResult:
        """Invoke a tool using the same transport approach as the direct strategy.

        Args:
            server_id: The Gateway server identifier.
            tool_name: The tool identifier to invoke.
            args: The tool parameters to send.

        Returns:
            A `ToolCallResult` describing the outcome.

        Raises:
            httpx.HTTPStatusError: If the HTTP response indicates an error.
            httpx.TransportError: For transport-level HTTP issues.
            pydantic.ValidationError: If the invocation payload fails validation.
        """
        gw = self._gateway.admin.gateway.get(server_id)
        base = (gw.url or "").rstrip("/")
        self._logger.debug(
            "AsyncGatewayMcpStrategy.call_tool (direct-style): base=%s tool=%s args_keys=%s",
            base,
            tool_name,
            list((args or {}).keys()),
        )
        async with self._mcp_transport.session(base) as session:
            res = await session.call_tool(name=tool_name, arguments=args or {})
            if hasattr(res, "model_dump"):
                result = ToolCallResult(ok=True, data=res.model_dump())
            elif isinstance(res, dict):
                result = ToolCallResult(ok=True, data=res)
            else:
                result = ToolCallResult(ok=True, data={"ok": res})

        # Invalidate cache if mutating tool (prefer cached metadata if available)
        cached = self._tools_cache.get(server_id)
        is_mut: Optional[bool] = None
        if cached:
            for t in cached[0]:
                if t.name == tool_name:
                    is_mut = t.mutating
                    break
        if is_mut is None:
            is_mut = self._is_mutating_tool_name(tool_name)
        if is_mut:
            self._tools_cache.pop(server_id, None)

        return result

    async def list_tools_page(
        self,
        server_id: str,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> ToolsPage:
        """Return a page of tools using admin utility with pagination.

        Args:
            server_id: The Gateway server identifier.
            cursor: Cursor returned from a previous page.
            limit: Max number of items to request per page.

        Returns:
            A `ToolsPage` with `items` and optional `next_cursor`.

        Raises:
            httpx.HTTPStatusError: If the HTTP response indicates an error.
            httpx.TransportError: For transport-level HTTP issues.
            pydantic.ValidationError: If the tools payload fails validation.
        """
        from ..gateway_api.models.dto import AdminToolsListResponseDTO

        # Map cursor to offset
        offset: int = 0
        if cursor is not None:
            try:
                offset = int(cursor)
            except Exception:
                offset = 0
        self._logger.debug("AsyncGatewayMcpStrategy.list_tools_page: GET admin/tools offset=%s limit=%s", offset, limit)
        dto: AdminToolsListResponseDTO = self._gateway.admin.tools.list(offset=offset, limit=limit or 50)
        tools: List[McpTool] = []
        for t in dto.data or []:
            name = t.customName or t.originalName or t.name
            input_schema = (
                t.inputSchema.model_dump() if getattr(t.inputSchema, "model_dump", None) else (t.inputSchema or {})
            )
            # Ensure input_schema is a dict for _infer_arguments
            schema_dict: Dict[str, Any] = input_schema if isinstance(input_schema, dict) else {}
            tools.append(
                McpTool(
                    name=name,
                    description=t.description,
                    mutating=self._is_mutating_tool_name(name),
                    arguments=self._infer_arguments(schema_dict),
                    raw_parameters_schema=input_schema,
                )
            )
        # Only update cache for unpaginated fetches
        if cursor is None and limit is None:
            import time as _time

            now = _time.monotonic()
            self._tools_cache[server_id] = (tools, now + self._ttl)
        # Derive next cursor from pagination if available
        next_cursor: Optional[str] = None
        if (
            dto.pagination
            and dto.pagination.page is not None
            and dto.pagination.perPage is not None
            and dto.pagination.total is not None
        ):
            page = dto.pagination.page
            per = dto.pagination.perPage
            total = dto.pagination.total
            if (page * per) < total:
                next_cursor = str(page * per)
        return ToolsPage(items=tools, next_cursor=next_cursor)
