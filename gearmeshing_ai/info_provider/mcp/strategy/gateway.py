"""Gateway MCP strategy (sync)

This module implements a synchronous strategy that discovers MCP servers via a
Gateway management API and interacts with their streamable HTTP endpoints.

Targets and use-cases:
- Centralized server discovery and management via Gateway.
- Environments where authorization and transport details are brokered by Gateway.

Highlights:
- Per-server tools caching with configurable TTL.
- Tool invocation through Gateway's streamable HTTP endpoint.
- Optional pagination for tools listing, returning `ToolsPage`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

from ..gateway_api import GatewayApiClient
from ..gateway_api.models.domain import GatewayTransport
from ..gateway_api.models.dto import AdminToolsListResponseDTO
from ..schemas.core import (
    McpServerRef,
    McpTool,
    ServerKind,
    ToolCallResult,
    ToolsPage,
    TransportType,
)
from ..transport.mcp import AsyncMCPTransport, StreamableHttpMCPTransport
from .base import StrategyCommonMixin, SyncStrategy
from .models.dto import (
    ToolInvokePayloadDTO,
    ToolInvokeRequestDTO,
    ToolsListPayloadDTO,
    ToolsListQuery,
)


class GatewayMcpStrategy(StrategyCommonMixin, SyncStrategy):
    """Synchronous strategy for Gateway-discovered MCP servers.

    - Uses `GatewayApiClient` for server discovery and auth propagation.
    - Lists tools via `GET {gateway}/servers/{id}/mcp/tools`.
    - Invokes tools via `POST {gateway}/servers/{id}/mcp/a2a/{tool}/invoke`.
    - Maintains per-server tools cache with TTL to reduce redundant calls.

    Example:
        gw = GatewayApiClient("https://gw.example", auth_token="Bearer ...")
        strat = GatewayMcpStrategy(gw)
        tools = list(strat.list_tools("server-id"))
        res = strat.call_tool("server-id", "echo", {"text": "hi"})
    """

    def __init__(
        self, gateway: GatewayApiClient, *, client: Optional[httpx.Client] = None, ttl_seconds: float = 10.0,
        mcp_transport: Optional[AsyncMCPTransport] = None,
    ) -> None:
        self._gateway = gateway
        # Sync client for streamable HTTP endpoints under the Gateway
        self._http = client or httpx.Client(timeout=10.0, follow_redirects=True)
        self._logger = logging.getLogger(__name__)
        self._ttl = ttl_seconds
        # cache: server_id -> (tools, expires_at)
        self._tools_cache: Dict[str, Tuple[List[McpTool], float]] = {}
        self._mcp_transport: AsyncMCPTransport = mcp_transport or StreamableHttpMCPTransport()

    def _map_transport(self, t: GatewayTransport) -> TransportType:
        """Map a Gateway transport enum to the core `TransportType`.

        Args:
            t: Gateway transport value reported by the management API.

        Returns:
            The corresponding `TransportType` used by domain references.
        """
        if t == GatewayTransport.STREAMABLE_HTTP:
            return TransportType.STREAMABLE_HTTP
        if t == GatewayTransport.SSE:
            return TransportType.SSE
        return TransportType.STDIO

    def list_servers(self) -> Iterable[McpServerRef]:
        """Yield `McpServerRef` entries using Gateway admin utility.

        Uses `client.admin.gateway.list()` and maps entries to `McpServerRef` by
        leveraging the `url` and `transport` reported for each gateway entry.

        Returns:
            Iterable of `McpServerRef` discovered via the Gateway admin API.
        """
        gateways = self._gateway.admin.gateway.list()
        for gw in gateways:
            transport = TransportType.SSE
            if (gw.transport or "SSE") == "STREAMABLEHTTP":
                transport = TransportType.STREAMABLE_HTTP
            elif (gw.transport or "SSE") == "STDIO":
                transport = TransportType.STDIO
            yield McpServerRef(
                id=gw.id or gw.slug or gw.name,
                display_name=gw.name,
                kind=ServerKind.GATEWAY,
                transport=transport,
                endpoint_url=str(gw.url).rstrip("/"),
                auth_token=self._gateway.auth_token,
            )

    def list_tools(self, server_id: str) -> Iterable[McpTool]:
        """Return tools for a Gateway server, honoring per-server cache.

        - GET `{gateway}/servers/{server_id}/mcp/tools`
        - Normalizes to `McpTool` via `ToolsListPayloadDTO` -> `ToolDescriptorDTO`

        Args:
            server_id: The Gateway server identifier.

        Returns:
            Iterable of `McpTool`.

        Raises:
            httpx.HTTPStatusError: If the HTTP response indicates an error.
            httpx.TransportError: For transport-level HTTP issues.
            pydantic.ValidationError: If the tools payload fails validation.
        """
        # serve from cache if valid
        cached = self._tools_cache.get(server_id)
        import time as _time

        now = _time.monotonic()
        if cached and cached[1] > now:
            return list(cached[0])

        # Use admin tools listing utility
        self._logger.debug("GatewayMcpStrategy.list_tools: GET admin/tools via client utility")
        dto: AdminToolsListResponseDTO = self._gateway.admin.tools.list()
        tools: List[McpTool] = []
        for t in dto.data or []:
            name = t.customName or t.originalName or t.name
            input_schema = (
                t.inputSchema.model_dump() if getattr(t.inputSchema, "model_dump", None) else (t.inputSchema or {})
            )
            tools.append(
                McpTool(
                    name=name,
                    description=t.description,
                    mutating=self._is_mutating_tool_name(name),
                    arguments=self._infer_arguments(input_schema),
                    raw_parameters_schema=input_schema,
                )
            )
        # update cache
        self._tools_cache[server_id] = (tools, now + self._ttl)
        return tools

    def list_tools_page(
        self,
        server_id: str,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> ToolsPage:
        """Return a page of tools for a server when pagination is supported.

        - Query built via `ToolsListQuery(cursor, limit).to_params()`
        - Returns `ToolsPage(items, next_cursor)`
        - Updates cache only for non-paginated requests

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
        # Use admin tools listing with pagination params mapped to offset/limit
        offset: int = 0
        if cursor is not None:
            try:
                offset = int(cursor)
            except Exception:
                offset = 0
        dto: AdminToolsListResponseDTO = self._gateway.admin.tools.list(offset=offset, limit=limit or 50)
        tools: List[McpTool] = []
        for t in dto.data or []:
            name = t.customName or t.originalName or t.name
            input_schema = (
                t.inputSchema.model_dump() if getattr(t.inputSchema, "model_dump", None) else (t.inputSchema or {})
            )
            tools.append(
                McpTool(
                    name=name,
                    description=t.description,
                    mutating=self._is_mutating_tool_name(name),
                    arguments=self._infer_arguments(input_schema),
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
        if dto.pagination and dto.pagination.page is not None and dto.pagination.perPage is not None and dto.pagination.total is not None:
            page = dto.pagination.page
            per = dto.pagination.perPage
            total = dto.pagination.total
            if (page * per) < total:
                next_cursor = str(page * per)
        return ToolsPage(items=tools, next_cursor=next_cursor)

    def call_tool(
        self,
        server_id: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        agent_id: str | None = None,  # noqa: ARG002
    ) -> ToolCallResult:
        """Invoke a tool using the same mechanism as the direct strategy.

        The Gateway management API does not execute tools; we resolve the
        underlying server endpoint via `admin.gateway.get(server_id)` and call
        the tool over the MCP transport.

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
            "GatewayMcpStrategy.call_tool (direct-style): base=%s tool=%s args_keys=%s",
            base,
            tool_name,
            list((args or {}).keys()),
        )
        import asyncio as _asyncio

        async def _work() -> Tuple[bool, Dict[str, Any]]:
            async with self._mcp_transport.session(base) as session:
                res = await session.call_tool(name=tool_name, arguments=args or {})
                if hasattr(res, "model_dump"):
                    return True, res.model_dump()
                if isinstance(res, dict):
                    return True, res
                return True, {"ok": res}

        ok, data = _asyncio.run(_work())
        result = ToolCallResult(ok=ok, data=data)
        # Invalidate cache if mutating tool and call succeeded (prefer cached metadata if available)
        cached = self._tools_cache.get(server_id)
        is_mut = None
        if cached:
            for t in cached[0]:
                if t.name == tool_name:
                    is_mut = t.mutating
                    break
        if is_mut is None:
            is_mut = self._is_mutating_tool_name(tool_name)
        if is_mut and result.ok:
            self._tools_cache.pop(server_id, None)
        return result

    def _base_for(self, server_id: str) -> str:
        """Construct the Gateway streamable HTTP base URL for a server.

        Args:
            server_id: The Gateway server identifier.

        Returns:
            The base URL where MCP endpoints for the server are exposed.
        """
        return f"{self._gateway.base_url}/servers/{server_id}/mcp"

    def _headers(self) -> Dict[str, str]:
        """Build standard JSON headers; propagate Gateway auth if configured.

        Returns:
            A dictionary with `Content-Type` and optional `Authorization` header.
        """
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        token = self._gateway.auth_token
        if token:
            headers["Authorization"] = token
        return headers
