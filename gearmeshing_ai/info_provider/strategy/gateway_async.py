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
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from gearmeshing_ai.info_provider.gateway_api.client import GatewayApiClient
from gearmeshing_ai.info_provider.schemas.core import (
    McpTool,
    ToolCallResult,
    ToolsPage,
)

from .base import AsyncStrategy, StrategyCommonMixin
from .models.dto import (
    ToolInvokePayloadDTO,
    ToolInvokeRequestDTO,
    ToolsListPayloadDTO,
    ToolsListQuery,
)


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
    ) -> None:
        self._gateway = gateway
        self._http = client or httpx.AsyncClient(timeout=10.0, follow_redirects=True)
        self._logger = logging.getLogger(__name__)
        self._ttl = ttl_seconds
        self._tools_cache: Dict[str, Tuple[List[McpTool], float]] = {}
        self._sse_client = sse_client

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
        """Return tools for a Gateway server, honoring the per-server cache.

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

        base = self._base_for(server_id).rstrip("/")
        self._logger.debug("AsyncGatewayMcpStrategy.list_tools: GET %s/tools", base)
        r = await self._http.get(f"{base}/tools", headers=self._headers())
        r.raise_for_status()
        data = r.json()
        payload = ToolsListPayloadDTO.model_validate(data)
        tools: List[McpTool] = []
        for td in payload.tools:
            tools.append(td.to_mcp_tool(self._infer_arguments, self._is_mutating_tool_name))
        self._tools_cache[server_id] = (tools, now + self._ttl)
        return tools

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        args: Dict[str, Any],
    ) -> ToolCallResult:
        """Invoke a tool via the Gateway streamable HTTP endpoint.

        - POST `{gateway}/servers/{server_id}/mcp/a2a/{tool}/invoke`
        - Payload is `ToolInvokeRequestDTO(parameters=args)`
        - Normalized to `ToolCallResult` via `ToolInvokePayloadDTO`
        - Invalidates cache on success when tool is mutating

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
        base = self._base_for(server_id).rstrip("/")
        payload = ToolInvokeRequestDTO(parameters=args or {})
        self._logger.debug(
            "AsyncGatewayMcpStrategy.call_tool: POST %s/a2a/%s/invoke args_keys=%s",
            base,
            tool_name,
            list((args or {}).keys()),
        )
        r = await self._http.post(
            f"{base}/a2a/{tool_name}/invoke",
            headers=self._headers(),
            json=payload.model_dump(by_alias=True, mode="json"),
        )
        r.raise_for_status()
        body = r.json()
        inv = ToolInvokePayloadDTO.model_validate(body)
        result = inv.to_tool_call_result()

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
        base = self._base_for(server_id).rstrip("/")
        q = ToolsListQuery(cursor=cursor, limit=limit)
        params = q.to_params()
        self._logger.debug("AsyncGatewayMcpStrategy.list_tools_page: GET %s/tools params=%s", base, params)
        r = await self._http.get(f"{base}/tools", headers=self._headers(), params=params or None)
        r.raise_for_status()
        data = r.json()
        payload = ToolsListPayloadDTO.model_validate(data)
        tools: List[McpTool] = []
        for td in payload.tools:
            tools.append(td.to_mcp_tool(self._infer_arguments, self._is_mutating_tool_name))
        # Only update cache for unpaginated fetches
        if cursor is None and limit is None:
            import time as _time

            now = _time.monotonic()
            self._tools_cache[server_id] = (tools, now + self._ttl)
        return ToolsPage(items=tools, next_cursor=payload.next_cursor)

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
        """Yield raw SSE lines from the Gateway-connected server.

        Args:
            server_id: The Gateway server identifier.
            path: The SSE path relative to the server's MCP base.
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
        from gearmeshing_ai.info_provider.transport.sse import BasicSseTransport

        base = self._base_for(server_id)
        sse = BasicSseTransport(
            base,
            client=self._sse_client,
            auth_token=self._gateway.auth_token,
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
            self._logger.debug("AsyncGatewayMcpStrategy.stream_events: start server_id=%s path=%s", server_id, path)
            async for line in sse.aiter():
                yield line
        finally:
            self._logger.debug("AsyncGatewayMcpStrategy.stream_events: stop server_id=%s path=%s", server_id, path)
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
        """Yield parsed SSE events as dictionaries: {id, event, data}.

        - Multiple data lines are joined with `\n`.
        - Comments (lines starting with `:`) are ignored.
        - Event boundary is a blank line.

        Args:
            server_id: The Gateway server identifier.
            path: The SSE path relative to the server's MCP base.
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
                # comment
                continue
            # field parsing: key: value (value may be empty)
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
                # ignore other fields for now
                pass
        # flush tail if stream ended without final blank line
        if buf_id is not None or buf_event is not None or buf_data:
            yield {
                "id": buf_id,
                "event": buf_event,
                "data": "\n".join(buf_data),
            }
