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

from gearmeshing_ai.mcp_client.gateway_api import GatewayApiClient
from gearmeshing_ai.mcp_client.gateway_api.models.domain import GatewayTransport
from gearmeshing_ai.mcp_client.schemas.core import (
    McpServerRef,
    McpTool,
    ToolCallResult,
    ToolsPage,
    TransportType,
)

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
        self, gateway: GatewayApiClient, *, client: Optional[httpx.Client] = None, ttl_seconds: float = 10.0
    ) -> None:
        self._gateway = gateway
        # Sync client for streamable HTTP endpoints under the Gateway
        self._http = client or httpx.Client(timeout=10.0, follow_redirects=True)
        self._logger = logging.getLogger(__name__)
        self._ttl = ttl_seconds
        # cache: server_id -> (tools, expires_at)
        self._tools_cache: Dict[str, Tuple[List[McpTool], float]] = {}

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
        """Yield `McpServerRef` entries from the Gateway.

        Uses DTO->domain mapping provided by the Gateway models.

        Returns:
            Iterable of `McpServerRef` discovered via the Gateway API.
        """
        servers = self._gateway.list_servers()
        for s in servers:
            # Use model conversion to centralize mapping to domain ref
            yield s.to_server_ref(self._gateway.base_url)

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

        base = self._base_for(server_id).rstrip("/")
        self._logger.debug("GatewayMcpStrategy.list_tools: GET %s/tools", base)
        r = self._http.get(f"{base}/tools", headers=self._headers())
        r.raise_for_status()
        data = r.json()
        payload = ToolsListPayloadDTO.model_validate(data)
        tools: List[McpTool] = []
        for td in payload.tools:
            tools.append(td.to_mcp_tool(self._infer_arguments, self._is_mutating_tool_name))
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
        base = self._base_for(server_id).rstrip("/")
        q = ToolsListQuery(cursor=cursor, limit=limit)
        params = q.to_params()
        self._logger.debug("GatewayMcpStrategy.list_tools_page: GET %s/tools params=%s", base, params)
        r = self._http.get(f"{base}/tools", headers=self._headers(), params=params or None)
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

    def call_tool(
        self,
        server_id: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        agent_id: str | None = None,  # noqa: ARG002
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
            "GatewayMcpStrategy.call_tool: POST %s/a2a/%s/invoke args_keys=%s",
            base,
            tool_name,
            list((args or {}).keys()),
        )
        r = self._http.post(
            f"{base}/a2a/{tool_name}/invoke",
            json=payload.model_dump(by_alias=True, mode="json"),
            headers=self._headers(),
        )
        r.raise_for_status()
        body = r.json()
        inv = ToolInvokePayloadDTO.model_validate(body)
        result = inv.to_tool_call_result()
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
