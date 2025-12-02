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
    TransportType,
    ToolsPage,
)

from .base import StrategyCommonMixin, SyncStrategy
from .dto import (
    ToolInvokePayloadDTO,
    ToolInvokeRequestDTO,
    ToolsListPayloadDTO,
    ToolsListQuery,
)


class GatewayMcpStrategy(StrategyCommonMixin, SyncStrategy):
    """
    Strategy that discovers servers via the MCP Gateway management API and
    (optionally) interacts with their streamable HTTP endpoints.

    Note: Tool listing and invocation are kept light and may be wired to real
    HTTP I/O later.
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
        if t == GatewayTransport.STREAMABLE_HTTP:
            return TransportType.STREAMABLE_HTTP
        if t == GatewayTransport.SSE:
            return TransportType.SSE
        return TransportType.STDIO

    def list_servers(self) -> Iterable[McpServerRef]:
        servers = self._gateway.list_servers()
        for s in servers:
            # Use model conversion to centralize mapping to domain ref
            yield s.to_server_ref(self._gateway.base_url)

    def list_tools(self, server_id: str) -> Iterable[McpTool]:
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
        """
        Return a page of tools for a server, if the backend supports pagination.

        - cursor: opaque string from a previous response's next_cursor
        - limit: page size hint

        Example:
            page = strategy.list_tools_page("s1", limit=50)
            tools = list(page.items)
            while page.next_cursor:
                page = strategy.list_tools_page("s1", cursor=page.next_cursor, limit=50)
                tools.extend(page.items)
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
        # Construct the streamable HTTP base under the gateway
        return f"{self._gateway.base_url}/servers/{server_id}/mcp"

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        token = self._gateway.auth_token
        if token:
            headers["Authorization"] = token
        return headers
