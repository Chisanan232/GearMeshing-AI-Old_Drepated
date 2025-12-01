from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

from gearmeshing_ai.mcp_client.gateway_api import GatewayApiClient
from gearmeshing_ai.mcp_client.gateway_api.models import GatewayTransport
from gearmeshing_ai.mcp_client.schemas.core import (
    McpServerRef,
    McpTool,
    ServerKind,
    ToolArgument,
    ToolCallResult,
    TransportType,
)


class GatewayMcpStrategy:
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
            yield McpServerRef(
                id=s.id,
                display_name=s.name,
                kind=ServerKind.GATEWAY,
                transport=self._map_transport(s.transport),
                # Per spec, the Gateway exposes /servers/{id}/mcp/ for streamable HTTP
                endpoint_url=f"{self._gateway.base_url}/servers/{s.id}/mcp/",
            )

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
                # Prefer explicit metadata (x-mutating) from item or input schema; fallback to heuristic
                explicit = item.get("x-mutating")
                if explicit is None:
                    explicit = input_schema.get("x-mutating") if isinstance(input_schema, dict) else None
                if explicit is True:
                    is_mut = True
                elif explicit is False:
                    is_mut = False
                else:
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
        base = self._base_for(server_id).rstrip("/")
        payload: Dict[str, Any] = {"parameters": args or {}}
        self._logger.debug(
            "GatewayMcpStrategy.call_tool: POST %s/a2a/%s/invoke args_keys=%s",
            base,
            tool_name,
            list((args or {}).keys()),
        )
        r = self._http.post(f"{base}/a2a/{tool_name}/invoke", json=payload, headers=self._headers())
        r.raise_for_status()
        body = r.json()
        ok = bool(body.get("ok", True)) if isinstance(body, dict) else True
        data: Dict[str, Any] = body if isinstance(body, dict) else {"result": body}
        # Invalidate cache if mutating tool (prefer cached metadata if available)
        cached = self._tools_cache.get(server_id)
        is_mut = None
        if cached:
            for t in cached[0]:
                if t.name == tool_name:
                    is_mut = bool(getattr(t, "mutating", False))
                    break
        if is_mut is None:
            is_mut = self._is_mutating_tool_name(tool_name)
        if is_mut:
            self._tools_cache.pop(server_id, None)
        return ToolCallResult(ok=ok, data=data)

    def _base_for(self, server_id: str) -> str:
        # Construct the streamable HTTP base under the gateway
        return f"{self._gateway.base_url}/servers/{server_id}/mcp"

    def _infer_arguments(self, input_schema: Dict[str, Any]) -> List[ToolArgument]:
        args: List[ToolArgument] = []
        props = input_schema.get("properties") if isinstance(input_schema, dict) else None
        required = set(input_schema.get("required") or []) if isinstance(input_schema, dict) else set()
        if isinstance(props, dict):
            for k, v in props.items():
                if not isinstance(v, dict):
                    continue
                typ = v.get("type") if isinstance(v.get("type"), str) else "string"
                desc = v.get("description") if isinstance(v.get("description"), str) else None
                args.append(
                    ToolArgument(
                        name=str(k),
                        type=str(typ),
                        required=str(k) in required,
                        description=desc,
                    )
                )
        return args

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        token = getattr(self._gateway, "auth_token", None)
        if token:
            headers["Authorization"] = token
        return headers

    @staticmethod
    def _is_mutating_tool_name(name: str) -> bool:
        n = name.lower()
        prefixes = ("create", "update", "delete", "remove", "post_", "put_", "patch_", "write", "set_")
        return n.startswith(prefixes)
