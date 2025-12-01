from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

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

    def __init__(self, gateway: GatewayApiClient, *, client: Optional[httpx.Client] = None) -> None:
        self._gateway = gateway
        # Sync client for streamable HTTP endpoints under the Gateway
        self._http = client or httpx.Client(timeout=10.0, follow_redirects=True)
        self._logger = logging.getLogger(__name__)

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
        base = self._base_for(server_id).rstrip("/")
        self._logger.debug("GatewayMcpStrategy.list_tools: GET %s/tools", base)
        r = self._http.get(f"{base}/tools")
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
                tools.append(
                    McpTool(
                        name=name,
                        description=description,
                        arguments=self._infer_arguments(input_schema),
                        raw_parameters_schema=input_schema,
                    )
                )
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
        r = self._http.post(f"{base}/a2a/{tool_name}/invoke", json=payload)
        r.raise_for_status()
        body = r.json()
        ok = bool(body.get("ok", True)) if isinstance(body, dict) else True
        data: Dict[str, Any] = body if isinstance(body, dict) else {"result": body}
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
