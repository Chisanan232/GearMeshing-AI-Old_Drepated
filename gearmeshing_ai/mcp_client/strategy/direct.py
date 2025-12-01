from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

from gearmeshing_ai.mcp_client.schemas.config import ServerConfig
from gearmeshing_ai.mcp_client.schemas.core import (
    McpServerRef,
    McpTool,
    ServerKind,
    ToolCallResult,
    TransportType,
)

from .base import StrategyCommonMixin, SyncStrategy


class DirectMcpStrategy(StrategyCommonMixin, SyncStrategy):
    """
    Strategy for directly connecting to MCP servers.

    For now, list_servers is implemented from provided ServerConfig entries.
    list_tools and call_tool are stubs to be implemented with real transports later.
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
        for s in self._servers:
            if s.name == server_id:
                return s
        raise ValueError(f"Unknown server_id: {server_id}")

    def _headers(self, cfg: ServerConfig) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if cfg.auth_token:
            headers["Authorization"] = cfg.auth_token
        return headers

    # _infer_arguments and _is_mutating_tool_name inherited from StrategyCommonMixin
