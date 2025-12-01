from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Protocol, runtime_checkable, Self

import httpx

from gearmeshing_ai.mcp_client.policy import ToolPolicy
from gearmeshing_ai.mcp_client.schemas.core import McpServerRef, McpTool, ToolCallResult
from gearmeshing_ai.mcp_client.strategy.base import is_mutating_tool_name


@runtime_checkable
class SyncClientProtocol(Protocol):
    @classmethod
    def from_config(
        cls,
        config: "McpClientConfig",
        *,
        agent_policies: Optional["PolicyMap"] = None,
        direct_http_client: Optional[httpx.Client] = None,
        gateway_mgmt_client: Optional[httpx.Client] = None,
        gateway_http_client: Optional[httpx.Client] = None,
    ) -> Self: ...

    def list_servers(self, *, agent_id: str | None = None) -> List[McpServerRef]: ...

    def list_tools(self, server_id: str, *, agent_id: str | None = None) -> List[McpTool]: ...

    def call_tool(
        self,
        server_id: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        agent_id: str | None = None,
    ) -> ToolCallResult: ...


@runtime_checkable
class AsyncClientProtocol(Protocol):
    @classmethod
    async def from_config(
        cls,
        config: "McpClientConfig",
        *,
        agent_policies: Optional["PolicyMap"] = None,
        gateway_mgmt_client: Optional[httpx.Client] = None,
        gateway_http_client: Optional[httpx.AsyncClient] = None,
        gateway_sse_client: Optional[httpx.AsyncClient] = None,
    ) -> Self: ...

    async def list_tools(self, server_id: str, *, agent_id: str | None = None) -> List[McpTool]: ...

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        agent_id: str | None = None,
    ) -> ToolCallResult: ...

    async def stream_events(self, server_id: str, path: str = "/sse", **kwargs: Any) -> AsyncIterator[str]: ...

    async def stream_events_parsed(
        self,
        server_id: str,
        path: str = "/sse",
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]: ...


class ClientCommonMixin:
    def _filter_servers_by_policy(
        self,
        servers: List[McpServerRef],
        policy: Optional[ToolPolicy],
    ) -> List[McpServerRef]:
        if not policy or policy.allowed_servers is None:
            return servers
        allowed = policy.allowed_servers
        return [s for s in servers if s.id in allowed]

    def _filter_tools_by_policy(
        self,
        tools: List[McpTool],
        policy: Optional[ToolPolicy],
    ) -> List[McpTool]:
        if not policy:
            return tools
        res = tools
        if policy.allowed_tools is not None:
            res = [t for t in res if t.name in policy.allowed_tools]
        if policy.read_only:
            res = [
                t
                for t in res
                if not (t.mutating or is_mutating_tool_name(t.name))
            ]
        return res

    def _should_block_read_only(
        self,
        policy: Optional[ToolPolicy],
        tool_name: str,
        listed_lookup: Optional[Dict[str, McpTool]] = None,
    ) -> bool:
        if not policy or not policy.read_only:
            return False
        if listed_lookup is not None:
            t = listed_lookup.get(tool_name)
            if t is not None and t.mutating:
                return True
        return is_mutating_tool_name(tool_name)
