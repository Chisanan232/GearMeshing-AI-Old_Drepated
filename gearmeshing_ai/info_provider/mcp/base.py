"""Client protocols and shared helpers for the MCP client facade.

Defines the sync/async client protocols that concrete clients must satisfy,
and a `ClientCommonMixin` with helpers for policy-based filtering.

Usage:
- `McpClient` and `AsyncMcpClient` implement these protocols.
- Strategies focus on transport; clients focus on policy and composition.
"""

from __future__ import annotations

from typing import (
    Dict,
    List,
    Optional,
    Protocol,
    Self,
    runtime_checkable,
)

import httpx

from .policy import PolicyMap, ToolPolicy
from .schemas.config import McpClientConfig
from .schemas.core import (
    McpServerRef,
    McpTool,
    ToolsPage,
)
from .strategy.base import is_mutating_tool_name


@runtime_checkable
class BaseMCPInfoProvider(Protocol):
    """Protocol for synchronous MCP info providers.

    Exposes only read-only MCP metadata needed by AI agents and other
    consumers:

    - discovery of MCP endpoints (servers)
    - listing tools for a given endpoint
    - optional paginated tool listing

    Concrete facades such as `McpClient` should also apply `ToolPolicy`
    constraints via `ClientCommonMixin` when an `agent_id` is supplied.
    """

    @classmethod
    def from_config(
        cls,
        config: McpClientConfig,
        *,
        agent_policies: Optional[PolicyMap] = None,
        direct_http_client: Optional[httpx.Client] = None,
        gateway_mgmt_client: Optional[httpx.Client] = None,
        gateway_http_client: Optional[httpx.Client] = None,
    ) -> Self: ...

    def get_endpoints(self, *, agent_id: str | None = None) -> List[McpServerRef]: ...

    def list_tools(self, server_id: str, *, agent_id: str | None = None) -> List[McpTool]: ...

    def list_tools_page(
        self,
        server_id: str,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        agent_id: str | None = None,
    ) -> ToolsPage: ...


@runtime_checkable
class BaseAsyncMCPInfoProvider(Protocol):
    """Protocol for asynchronous MCP info providers.

    Async counterpart to `MCPInfoProvider`, exposing endpoint and tool
    discovery APIs only. Concrete facades such as `AsyncMcpClient` may
    offer additional capabilities (tool invocation, streaming), but those
    are not part of this minimal info-provider contract.
    """

    @classmethod
    async def from_config(
        cls,
        config: McpClientConfig,
        *,
        agent_policies: Optional[PolicyMap] = None,
        gateway_mgmt_client: Optional[httpx.Client] = None,
        gateway_http_client: Optional[httpx.AsyncClient] = None,
        gateway_sse_client: Optional[httpx.AsyncClient] = None,
    ) -> Self: ...

    async def get_endpoints(self, *, agent_id: str | None = None) -> List[McpServerRef]: ...

    async def list_tools(self, server_id: str, *, agent_id: str | None = None) -> List[McpTool]: ...

    async def list_tools_page(
        self,
        server_id: str,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        agent_id: str | None = None,
    ) -> ToolsPage: ...


class ClientCommonMixin:
    """Shared policy helpers for sync/async client facades."""

    def _filter_servers_by_policy(
        self,
        servers: List[McpServerRef],
        policy: Optional[ToolPolicy],
    ) -> List[McpServerRef]:
        """Filter servers by `policy.allowed_servers` if configured.

        Args:
            servers: The list of discovered `McpServerRef`.
            policy: Optional `ToolPolicy` to apply.

        Returns:
            The filtered list of servers. If no policy or no allow-list is set, returns the original list.
        """
        if not policy or policy.allowed_servers is None:
            return servers
        allowed = policy.allowed_servers
        return [s for s in servers if s.id in allowed]

    def _filter_tools_by_policy(
        self,
        tools: List[McpTool],
        policy: Optional[ToolPolicy],
    ) -> List[McpTool]:
        """Apply allowed_tools and read-only filtering to a tool list.

        Args:
            tools: The tool list to filter.
            policy: Optional `ToolPolicy` whose `allowed_tools` and `read_only` fields are honored.

        Returns:
            The filtered tool list.
        """
        if not policy:
            return tools
        res = tools
        if policy.allowed_tools is not None:
            res = [t for t in res if t.name in policy.allowed_tools]
        if policy.read_only:
            res = [t for t in res if not (t.mutating or is_mutating_tool_name(t.name))]
        return res

    def _should_block_read_only(
        self,
        policy: Optional[ToolPolicy],
        tool_name: str,
        listed_lookup: Optional[Dict[str, McpTool]] = None,
    ) -> bool:
        """Return True if a call to `tool_name` should be blocked by read-only policy.

        Args:
            policy: Optional `ToolPolicy` to check.
            tool_name: The tool name being invoked.
            listed_lookup: Optional mapping of tool name to `McpTool` metadata to improve mutating detection.

        Returns:
            True if the call should be blocked due to read-only policy; otherwise False.
        """
        if not policy or not policy.read_only:
            return False
        if listed_lookup is not None:
            t = listed_lookup.get(tool_name)
            if t is not None and t.mutating:
                return True
        return is_mutating_tool_name(tool_name)
