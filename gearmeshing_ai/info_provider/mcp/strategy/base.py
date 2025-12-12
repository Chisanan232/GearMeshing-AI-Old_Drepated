"""Strategy protocols and shared helpers.

Defines the `SyncStrategy` and `AsyncStrategy` protocols that concrete
strategies must implement, along with a `StrategyCommonMixin` that provides
helpers for argument inference and mutating-tool detection.

Usage:
- DirectMcpStrategy, GatewayMcpStrategy, and their async counterparts implement
  these protocols and may reuse the mixin.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Protocol,
    runtime_checkable,
)

from ..schemas.core import (
    McpServerRef,
    McpTool,
    ToolArgument,
    ToolCallResult,
)


def is_mutating_tool_name(name: str) -> bool:
    """Heuristic to determine whether a tool name implies state mutation.

    Args:
        name: The tool name to inspect.

    Returns:
        True if the name starts with common mutating prefixes such as
        create/update/delete/remove/post_/put_/patch_/write/set_.
    """
    n = name.lower()
    prefixes = ("create", "update", "delete", "remove", "post_", "put_", "patch_", "write", "set_")
    return n.startswith(prefixes)


@runtime_checkable
class SyncStrategy(Protocol):
    """Protocol for synchronous strategies.

    Implementations should expose discovery (optional), tools listing, and
    tool invocation for a single transport family (direct or gateway).

    Examples:
        >>> # iterate over available servers
        >>> for s in strategy.list_servers():
        ...     print(s.id, s.label)
    """

    def list_servers(self) -> Iterable[McpServerRef]:
        """List all MCP servers accessible through this strategy.

        Args:
            None

        Returns:
            Iterable[McpServerRef]: An iterable of server references describing
            the server id and human-readable label. Implementations may return
            a generator; callers should not assume materialized lists.

        Raises:
            Exception: Implementations may raise transport- or discovery-related
            errors (e.g., network failures) where applicable.

        Examples:
            >>> servers = list(strategy.list_servers())
            >>> any(s.id == "default" for s in servers)
        """
        ...

    def list_tools(self, server_id: str) -> Iterable[McpTool]:
        """List tools exposed by a specific server.

        Args:
            server_id: Identifier of the target server.

        Returns:
            Iterable[McpTool]: An iterable of tool descriptors available on the
            server. Order is not guaranteed unless documented by the backend.

        Raises:
            Exception: Implementations may raise errors if the server is
            unreachable or does not exist.

        Examples:
            >>> tools = list(strategy.list_tools("server-1"))
            >>> assert any(t.name == "search" for t in tools)
        """
        ...

    def call_tool(
        self,
        server_id: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        agent_id: str | None = None,
    ) -> ToolCallResult:
        """Invoke a tool on a server with the provided arguments.

        Args:
            server_id: Identifier of the target server.
            tool_name: Name of the tool to call.
            args: JSON-serializable arguments adhering to the tool's schema.
            agent_id: Optional caller identity for auditing/policy contexts.

        Returns:
            ToolCallResult: A normalized result envelope indicating success and
            carrying the server's response payload.

        Raises:
            Exception: Implementations may raise errors for connectivity
            issues, tool-not-found, or backend failures.

        Examples:
            >>> result = strategy.call_tool("server-1", "search", {"q": "foo"})
            >>> assert result.ok
        """
        ...


@runtime_checkable
class AsyncStrategy(Protocol):
    """Protocol for asynchronous strategies.

    Implementations provide async list/invoke and optional SSE streaming
    generators for raw and parsed events.

    Examples:
        >>> tools = await strategy.list_tools("server-1")
        >>> res = await strategy.call_tool("server-1", "search", {"q": "bar"})
    """

    async def list_tools(self, server_id: str) -> List[McpTool]:
        """Asynchronously list tools for a specific server.

        Args:
            server_id: Identifier of the target server.

        Returns:
            List[McpTool]: A list of tool descriptors available on the server.

        Raises:
            Exception: Implementations may raise errors if the server is
            unreachable or does not exist.
        """
        ...

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> ToolCallResult:
        """Asynchronously invoke a tool on a server.

        Args:
            server_id: Identifier of the target server.
            tool_name: Name of the tool to call.
            args: JSON-serializable arguments adhering to the tool's schema.

        Returns:
            ToolCallResult: A normalized result envelope indicating success and
            carrying the server's response payload.

        Raises:
            Exception: Implementations may raise errors for connectivity
            issues, tool-not-found, or backend failures.
        """
        ...

    # Streaming methods removed; streaming may be provided by dedicated transports/services.


class StrategyCommonMixin:
    """Shared small utilities commonly used by concrete strategies."""

    def _infer_arguments(self, input_schema: Dict[str, Any]) -> List[ToolArgument]:
        """Infer a simplified list of `ToolArgument` from a JSON Schema object.

        Args:
            input_schema: A JSON Schema dictionary describing tool parameters.

        Returns:
            A list of `ToolArgument` entries derived from properties/required.
        """
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

    @staticmethod
    def _is_mutating_tool_name(name: str) -> bool:
        """Delegate to the module-level mutating name heuristic.

        Args:
            name: The tool name to inspect.

        Returns:
            True if the tool name is considered mutating by the heuristic.
        """
        return is_mutating_tool_name(name)
