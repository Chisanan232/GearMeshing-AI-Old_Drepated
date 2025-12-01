from __future__ import annotations

from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

from gearmeshing_ai.mcp_client.schemas.core import (
    McpServerRef,
    McpTool,
    ToolArgument,
    ToolCallResult,
)


def is_mutating_tool_name(name: str) -> bool:
    n = name.lower()
    prefixes = ("create", "update", "delete", "remove", "post_", "put_", "patch_", "write", "set_")
    return n.startswith(prefixes)


@runtime_checkable
class SyncStrategy(Protocol):
    def list_servers(self) -> Iterable[McpServerRef]: ...

    def list_tools(self, server_id: str) -> Iterable[McpTool]: ...

    def call_tool(
        self,
        server_id: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        agent_id: str | None = None,
    ) -> ToolCallResult: ...


@runtime_checkable
class AsyncStrategy(Protocol):
    async def list_tools(self, server_id: str) -> List[McpTool]: ...

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> ToolCallResult: ...

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
    ) -> AsyncIterator[str]: ...

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
    ) -> AsyncIterator[Dict[str, Any]]: ...


class StrategyCommonMixin:

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

    @staticmethod
    def _is_mutating_tool_name(name: str) -> bool:
        return is_mutating_tool_name(name)
