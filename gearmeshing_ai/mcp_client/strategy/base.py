from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterable, List, Protocol, runtime_checkable

from gearmeshing_ai.mcp_client.schemas.core import McpServerRef, McpTool, ToolArgument, ToolCallResult


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

    async def stream_events(self, server_id: str, path: str = "/sse") -> AsyncIterator[str]: ...

    async def stream_events_parsed(self, server_id: str, path: str = "/sse") -> AsyncIterator[Dict[str, Any]]: ...


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
        n = name.lower()
        prefixes = ("create", "update", "delete", "remove", "post_", "put_", "patch_", "write", "set_")
        return n.startswith(prefixes)
