"""DTO models used by MCP strategies (Gateway/Direct).

These Pydantic models centralize serialization/deserialization for strategy
HTTP interactions so that strategies remain thin. They also provide helpers to
map DTOs into core domain models used by the client layer.

Guidelines:
- Prefer adding validators here instead of ad-hoc parsing in strategies.
- Keep aliasing and coercions explicit to match MCP server behaviors.
- Provide small, focused helpers (e.g., `to_mcp_tool`, `to_params`).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, cast

from pydantic import ConfigDict, Field, model_validator

from gearmeshing_ai.mcp_client.schemas.base import BaseSchema
from gearmeshing_ai.mcp_client.schemas.core import McpTool, ToolArgument, ToolCallResult

# Use a descriptive alias without recursive typing to avoid Pydantic recursion issues
JSONValue = Any


class ToolDescriptorDTO(BaseSchema):
    """Descriptor of a tool returned by an MCP server.

    Purpose:
    - Capture raw tool metadata including JSON Schema for inputs.
    - Provide mapping to domain `McpTool` via `to_mcp_tool`.

    Usage:
    - Parsed from server responses via `ToolsListPayloadDTO`.
    - Call `to_mcp_tool(infer_arguments, is_mutating_tool_name)` to convert.
    """
    model_config = ConfigDict(extra="allow")
    name: str = Field(
        ..., description="Tool name as exposed by the MCP server (unique within a server).", examples=["get_issue"]
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of what the tool does.",
        examples=["Fetch an issue by ID from the tracker"],
    )
    title: Optional[str] = Field(
        default=None, description="Optional short display title for the tool.", examples=["Get Issue"]
    )
    icons: Optional[List[Dict[str, JSONValue]]] = Field(
        default=None,
        description="Optional list of icon descriptors (vendor-defined schema).",
        examples=[[{"type": "emoji", "value": "🛠️"}]],
    )
    input_schema: Dict[str, JSONValue] = Field(
        default_factory=dict,
        alias="inputSchema",
        description="JSON Schema for tool input parameters (object with properties/required).",
        examples=[{"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}],
    )
    x_mutating: Optional[bool] = Field(
        default=None,
        alias="x-mutating",
        description="Vendor extension indicating tool mutates state. If omitted, inferred by name heuristic.",
        examples=[True, False],
    )

    def to_mcp_tool(
        self,
        infer_arguments: Callable[[Dict[str, Any]], List[ToolArgument]],
        is_mutating_tool_name: Callable[[str], bool],
    ) -> McpTool:
        """Map this descriptor to the `McpTool` domain model.

        - Arguments are inferred from `input_schema` via the provided `infer_arguments` callback.
        - Mutating flag is decided with precedence: explicit `x-mutating` → schema `x-mutating` → heuristic name.
        """
        schema: Dict[str, Any] = dict(self.input_schema or {})
        explicit = self.x_mutating
        if explicit is None and isinstance(schema, dict):
            explicit = schema.get("x-mutating")
        if explicit is True:
            is_mut = True
        elif explicit is False:
            is_mut = False
        else:
            is_mut = is_mutating_tool_name(self.name)
        return McpTool(
            name=self.name,
            description=self.description,
            mutating=is_mut,
            arguments=infer_arguments(schema),
            raw_parameters_schema=schema,
        )


class ToolInvokeRequestDTO(BaseSchema):
    """Payload for invoking a tool: `{parameters: {...}}`.

    Build using keyword args or dict literal; dump via `model_dump(by_alias=True, mode="json")` when sending.
    """
    parameters: Dict[str, JSONValue] = Field(
        default_factory=dict,
        description="Arguments to pass to the tool as per its inputSchema.",
        examples=[{"id": "ISSUE-123"}],
    )


class ToolsListQuery(BaseSchema):
    """Query parameters for listing tools with optional pagination."""
    cursor: Optional[str] = Field(
        default=None,
        description="Opaque pagination cursor returned by a previous tools list response.",
        examples=["abc123"],
    )
    limit: Optional[int] = Field(
        default=None,
        description="Maximum number of tools to return for this page, if supported by server.",
        ge=1,
        examples=[50],
    )

    def to_params(self) -> Dict[str, str]:
        """Serialize to HTTP query params, excluding None values.

        Values are stringified to be safe for `httpx`/requests.
        """
        data = self.model_dump(exclude_none=True, by_alias=True)
        params: Dict[str, str] = {}
        for k, v in data.items():
            params[k] = str(v)
        return params


class ToolsListPayloadDTO(BaseSchema):
    """Normalized payload for a tools listing response.

    Accepts multiple wire shapes:
    - List of tool descriptors → `{tools: [...]}`
    - `{items: [...]}` → `{tools: [...]}`
    - `{tools: [...]}` preserved
    """
    tools: List[ToolDescriptorDTO] = Field(
        ..., description="Normalized list of tools regardless of source response shape."
    )
    next_cursor: Optional[str] = Field(
        default=None,
        alias="nextCursor",
        description="Pagination cursor if available.",
        examples=["abc123"],
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, v: Any):
        """Coerce supported wire shapes into the normalized model structure."""
        if isinstance(v, list):
            return {"tools": v}
        if isinstance(v, dict):
            if isinstance(v.get("tools"), list):
                return v
            items = v.get("items")
            if isinstance(items, list):
                nv: Dict[str, Any] = dict(v)
                nv["tools"] = items
                nv.pop("items", None)
                return nv
        return v


class ToolInvokePayloadDTO(BaseSchema):
    """Normalized envelope for a tool invocation response.

    Accepts:
    - Dict bodies, preserving keys in `data` and lifting a top-level `ok` if present.
    - Non-dict bodies, wrapped under `data={"result": ...}` with implicit `ok=True`.
    """
    ok: Optional[bool] = Field(
        default=None,
        description="Optional success flag; if None, treated as success unless response indicates otherwise.",
        examples=[True, False],
    )
    data: Dict[str, JSONValue] = Field(
        default_factory=dict,
        description="Normalized response payload. If raw body is a dict, preserved as-is under 'data'; otherwise wrapped as {'result': <value>}.",
        examples=[{"result": {"title": "Example"}}],
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, v: Any):
        """Coerce raw response into `{ok?, data}` while excluding `ok` from `data`."""
        if isinstance(v, dict):
            # Preserve dict in data but do not carry 'ok' inside data payload
            raw = cast(Dict[str, Any], v)
            data_only: Dict[str, Any] = dict(raw)
            if "ok" in data_only:
                data_only = {k: val for k, val in data_only.items() if k != "ok"}
            nv: Dict[str, Any] = {"data": data_only}
            ok_val = raw.get("ok")
            if isinstance(ok_val, bool):
                nv["ok"] = ok_val
            return nv
        # Non-dict body -> wrap under result
        return {"ok": True, "data": {"result": v}}

    def to_tool_call_result(self) -> ToolCallResult:
        """Convert normalized envelope into a `ToolCallResult` domain object."""
        ok = True if self.ok is None else bool(self.ok)
        return ToolCallResult(ok=ok, data=self.data)
