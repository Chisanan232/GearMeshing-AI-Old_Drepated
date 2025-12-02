from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, cast

from pydantic import ConfigDict, Field, model_validator

from gearmeshing_ai.mcp_client.schemas.base import BaseSchema
from gearmeshing_ai.mcp_client.schemas.core import McpTool, ToolArgument, ToolCallResult

# Use a descriptive alias without recursive typing to avoid Pydantic recursion issues
JSONValue = Any


class ToolDescriptorDTO(BaseSchema):
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
    parameters: Dict[str, JSONValue] = Field(
        default_factory=dict,
        description="Arguments to pass to the tool as per its inputSchema.",
        examples=[{"id": "ISSUE-123"}],
    )


class ToolsListQuery(BaseSchema):
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
        data = self.model_dump(exclude_none=True, by_alias=True)
        params: Dict[str, str] = {}
        for k, v in data.items():
            params[k] = str(v)
        return params


class ToolsListPayloadDTO(BaseSchema):
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
        if isinstance(v, dict):
            # Preserve full dict in data; propagate ok if present
            nv: Dict[str, Any] = {"data": cast(Dict[str, Any], v)}
            ok_val = v.get("ok")
            if isinstance(ok_val, bool):
                nv["ok"] = ok_val
            return nv
        # Non-dict body -> wrap under result
        return {"ok": True, "data": {"result": v}}

    def to_tool_call_result(self) -> ToolCallResult:
        ok = True if self.ok is None else bool(self.ok)
        return ToolCallResult(ok=ok, data=self.data)
