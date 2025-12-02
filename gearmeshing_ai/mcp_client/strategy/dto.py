from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..schemas.base import BaseSchema
from ..schemas.core import McpTool, ToolArgument, ToolCallResult

# Use a descriptive alias without recursive typing to avoid Pydantic recursion issues
JSONValue = Any


class ToolDescriptorDTO(BaseSchema):
    model_config = ConfigDict(extra="allow")
    name: str
    description: Optional[str] = None
    title: Optional[str] = None
    icons: Optional[List[Dict[str, JSONValue]]] = None
    input_schema: Dict[str, JSONValue] = Field(default_factory=dict, alias="inputSchema")
    x_mutating: Optional[bool] = Field(default=None, alias="x-mutating")

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


class ToolsListEnvelopeDTO(BaseSchema):
    items: List[ToolDescriptorDTO]


class ToolInvokeRequestDTO(BaseSchema):
    parameters: Dict[str, JSONValue] = Field(default_factory=dict)


class FlexibleDTO(BaseModel):
    model_config = ConfigDict(extra="allow")


class ToolInvokeResponseDTO(FlexibleDTO):
    ok: Optional[bool] = None


class ToolsListResultDTO(BaseSchema):
    tools: List[ToolDescriptorDTO]
    next_cursor: Optional[str] = Field(default=None, alias="nextCursor")


def extract_tool_descriptors(data: Any) -> List[ToolDescriptorDTO]:
    items: List[ToolDescriptorDTO] = []
    if isinstance(data, dict):
        try:
            env = ToolsListEnvelopeDTO.model_validate(data)
            return list(env.items)
        except Exception:
            pass
        try:
            res = ToolsListResultDTO.model_validate(data)
            return list(res.tools)
        except Exception:
            pass
        raw = data.get("items") if isinstance(data.get("items"), list) else []
        for x in raw or []:
            if isinstance(x, dict):
                try:
                    items.append(ToolDescriptorDTO.model_validate(x))
                except Exception:
                    continue
    elif isinstance(data, list):
        for x in data:
            if isinstance(x, dict):
                try:
                    items.append(ToolDescriptorDTO.model_validate(x))
                except Exception:
                    continue
    return items


class ToolsListPayloadDTO(BaseSchema):
    tools: List[ToolDescriptorDTO]
    next_cursor: Optional[str] = Field(default=None, alias="nextCursor")

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
                nv = dict(v)
                nv["tools"] = items
                nv.pop("items", None)
                return nv
        return v


class ToolInvokePayloadDTO(BaseSchema):
    ok: Optional[bool] = None
    data: Dict[str, JSONValue] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, v: Any):
        if isinstance(v, dict):
            # Preserve full dict in data; propagate ok if present
            nv = {"data": v}
            if "ok" in v:
                nv["ok"] = v.get("ok")
            return nv
        # Non-dict body -> wrap under result
        return {"ok": True, "data": {"result": v}}

    def to_tool_call_result(self) -> ToolCallResult:
        ok = True if self.ok is None else bool(self.ok)
        return ToolCallResult(ok=ok, data=self.data)
