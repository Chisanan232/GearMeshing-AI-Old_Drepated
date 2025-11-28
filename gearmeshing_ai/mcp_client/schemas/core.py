from __future__ import annotations

from typing import Any, Dict, Optional, Literal

from pydantic import BaseModel, Field


class ToolMetadata(BaseModel):
    """Describes a tool exposed by an MCP server/gateway."""

    name: str = Field(
        ..., description="Unique tool name exposed by the MCP server.", min_length=1, examples=["echo", "search"]
    )
    description: Optional[str] = Field(
        default=None, description="Human-readable summary of what the tool does.", examples=["Echo tool"]
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON Schema-like parameters accepted by the tool (shape defined by MCP server).",
        examples=[{"text": {"type": "string"}}],
    )


class ToolResult(BaseModel):
    """The normalized result of calling a tool."""

    ok: bool = Field(
        ..., description="Whether the tool invocation succeeded (True) or failed (False).", examples=[True]
    )
    data: Optional[Any] = Field(
        default=None, description="Normalized result payload returned by the tool when ok=True."
    )
    error: Optional[str] = Field(
        default=None, description="Error message when ok=False.", examples=["HTTP 500: Internal Server Error"]
    )
    raw: Optional[Any] = Field(
        default=None, description="Raw response payload from transport/server prior to normalization."
    )


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request model."""

    jsonrpc: Literal["2.0"] = Field(
        default="2.0", description="JSON-RPC protocol version (must be '2.0').", examples=["2.0"]
    )
    method: str = Field(
        ..., description="RPC method name.", min_length=1, examples=["tools/list", "tools/invoke"]
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None, description="Parameters for the method call as a JSON object."
    )
    id: Optional[int | str] = Field(
        default=None, description="Identifier established by the client for matching responses."
    )


class JSONRPCError(BaseModel):
    code: int = Field(
        ..., description="Error code as defined by JSON-RPC or server-specific codes.", examples=[-32601]
    )
    message: str = Field(
        ..., description="Short error message summarizing the failure.", min_length=1, examples=["Method not found"]
    )
    data: Optional[Any] = Field(
        default=None, description="Optional additional info about the error."
    )


class JSONRPCResponse(BaseModel):
    jsonrpc: Literal["2.0"] = Field(
        default="2.0", description="JSON-RPC protocol version (must be '2.0').", examples=["2.0"]
    )
    result: Optional[Any] = Field(
        default=None, description="Result for a successful call. Must be omitted when error is set."
    )
    error: Optional[JSONRPCError] = Field(
        default=None, description="Error for a failed call. Must be omitted when result is set."
    )
    id: Optional[int | str] = Field(
        default=None, description="Identifier matching the request id or null for notifications."
    )
