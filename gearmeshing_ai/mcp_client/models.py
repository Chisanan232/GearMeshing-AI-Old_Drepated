from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class ToolMetadata(BaseModel):
    """Describes a tool exposed by an MCP server/gateway."""

    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolResult(BaseModel):
    """The normalized result of calling a tool."""

    ok: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    raw: Optional[Any] = None


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request model."""

    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[int | str] = None


class JSONRPCError(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None


class JSONRPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None
    id: Optional[int | str] = None
