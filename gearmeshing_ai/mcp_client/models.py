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
