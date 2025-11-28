from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ToolMetadata:
    """Describes a tool exposed by an MCP server/gateway."""

    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ToolResult:
    """The normalized result of calling a tool."""

    ok: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    raw: Optional[Any] = None
