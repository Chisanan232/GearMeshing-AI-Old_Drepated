from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import Field

from .base import BaseSchema


class ToolMetadata(BaseSchema):
    """
    Minimal tool metadata used by strategies and tests.

    This aligns loosely with MCP tool descriptions and gateway listings.
    """

    name: str = Field(..., description="Tool name", min_length=1, max_length=128)
    description: Optional[str] = Field(
        None, description="Short human-readable description of the tool.", max_length=2048
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Parameters schema for the tool. When sourced from MCP servers, this" " may be called 'inputSchema'."
        ),
    )


class ToolResult(BaseSchema):
    """
    Result wrapper for tool invocation.
    """

    ok: bool = Field(True, description="Whether the tool invocation succeeded.")
    data: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary JSON payload returned by the tool.")
