from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import AnyHttpUrl, Field

from .base import BaseSchema


class TransportType(str, Enum):
    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"
    SSE = "sse"


class ServerKind(str, Enum):
    DIRECT = "direct"
    GATEWAY = "gateway"


class McpServerRef(BaseSchema):
    id: str = Field(
        ...,
        description=(
            "Internal identifier for the MCP server within the client layer. "
            "This may map to a direct MCP server or a Gateway-registered server ID."
        ),
        min_length=1,
        max_length=128,
        examples=["clickup-mcp", "gateway:16f3f52c-4d4a-4a79-8d8e-7f02cdf0"],
    )
    display_name: str = Field(
        ...,
        description="Human-readable name used in logs and UI when referring to this MCP server.",
        min_length=1,
        max_length=128,
        examples=["ClickUp MCP", "GitHub MCP via Gateway"],
    )
    kind: ServerKind = Field(
        ...,
        description="Indicates whether this server is reached directly or through an MCP Gateway.",
        examples=[ServerKind.DIRECT],
    )
    transport: TransportType = Field(
        ...,
        description=(
            "Transport used by the underlying MCP server. "
            "STREAMABLE_HTTP is preferred; SSE and STDIO are kept for compatibility."
        ),
        examples=[TransportType.STREAMABLE_HTTP],
    )
    endpoint_url: AnyHttpUrl = Field(
        ...,
        description=(
            "Effective MCP endpoint URL visible to the client. "
            "For direct servers, this points at the MCP server itself; "
            "for Gateway servers, this points at the Gateway's /servers/{id}/mcp/ or /sse endpoint."
        ),
        examples=[
            "https://mcp-clickup.internal/mcp/",
            "https://gateway.internal/servers/16f3f52c/mcp/",
        ],
    )
    auth_token: Optional[str] = Field(
        None,
        description=(
            "Optional token that should be sent as an Authorization header " "when talking to this MCP server."
        ),
        min_length=1,
        max_length=512,
        examples=["Bearer ghp_exampletoken", "Bearer hg_123e4567"],
    )


class ToolArgument(BaseSchema):
    name: str = Field(..., description="Argument name", min_length=1, max_length=128)
    type: str = Field(..., description="JSON Schema type of the argument", min_length=1, max_length=64)
    required: bool = Field(False, description="Whether this argument is required by the tool")
    description: Optional[str] = Field(None, description="Human-friendly description of the argument.")


class McpTool(BaseSchema):
    name: str = Field(..., description="Tool name.", min_length=1, max_length=128)
    description: Optional[str] = Field(None, description="Short description of what the tool does.")
    mutating: bool = Field(
        default=False,
        description="Whether the tool is expected to mutate external state (used for read-only policies).",
    )
    arguments: List[ToolArgument] = Field(
        default_factory=list,
        description="List of arguments as a simplified domain view derived from schemas.",
    )
    raw_parameters_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw JSON schema for parameters as provided by MCP server or gateway.",
    )


class ToolCallResult(BaseSchema):
    ok: bool = Field(True, description="Whether the tool invocation succeeded.")
    data: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary JSON payload returned by the tool.")
    error: Optional[str] = Field(None, description="Error message if invocation failed.")
