from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import AnyHttpUrl, Field

from ..schemas.base import BaseSchema


class GatewayTransport(str, Enum):
    SSE = "SSE"
    STREAMABLE_HTTP = "STREAMABLEHTTP"
    STDIO = "STDIO"


class GatewayServer(BaseSchema):
    id: str = Field(
        ...,
        description=(
            "Unique identifier of the server as stored in the MCP Gateway. "
            "This ID is referenced in /servers/{server_id} paths."
        ),
        examples=["16f3f52c-4d4a-4a79-8d8e-7f02cdf0"],
    )
    name: str = Field(
        ...,
        description="Human-readable name assigned to the server in the Gateway.",
        min_length=1,
        max_length=128,
        examples=["clickup-mcp", "github-mcp"],
    )
    url: AnyHttpUrl = Field(
        ...,
        description=(
            "Base URL of the underlying MCP server configured in the Gateway. "
            "This points to the source MCP server, not the Gateway endpoint."
        ),
        examples=["http://clickup-mcp:8000/mcp/"],
    )
    transport: GatewayTransport = Field(
        ...,
        description=("Transport type used by the Gateway when talking to the underlying MCP server."),
        examples=[GatewayTransport.STREAMABLE_HTTP],
    )


class GatewayServerCreate(BaseSchema):
    name: str = Field(
        ...,
        description="Desired human-readable name for the server inside the Gateway.",
        min_length=1,
        max_length=128,
        examples=["clickup-mcp"],
    )
    url: AnyHttpUrl = Field(
        ...,
        description="Base URL of the MCP server to be registered in the Gateway.",
        examples=["http://clickup-mcp:8000/mcp/"],
    )
    transport: GatewayTransport = Field(
        ...,
        description="Transport used to connect the Gateway to the underlying MCP server.",
        examples=[GatewayTransport.STREAMABLE_HTTP],
    )
    auth_token: Optional[str] = Field(
        None,
        description="Optional token the Gateway should use when calling the underlying server.",
        min_length=1,
        max_length=512,
        examples=["Bearer ghp_exampletoken"],
    )
