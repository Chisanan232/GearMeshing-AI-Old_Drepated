from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import AnyHttpUrl, Field

from gearmeshing_ai.mcp_client.schemas.base import BaseSchema
from gearmeshing_ai.mcp_client.schemas.core import (
    McpServerRef,
    ServerKind,
    TransportType,
)


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
    description: Optional[str] = Field(
        None,
        description="Optional description for the server as stored in the Gateway.",
        max_length=1024,
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Optional list of tags associated with the server in the Gateway.",
    )
    visibility: Optional[str] = Field(
        default=None,
        description="Visibility level of the server (e.g., public/team/private).",
    )
    team_id: Optional[str] = Field(
        default=None,
        description="Team identifier that owns or manages this server entry.",
        max_length=128,
    )
    is_active: Optional[bool] = Field(
        default=None,
        description="Whether the server is currently active within the Gateway.",
    )
    metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metrics object reported by the Gateway for this server.",
    )

    def to_server_ref(self, base_url: str) -> McpServerRef:
        if self.transport == GatewayTransport.STREAMABLE_HTTP:
            t = TransportType.STREAMABLE_HTTP
        elif self.transport == GatewayTransport.SSE:
            t = TransportType.SSE
        else:
            t = TransportType.STDIO
        return McpServerRef(
            id=self.id,
            display_name=self.name,
            kind=ServerKind.GATEWAY,
            transport=t,
            endpoint_url=f"{base_url}/servers/{self.id}/mcp/",
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
    tags: Optional[List[str]] = Field(
        default=None,
        description="Optional tags to associate with the server upon creation.",
    )
    visibility: Optional[str] = Field(
        default=None,
        description="Desired visibility (e.g., team/private).",
    )
    team_id: Optional[str] = Field(
        default=None,
        description="Team ID to associate with this server.",
        max_length=128,
    )
