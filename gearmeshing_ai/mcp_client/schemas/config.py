from __future__ import annotations
from typing import Optional

from pydantic import Field
from .base import BaseSchema


class MCPConfig(BaseSchema):
    """
    Minimal configuration for AsyncHTTPStrategy.

    Note: Kept intentionally simple to satisfy current unit tests. Can be
    expanded later per full spec (GatewayConfig, ServerConfig, etc.).
    """

    base_url: str = Field(
        ...,
        description=(
            "Base URL for the MCP server or gateway when using HTTP. For tests, a"
            " mock base URL may be provided (e.g., 'http://mock')."
        ),
        examples=[
            "http://localhost:8000",
            "http://mock",
        ],
        min_length=1,
        max_length=512,
    )


class GatewayConfig(BaseSchema):
    base_url: str = Field(
        ...,
        description="Base URL of the MCP Gateway (Context Forge) used by MCP client.",
        examples=["https://mcp-gateway.internal", "http://localhost:4444"],
        min_length=1,
        max_length=512,
    )
    auth_token: Optional[str] = Field(
        None,
        description="Optional bearer token sent as Authorization header when calling the Gateway.",
        examples=["hg_123e4567-e89b-12d3-a456-426614174000"],
        min_length=1,
        max_length=512,
    )
    request_timeout_seconds: float = Field(
        10.0,
        description="Per-request timeout in seconds when calling the Gateway REST API.",
        ge=0.1,
        le=120.0,
        examples=[5.0, 10.0],
    )


class ServerConfig(BaseSchema):
    name: str = Field(
        ...,
        description="Internal name for the directly-connected MCP server.",
        min_length=1,
        max_length=128,
        examples=["clickup-mcp", "github-mcp"],
    )
    endpoint_url: str = Field(
        ...,
        description=(
            "Endpoint URL for the MCP server (streamable HTTP preferred). For direct servers,"
            " this points at the MCP server itself."
        ),
        examples=["http://clickup-mcp:8000/mcp/", "https://mcp.example/mcp/"],
        min_length=1,
        max_length=512,
    )
    auth_token: Optional[str] = Field(
        None,
        description="Optional bearer token sent as Authorization header when calling the server.",
        examples=["Bearer ghp_exampletoken"],
        min_length=1,
        max_length=512,
    )


class McpClientConfig(BaseSchema):
    gateway: Optional[GatewayConfig] = Field(
        None,
        description="Optional Gateway configuration. If provided, Gateway strategy can be enabled.",
    )
    servers: list[ServerConfig] = Field(
        default_factory=list,
        description="List of directly connected MCP servers available to the client.",
    )
    default_request_timeout_seconds: float = Field(
        10.0,
        description="Default per-request timeout in seconds for strategies unless overridden.",
        ge=0.1,
        le=120.0,
        examples=[5.0, 10.0],
    )
    request_timeout_seconds: float = Field(
        10.0,
        description="Per-request timeout in seconds for HTTP calls.",
        ge=0.1,
        le=120.0,
        examples=[5.0, 10.0],
    )
    auth_token: Optional[str] = Field(
        None,
        description=(
            "Optional bearer token to send as Authorization header when invoking"
            " tools over HTTP."
        ),
        examples=["Bearer hg_123e4567"],
        min_length=1,
        max_length=512,
    )
