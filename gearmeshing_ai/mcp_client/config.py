from __future__ import annotations

import os
from typing import Optional, Literal, cast

from pydantic import BaseModel, Field


class MCPConfig(BaseModel):
    """Configuration for the MCP client."""

    strategy: str = Field(
        default="gateway",
        description="High-level client strategy hint (backward compat).",
        examples=["gateway", "direct"],
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Base URL for HTTP transport (e.g., MCP gateway/server).",
        examples=["http://localhost:3004"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session/auth token to pass via 'mcp-session-id' header when using HTTP transport.",
    )
    timeout: int = Field(
        default=30,
        ge=1,
        le=600,
        description="Request timeout (seconds) for HTTP operations.",
        examples=[30],
    )
    policy_path: Optional[str] = Field(
        default=None,
        description="Optional path to a YAML file containing policy rules.",
    )
    transport: Literal["http", "stdio"] = Field(
        default="http",
        description="Transport type for the async strategy layer.",
        examples=["http"],
    )
    stdio_cmd: Optional[str] = Field(
        default=None,
        description="Executable (and args) to launch the MCP server via stdio transport.",
        examples=["./server_binary --flag"],
    )

    @staticmethod
    def from_env() -> "MCPConfig":
        return MCPConfig(
            strategy=os.getenv("MCP_STRATEGY", "gateway"),
            base_url=os.getenv("MCP_GATEWAY_URL"),
            session_id=os.getenv("MCP_SESSION_ID") or os.getenv("SLACK_BOT_TOKEN"),
            timeout=int(os.getenv("MCP_TIMEOUT", "30")),
            policy_path=os.getenv("MCP_POLICY_PATH"),
            transport=cast(Literal["http", "stdio"], os.getenv("MCP_TRANSPORT", "http")),
            stdio_cmd=os.getenv("MCP_STDIO_CMD"),
        )
