from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import os


@dataclass(frozen=True)
class MCPConfig:
    """Configuration for the MCP client.

    Attributes:
        strategy: Which transport strategy to use: 'gateway' or 'direct'.
        base_url: Base URL for HTTP-based strategies.
        session_id: Optional session/auth token header (e.g., 'mcp-session-id').
        timeout: Request timeout in seconds.
    """

    strategy: str = "gateway"
    base_url: Optional[str] = None
    session_id: Optional[str] = None
    timeout: int = 30

    @staticmethod
    def from_env() -> "MCPConfig":
        return MCPConfig(
            strategy=os.getenv("MCP_STRATEGY", "gateway"),
            base_url=os.getenv("MCP_GATEWAY_URL"),
            session_id=os.getenv("MCP_SESSION_ID") or os.getenv("SLACK_BOT_TOKEN"),
            timeout=int(os.getenv("MCP_TIMEOUT", "30")),
        )
