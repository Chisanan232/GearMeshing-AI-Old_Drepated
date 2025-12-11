from __future__ import annotations

from gearmeshing_ai.info_provider.mcp.base import AsyncMCPInfoProvider, MCPInfoProvider
from gearmeshing_ai.info_provider.mcp.provider import AsyncMcpClient, McpClient


def test_sync_client_conforms_runtime_protocol() -> None:
    c = McpClient(strategies=[])
    assert isinstance(c, MCPInfoProvider)


def test_async_client_conforms_runtime_protocol() -> None:
    c = AsyncMcpClient(strategies=[])
    assert isinstance(c, AsyncMCPInfoProvider)
