from __future__ import annotations

from gearmeshing_ai.info_provider.base import AsyncMCPInfoProvider, MCPInfoProvider
from gearmeshing_ai.info_provider.client_async import AsyncMcpClient
from gearmeshing_ai.info_provider.client_sync import McpClient


def test_sync_client_conforms_runtime_protocol() -> None:
    c = McpClient(strategies=[])
    assert isinstance(c, MCPInfoProvider)


def test_async_client_conforms_runtime_protocol() -> None:
    c = AsyncMcpClient(strategies=[])
    assert isinstance(c, AsyncMCPInfoProvider)
