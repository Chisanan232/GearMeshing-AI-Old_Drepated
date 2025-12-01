from __future__ import annotations

from gearmeshing_ai.mcp_client.base import SyncClientProtocol, AsyncClientProtocol
from gearmeshing_ai.mcp_client.client_sync import McpClient
from gearmeshing_ai.mcp_client.client_async import AsyncMcpClient


def test_sync_client_conforms_runtime_protocol() -> None:
    c = McpClient(strategies=[])
    assert isinstance(c, SyncClientProtocol)


def test_async_client_conforms_runtime_protocol() -> None:
    c = AsyncMcpClient(strategies=[])
    assert isinstance(c, AsyncClientProtocol)
