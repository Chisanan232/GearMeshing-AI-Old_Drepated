from __future__ import annotations

import pytest

from gearmeshing_ai.info_provider.mcp.client_async import AsyncMcpClient
from gearmeshing_ai.info_provider.mcp.client_sync import McpClient


class BadSyncStrategy:
    # Missing required methods from SyncStrategy
    pass


class BadAsyncStrategy:
    # Missing required async methods from AsyncStrategy
    pass


def test_sync_client_rejects_non_protocol_strategy() -> None:
    with pytest.raises(TypeError):
        McpClient(strategies=[BadSyncStrategy()])  # type: ignore[list-item]


@pytest.mark.asyncio
async def test_async_client_rejects_non_protocol_strategy() -> None:
    with pytest.raises(TypeError):
        AsyncMcpClient(strategies=[BadAsyncStrategy()])  # type: ignore[list-item]
