from __future__ import annotations

import pytest

from gearmeshing_ai.info_provider.mcp.provider import AsyncMCPInfoProvider, MCPInfoProvider


class BadSyncStrategy:
    # Missing required methods from SyncStrategy
    pass


class BadAsyncStrategy:
    # Missing required async methods from AsyncStrategy
    pass


def test_sync_provider_rejects_non_protocol_strategy() -> None:
    with pytest.raises(TypeError):
        MCPInfoProvider(strategies=[BadSyncStrategy()])  # type: ignore[list-item]


@pytest.mark.asyncio
async def test_async_provider_rejects_non_protocol_strategy() -> None:
    with pytest.raises(TypeError):
        AsyncMCPInfoProvider(strategies=[BadAsyncStrategy()])  # type: ignore[list-item]
