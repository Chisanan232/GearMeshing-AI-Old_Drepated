from __future__ import annotations

from gearmeshing_ai.info_provider.mcp.base import BaseAsyncMCPInfoProvider, BaseMCPInfoProvider
from gearmeshing_ai.info_provider.mcp.provider import AsyncMCPInfoProvider, MCPInfoProvider


def test_sync_provider_conforms_runtime_protocol() -> None:
    c = MCPInfoProvider(strategies=[])
    assert isinstance(c, BaseMCPInfoProvider)


def test_async_provider_conforms_runtime_protocol() -> None:
    c = AsyncMCPInfoProvider(strategies=[])
    assert isinstance(c, BaseAsyncMCPInfoProvider)
