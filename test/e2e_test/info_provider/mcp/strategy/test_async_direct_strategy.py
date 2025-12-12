from __future__ import annotations

import pytest

from gearmeshing_ai.info_provider.mcp.schemas.config import ServerConfig
from gearmeshing_ai.info_provider.mcp.strategy.direct_async import (
    AsyncDirectMcpStrategy,
)
from gearmeshing_ai.info_provider.mcp.transport.mcp import SseMCPTransport


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_async_direct_strategy_lists_tools(clickup_base_url: str) -> None:
    strat = AsyncDirectMcpStrategy(
        servers=[ServerConfig(name="clickup", endpoint_url=clickup_base_url)],
        ttl_seconds=1.0,
        mcp_transport=SseMCPTransport(),
    )
    tools = await strat.list_tools("clickup")
    assert len(tools) >= 1
    assert all(t.name for t in tools)
