from __future__ import annotations

import pytest

from gearmeshing_ai.info_provider.mcp.schemas.config import ServerConfig
from gearmeshing_ai.info_provider.mcp.strategy import DirectMcpStrategy
from gearmeshing_ai.info_provider.mcp.transport import SseMCPTransport


@pytest.mark.e2e
def test_direct_strategy_lists_tools(clickup_base_url: str) -> None:
    strat = DirectMcpStrategy(
        servers=[ServerConfig(name="clickup", endpoint_url=clickup_base_url)],
        ttl_seconds=1.0,
        mcp_transport=SseMCPTransport(),
    )
    servers = list(strat.list_servers())
    assert any(s.id == "clickup" for s in servers)

    tools = list(strat.list_tools("clickup"))
    assert len(tools) >= 1
    assert all(t.name for t in tools)
