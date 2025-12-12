from __future__ import annotations

import pytest

from gearmeshing_ai.info_provider.mcp.provider import (
    AsyncMCPInfoProvider,
    MCPInfoProvider,
)
from gearmeshing_ai.info_provider.mcp.schemas.config import (
    McpClientConfig,
    ServerConfig,
)
from gearmeshing_ai.info_provider.mcp.transport.mcp import SseMCPTransport


@pytest.mark.e2e
def test_sync_provider_direct_lists_tools(clickup_base_url: str) -> None:
    cfg = McpClientConfig(servers=[ServerConfig(name="clickup", endpoint_url=clickup_base_url)])
    provider = MCPInfoProvider.from_config(cfg, mcp_transport=SseMCPTransport())
    tools = provider.list_tools("clickup")
    assert len(tools) >= 1
    assert all(t.name for t in tools)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_async_provider_direct_lists_tools(clickup_base_url: str) -> None:
    cfg = McpClientConfig(servers=[ServerConfig(name="clickup", endpoint_url=clickup_base_url)])
    provider = await AsyncMCPInfoProvider.from_config(cfg, mcp_transport=SseMCPTransport())
    tools = await provider.list_tools("clickup")
    assert len(tools) >= 1
    assert all(t.name for t in tools)
