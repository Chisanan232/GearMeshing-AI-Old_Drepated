from __future__ import annotations

import pytest

from gearmeshing_ai.info_provider.mcp.gateway_api.client import GatewayApiClient
from gearmeshing_ai.info_provider.mcp.strategy.gateway_async import (
    AsyncGatewayMcpStrategy,
)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_gateway_strategy_async_lists_tools(gateway_client_with_register_servers: GatewayApiClient) -> None:
    # Determine a valid server id from admin gateway list
    gateways = gateway_client_with_register_servers.admin.gateway.list()
    assert gateways and len(gateways) >= 1
    server_id = gateways[0].id or gateways[0].slug or gateways[0].name

    strat = AsyncGatewayMcpStrategy(gateway_client_with_register_servers)

    tools = await strat.list_tools(server_id)
    assert tools and len(tools) >= 1
    assert all(t.name for t in tools)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_gateway_strategy_async_list_tools_page(gateway_client_with_register_servers: GatewayApiClient) -> None:
    gateways = gateway_client_with_register_servers.admin.gateway.list()
    assert gateways and len(gateways) >= 1
    server_id = gateways[0].id or gateways[0].slug or gateways[0].name

    strat = AsyncGatewayMcpStrategy(gateway_client_with_register_servers)

    page1 = await strat.list_tools_page(server_id, limit=1)
    assert page1.items and len(page1.items) >= 1
    # If pagination info available, optionally fetch next page
    if page1.next_cursor:
        page2 = await strat.list_tools_page(server_id, cursor=page1.next_cursor, limit=1)
        assert page2.items and len(page2.items) >= 1
