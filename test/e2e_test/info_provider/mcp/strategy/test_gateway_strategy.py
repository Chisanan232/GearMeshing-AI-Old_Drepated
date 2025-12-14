from __future__ import annotations

import pytest

from gearmeshing_ai.info_provider.mcp.gateway_api.client import GatewayApiClient
from gearmeshing_ai.info_provider.mcp.strategy.gateway import GatewayMcpStrategy
from gearmeshing_ai.info_provider.mcp.transport.mcp import SseMCPTransport


@pytest.mark.e2e
def test_gateway_strategy_sync_lists_servers(gateway_client_with_register_servers: GatewayApiClient) -> None:
    strat = GatewayMcpStrategy(gateway_client_with_register_servers, mcp_transport=SseMCPTransport())

    servers = list(strat.list_servers())
    assert servers and len(servers) >= 1


@pytest.mark.e2e
def test_gateway_strategy_sync_lists_servers_and_tools(gateway_client_with_register_servers: GatewayApiClient) -> None:
    strat = GatewayMcpStrategy(gateway_client_with_register_servers, mcp_transport=SseMCPTransport())

    servers = list(strat.list_servers())
    assert servers and len(servers) >= 1

    # Prefer named/known server; otherwise use first
    sid = None
    for s in servers:
        if s.display_name == "clickup" or s.id == "clickup":
            sid = s.id
            break
    if not sid:
        sid = servers[0].id

    tools = list(strat.list_tools(sid))
    assert tools and len(tools) >= 1
    assert all(t.name for t in tools)


@pytest.mark.e2e
def test_gateway_strategy_sync_list_tools_page(gateway_client_with_register_servers: GatewayApiClient) -> None:
    strat = GatewayMcpStrategy(gateway_client_with_register_servers, mcp_transport=SseMCPTransport())

    servers = list(strat.list_servers())
    assert servers and len(servers) >= 1

    # Choose a valid server id
    sid = None
    for s in servers:
        if s.display_name == "clickup" or s.id == "clickup":
            sid = s.id
            break
    if not sid:
        sid = servers[0].id

    page = strat.list_tools_page(sid, limit=1)
    assert page.items and len(page.items) >= 1
