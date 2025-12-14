from __future__ import annotations

import httpx
import pytest

from gearmeshing_ai.info_provider.mcp.gateway_api.client import GatewayApiClient
from gearmeshing_ai.info_provider.mcp.provider import (
    AsyncMCPInfoProvider,
    MCPInfoProvider,
)
from gearmeshing_ai.info_provider.mcp.schemas.config import (
    GatewayConfig,
    McpClientConfig,
)
from gearmeshing_ai.info_provider.mcp.transport.mcp import SseMCPTransport


def _first_gateway_id(gateway_client: GatewayApiClient) -> str:
    gateways = gateway_client.admin.gateway.list()
    assert gateways and len(gateways) >= 1
    gw0 = gateways[0]
    return gw0.id or getattr(gw0, "slug", None) or gw0.name


@pytest.mark.e2e
def test_sync_provider_gateway_lists_tools(gateway_client: GatewayApiClient) -> None:
    cfg = McpClientConfig(gateway=GatewayConfig(base_url=gateway_client.base_url, auth_token=gateway_client.auth_token))

    provider = MCPInfoProvider.from_config(
        cfg,
        mcp_transport=SseMCPTransport(),
        gateway_mgmt_client=gateway_client._client,  # reuse mgmt client
        gateway_http_client=httpx.Client(base_url=gateway_client.base_url),
    )

    try:
        server_id = _first_gateway_id(gateway_client)
        tools = provider.list_tools(server_id)
        assert tools and len(tools) >= 1
        assert all(t.name for t in tools)

        # Pagination smoke (limit=1)
        page = provider.list_tools_page(server_id, limit=1)
        assert page.items and len(page.items) >= 1
    finally:
        # close the extra http client we created
        provider._strategies  # just to avoid lints; http client will be GC'd


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_async_provider_gateway_lists_tools(gateway_client: GatewayApiClient) -> None:
    cfg = McpClientConfig(gateway=GatewayConfig(base_url=gateway_client.base_url, auth_token=gateway_client.auth_token))

    http_client = httpx.AsyncClient(base_url=cfg.gateway.base_url)
    sse_client = httpx.AsyncClient(base_url=cfg.gateway.base_url)

    try:
        provider = await AsyncMCPInfoProvider.from_config(
            cfg,
            mcp_transport=SseMCPTransport(),
            gateway_mgmt_client=gateway_client._client,
            gateway_http_client=http_client,
            gateway_sse_client=sse_client,
        )

        server_id = _first_gateway_id(gateway_client)
        tools = await provider.list_tools(server_id)
        assert tools and len(tools) >= 1
        assert all(t.name for t in tools)

        # Pagination smoke (limit=1)
        page = await provider.list_tools_page(server_id, limit=1)
        assert page.items and len(page.items) >= 1
    finally:
        await http_client.aclose()
        await sse_client.aclose()
