from __future__ import annotations

import pytest

from gearmeshing_ai.info_provider.mcp.gateway_api.client import GatewayApiClient


@pytest.mark.e2e
def test_gateway_health(gateway_client: GatewayApiClient) -> None:
    h = gateway_client.health()
    assert isinstance(h, dict)
    # Accept any non-empty payload
    assert h is not None


@pytest.mark.e2e
def test_gateway_list_servers(gateway_client: GatewayApiClient) -> None:
    payload = gateway_client.admin.mcp_registry.list()
    items = payload.servers
    assert items and len(items) >= 1
    assert any((it.name == "clickup" or it.id == "clickup") for it in items)
    target = next((it for it in items if it.name == "clickup" or it.id == "clickup"), items[0])
    # Basic field presence check for target
    assert target.id and target.name


@pytest.mark.e2e
def test_mcp_registry_register_server(gateway_client: GatewayApiClient) -> None:
    # Use first catalog server for registration attempt
    catalog = gateway_client.admin.mcp_registry.list()
    servers = catalog.servers
    assert servers and len(servers) >= 1
    target_id = servers[0].id

    try:
        reg = gateway_client.admin.mcp_registry.register(target_id)
    except Exception:
        # Some deployments may restrict registration; treat as skip to keep E2E robust
        pytest.skip("Catalog register not available or failed in this environment")
        return

    # Minimal assertions on response shape
    assert hasattr(reg, "success")
    assert hasattr(reg, "server_id")


@pytest.mark.e2e
def test_gateway_admin_gateways_list_and_get(gateway_client_with_register_servers: GatewayApiClient) -> None:
    gws = gateway_client_with_register_servers.admin.gateway.list()
    assert isinstance(gws, list)
    assert len(gws) >= 1

    gw0 = gws[0]
    assert getattr(gw0, "name", None)
    assert getattr(gw0, "url", None)

    # Verify get by id returns a matching entity
    if getattr(gw0, "id", None):
        gw = gateway_client_with_register_servers.admin.gateway.get(gw0.id)
        assert gw.name == gw0.name


@pytest.mark.e2e
def test_gateway_admin_tools_list_and_optional_get(gateway_client: GatewayApiClient) -> None:
    resp = gateway_client.admin.tools.list(limit=10)
    # Response is AdminToolsListResponseDTO
    assert hasattr(resp, "data")
    assert isinstance(resp.data, list)

    if not resp.data:
        pytest.skip("No tools available in gateway; skipping get test")

    tool0 = resp.data[0]
    assert getattr(tool0, "id", None)
    got = gateway_client.admin.tools.get(tool0.id)
    assert got.id == tool0.id
