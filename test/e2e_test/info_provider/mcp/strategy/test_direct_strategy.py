from __future__ import annotations

import os
import time
from typing import List

import pytest
from testcontainers.core.container import DockerContainer

from gearmeshing_ai.info_provider.mcp.schemas.config import ServerConfig
from gearmeshing_ai.info_provider.mcp.strategy import DirectMcpStrategy
from gearmeshing_ai.info_provider.mcp.transport.mcp import SseMCPTransport


def _endpoint_candidates(host: str, port: int) -> List[str]:
    return [
        f"http://{host}:{port}/sse/sse",
    ]


def _wait_http_ready(urls: List[str], timeout: float = 30.0) -> str:
    last_err: Exception | None = None
    start = time.time()
    while time.time() - start < timeout:
        for base in urls:
            try:
                strat = DirectMcpStrategy(servers=[ServerConfig(name="clickup", endpoint_url=base)], ttl_seconds=1.0, mcp_transport=SseMCPTransport())
                _ = list(strat.list_tools("clickup"))
                return base
            except Exception as e:  # server not ready yet
                last_err = e
                time.sleep(0.5)
                continue
    if last_err:
        raise last_err
    raise RuntimeError("MCP server not ready and no error captured")


@pytest.mark.e2e
def test_clickup_direct_strategy_lists_tools() -> None:
    tag = os.getenv("GM_CLICKUP_MCP_TAG", "0.1.0")
    port_int = int(os.getenv("GM_CLICKUP_MCP_PORT", "9000"))

    container = (
        DockerContainer(f"chisanan232/clickup-mcp-server:{tag}")  # type: ignore[attr-defined]
        .with_exposed_ports(port_int)
        .with_env("SERVER_PORT", str(port_int))
        .with_env("CLICKUP_API_TOKEN", "e2e-test-token")
    )
    container.start()
    try:
        host = container.get_container_host_ip()
        port = int(container.get_exposed_port(port_int))

        base = _wait_http_ready(_endpoint_candidates("127.0.0.1", port), timeout=5.0)

        strat = DirectMcpStrategy(
            servers=[ServerConfig(name="clickup", endpoint_url=base)],
            ttl_seconds=1.0,
            mcp_transport=SseMCPTransport(),
        )
        servers = list(strat.list_servers())
        assert any(s.id == "clickup" for s in servers)

        tools = list(strat.list_tools("clickup"))
        assert len(tools) >= 1
        assert all(t.name for t in tools)
    finally:
        container.stop()
