from __future__ import annotations

import os
import time
from typing import Iterable, List

import pytest
from testcontainers.core.container import DockerContainer

from gearmeshing_ai.info_provider.mcp.schemas.config import ServerConfig
from gearmeshing_ai.info_provider.mcp.strategy import DirectMcpStrategy
from gearmeshing_ai.info_provider.mcp.transport.mcp import SseMCPTransport


def clickup_image_tag() -> str:
    return os.getenv("GM_CLICKUP_MCP_TAG", "0.1.0")


def clickup_port() -> int:
    return int(os.getenv("GM_CLICKUP_MCP_PORT", "9000"))


def endpoint_candidates(host: str, port: int) -> List[str]:
    return [
        f"http://{host}:{port}/sse/sse",
    ]


def wait_clickup_ready(urls: Iterable[str], timeout: float = 30.0) -> str:
    last_err: Exception | None = None
    start = time.time()
    for base in urls:
        # quick immediate probe in case it's already up
        try:
            strat = DirectMcpStrategy(
                servers=[ServerConfig(name="clickup", endpoint_url=base)],
                ttl_seconds=1.0,
                mcp_transport=SseMCPTransport(),
            )
            _ = list(strat.list_tools("clickup"))
            return base
        except Exception as e:
            last_err = e
            continue
    while time.time() - start < timeout:
        for base in urls:
            try:
                strat = DirectMcpStrategy(
                    servers=[ServerConfig(name="clickup", endpoint_url=base)],
                    ttl_seconds=1.0,
                    mcp_transport=SseMCPTransport(),
                )
                _ = list(strat.list_tools("clickup"))
                return base
            except Exception as e:  # server not ready yet
                last_err = e
                time.sleep(0.5)
                continue
    if last_err:
        raise last_err
    raise RuntimeError("MCP server not ready and no error captured")


@pytest.fixture
def clickup_container() -> DockerContainer:
    tag = clickup_image_tag()
    port_int = clickup_port()
    token = os.getenv("CLICKUP_API_TOKEN") or os.getenv("GM_CLICKUP_API_TOKEN") or "e2e-test-token"

    container = (
        DockerContainer(f"chisanan232/clickup-mcp-server:{tag}")  # type: ignore[attr-defined]
        .with_exposed_ports(port_int)
        .with_env("SERVER_PORT", str(port_int))
        .with_env("CLICKUP_API_TOKEN", token)
    )
    container.start()
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture
def clickup_base_url(clickup_container: DockerContainer) -> str:
    port_int = clickup_port()
    host = clickup_container.get_container_host_ip()
    # use localhost resolution for stability
    port = int(clickup_container.get_exposed_port(port_int))
    base = wait_clickup_ready(endpoint_candidates("127.0.0.1", port), timeout=10.0)
    return base
