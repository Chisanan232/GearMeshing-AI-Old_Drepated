from __future__ import annotations

import os
import time
import httpx
from pathlib import Path
from typing import Iterable, List

import pytest
from testcontainers.compose import DockerCompose

from gearmeshing_ai.info_provider.mcp.schemas.config import ServerConfig
from gearmeshing_ai.info_provider.mcp.strategy import DirectMcpStrategy
from gearmeshing_ai.info_provider.mcp.transport.mcp import SseMCPTransport
from gearmeshing_ai.info_provider.mcp.gateway_api import GatewayApiClient


def clickup_port() -> int:
    return int(os.getenv("CLICKUP_SERVER_PORT", os.getenv("GM_CLICKUP_MCP_PORT", "9000")))


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


@pytest.fixture(scope="session")
def _compose_env() -> Iterable[None]:
    # Provide env vars required by docker-compose_e2e.yml services
    prev: dict[str, str] = {}

    def _set(k: str, v: str) -> None:
        if k in os.environ:
            prev[k] = os.environ[k]
        os.environ[k] = v

    # PostgreSQL
    _set("POSTGRES_DB", os.getenv("POSTGRES_DB", "mcp"))
    _set("POSTGRES_USER", os.getenv("POSTGRES_USER", "mcp"))
    _set("POSTGRES_PASSWORD", os.getenv("POSTGRES_PASSWORD", "mcp"))

    # MCP Gateway *IBM/mcp-context-forge*
    _set("MCPGATEWAY_JWT_SECRET", os.getenv("MCPGATEWAY_JWT_SECRET", "my-test-key"))
    _set("MCPGATEWAY_ADMIN_PASSWORD", os.getenv("MCPGATEWAY_ADMIN_PASSWORD", "adminpass"))
    _set("MCPGATEWAY_ADMIN_EMAIL", os.getenv("MCPGATEWAY_ADMIN_EMAIL", "admin@example.com"))
    _set("MCPGATEWAY_ADMIN_FULL_NAME", os.getenv("MCPGATEWAY_ADMIN_FULL_NAME", "Admin User"))
    _set(
        "MCPGATEWAY_DB_URL",
        os.getenv("MCPGATEWAY_DB_URL", "postgresql+psycopg://mcp:mcp@postgres:5432/mcp"),
    )
    _set("MCPGATEWAY_REDIS_URL", os.getenv("MCPGATEWAY_REDIS_URL", "redis://redis:6379/0"))

    # ClickUp MCP
    _set("CLICKUP_SERVER_HOST", os.getenv("CLICKUP_SERVER_HOST", "0.0.0.0"))
    _set("CLICKUP_SERVER_PORT", os.getenv("CLICKUP_SERVER_PORT", "8082"))
    _set("CLICKUP_MCP_TRANSPORT", os.getenv("CLICKUP_MCP_TRANSPORT", "sse"))
    # Token must be provided by env for real runs; default for CI/e2e
    _set("CLICKUP_API_TOKEN", os.getenv("CLICKUP_API_TOKEN", os.getenv("GM_CLICKUP_API_TOKEN", "e2e-test-token")))
    _set("MQ_BACKEND", os.getenv("MQ_BACKEND", "redis"))

    try:
        yield
    finally:
        for k in list(
            {
                "POSTGRES_DB",
                "POSTGRES_USER",
                "POSTGRES_PASSWORD",
                "MCPGATEWAY_JWT_SECRET",
                "MCPGATEWAY_ADMIN_PASSWORD",
                "MCPGATEWAY_ADMIN_EMAIL",
                "MCPGATEWAY_ADMIN_FULL_NAME",
                "MCPGATEWAY_DB_URL",
                "MCPGATEWAY_REDIS_URL",
                "CLICKUP_SERVER_HOST",
                "CLICKUP_SERVER_PORT",
                "CLICKUP_MCP_TRANSPORT",
                "CLICKUP_API_TOKEN",
                "MQ_BACKEND",
            }
        ):
            if k in prev:
                os.environ[k] = prev[k]
            else:
                os.environ.pop(k, None)


@pytest.fixture(scope="session")
def compose_stack(_compose_env: Iterable[None]) -> Iterable[DockerCompose]:
    # Repo root is the CWD when running tests; compose file is at the root
    project_root = Path(os.getcwd()).resolve()
    compose = DockerCompose(str(project_root), compose_file_name="./docker-compose_e2e.yml")
    compose.start()
    try:
        yield compose
    finally:
        compose.stop()


@pytest.fixture
def clickup_container(compose_stack: DockerCompose) -> DockerCompose:
    return compose_stack


@pytest.fixture
def clickup_base_url(clickup_container: DockerCompose) -> str:
    port_int = clickup_port()
    base = wait_clickup_ready(endpoint_candidates("127.0.0.1", port_int), timeout=20.0)
    return base


def gateway_port() -> int:
    return 4444


def _write_catalog_for_gateway(clickup_base_url: str) -> Path:
    # Deprecated: compose uses a static catalog file; keep function for compatibility
    return Path("./configs/mcp_gateway/mcp-catalog_e2e.yml").resolve()


def _wait_gateway_ready(base_url: str, timeout: float = 30.0) -> None:
    start = time.time()
    last: Exception | None = None
    while time.time() - start < timeout:
        try:
            user = os.getenv("MCPGATEWAY_ADMIN_EMAIL", "admin@example.com")
            secret = os.getenv("MCPGATEWAY_JWT_SECRET", "my-test-key")
            token = GatewayApiClient.generate_bearer_token(jwt_secret_key=secret, username=user)
            r = httpx.get(f"{base_url}/health", headers={"Authorization": token}, timeout=3.0)
            if r.status_code == 200:
                return
        except Exception as e:
            last = e
        time.sleep(0.5)
    if last:
        raise last
    raise RuntimeError("Gateway not ready and no error captured")


@pytest.fixture(scope="session")
def gateway_client(compose_stack: DockerCompose):
    base = f"http://127.0.0.1:{gateway_port()}"
    # Generate token once
    secret = os.getenv("MCPGATEWAY_JWT_SECRET", "my-test-key")
    user = os.getenv("MCPGATEWAY_ADMIN_EMAIL", "admin@example.com")
    token = GatewayApiClient.generate_bearer_token(jwt_secret_key=secret, username=user)

    mgmt_client = httpx.Client(base_url=base)
    client = GatewayApiClient(base, client=mgmt_client, auth_token=token)

    # Ensure health before yielding
    start = time.time()
    last_err: Exception | None = None
    while time.time() - start < 30.0:
        try:
            h = client.health()
            if h:
                break
        except Exception as e:
            last_err = e
            time.sleep(0.5)
            continue
    if last_err and time.time() - start >= 30.0:
        raise last_err

    try:
        yield client
    finally:
        try:
            mgmt_client.close()
        except Exception:
            pass


@pytest.fixture
def gateway_container(compose_stack: DockerCompose, clickup_base_url: str) -> DockerCompose:
    return compose_stack


@pytest.fixture
def gateway_base_url(gateway_container: DockerCompose, gateway_client) -> str:
    return gateway_client.base_url
