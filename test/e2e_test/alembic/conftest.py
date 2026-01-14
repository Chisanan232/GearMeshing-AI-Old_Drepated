"""Fixtures for Alembic migration tests."""

import os
import time
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from testcontainers.compose import DockerCompose

from gearmeshing_ai.server.core.config import settings
from test.settings import test_settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@pytest.fixture(scope="session")
def _compose_env():
    """Set up environment variables for docker-compose.e2e.yml."""
    prev = {}

    def _set(k: str, v: str) -> None:
        if k in os.environ:
            prev[k] = os.environ[k]
        os.environ[k] = v

    # PostgreSQL - use server settings defaults
    postgres_config = settings.postgres
    _set("POSTGRES_DB", postgres_config.db)
    _set("POSTGRES_USER", postgres_config.user)
    _set("POSTGRES_PASSWORD", postgres_config.password)

    # MCP Gateway - use server settings defaults
    mcp_gateway_config = settings.mcp_gateway
    _set("MCPGATEWAY_JWT_SECRET", mcp_gateway_config.jwt_secret)
    _set("MCPGATEWAY_ADMIN_PASSWORD", mcp_gateway_config.admin_password)
    _set("MCPGATEWAY_ADMIN_EMAIL", mcp_gateway_config.admin_email)
    _set("MCPGATEWAY_ADMIN_FULL_NAME", mcp_gateway_config.admin_full_name)
    _set("MCPGATEWAY_DB_URL", mcp_gateway_config.db_url)
    _set("MCPGATEWAY_REDIS_URL", mcp_gateway_config.redis_url)

    # ClickUp MCP - use server settings defaults
    clickup_config = settings.clickup
    _set("CLICKUP_SERVER_HOST", clickup_config.server_host)
    _set("CLICKUP_SERVER_PORT", str(clickup_config.server_port))
    _set("CLICKUP_MCP_TRANSPORT", clickup_config.mcp_transport)
    # Token must be provided by env for real runs; default for CI/e2e
    _set("CLICKUP_API_TOKEN", clickup_config.api_token or "e2e-test-token")
    _set("MQ_BACKEND", "redis")

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
            }
        ):
            if k in prev:
                os.environ[k] = prev[k]
            else:
                os.environ.pop(k, None)


@pytest.fixture(scope="session")
def compose_stack(_compose_env):
    """Start docker-compose stack for testing."""
    project_root = Path(os.getcwd()).resolve()
    compose = DockerCompose(str(project_root), compose_file_name="./docker-compose.e2e.yml")
    compose.start()

    # Wait for PostgreSQL to be ready
    time.sleep(5)

    try:
        yield compose
    finally:
        compose.stop()


@pytest.fixture(scope="session")
def database_url(compose_stack) -> str:
    """Get async test database URL from test settings."""
    postgres_config = test_settings.postgres
    return f"postgresql+asyncpg://{postgres_config.user}:{postgres_config.password}@127.0.0.1:{postgres_config.port}/{postgres_config.db}"


@pytest.fixture(scope="session")
def sync_database_url(compose_stack) -> str:
    """Get sync test database URL from test settings."""
    postgres_config = test_settings.postgres
    return f"postgresql://{postgres_config.user}:{postgres_config.password}@127.0.0.1:{postgres_config.port}/{postgres_config.db}"


@pytest.fixture
async def async_engine(database_url: str):
    """Create async SQLAlchemy engine for tests."""
    engine = create_async_engine(database_url, echo=False)
    yield engine
    await engine.dispose()


@pytest.fixture
async def async_session_factory(async_engine):
    """Create async session factory."""
    return sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def async_session(async_session_factory):
    """Create async session instance."""
    async with async_session_factory() as session:
        yield session
