"""Fixtures for Alembic migration tests."""

import os
import time
from pathlib import Path
from test.settings import test_settings

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from testcontainers.compose import DockerCompose

from gearmeshing_ai.server.core.config import settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@pytest.fixture(scope="session")
def _compose_env():
    """Set up environment variables for docker-compose.e2e.yml."""
    prev = {}

    def _set(k: str, v: str) -> None:
        if k in os.environ:
            prev[k] = os.environ[k]
        os.environ[k] = v

    # PostgreSQL - use server settings defaults (with nested delimiter pattern)
    postgres_config = settings.database.postgres
    _set("DATABASE__POSTGRES__DB", postgres_config.db)
    _set("DATABASE__POSTGRES__USER", postgres_config.user)
    _set("DATABASE__POSTGRES__PASSWORD", postgres_config.password.get_secret_value())

    # MCP Gateway - use server settings defaults (with nested delimiter pattern)
    mcp_gateway_config = settings.mcp.gateway
    _set("MCPGATEWAY__JWT_SECRET", mcp_gateway_config.jwt_secret.get_secret_value())
    _set("MCPGATEWAY__ADMIN_PASSWORD", mcp_gateway_config.admin_password.get_secret_value())
    _set("MCPGATEWAY__ADMIN_EMAIL", mcp_gateway_config.admin_email)
    _set("MCPGATEWAY__ADMIN_FULL_NAME", mcp_gateway_config.admin_full_name)
    _set("MCPGATEWAY__DB_URL", mcp_gateway_config.db_url.get_secret_value())
    _set("MCPGATEWAY__REDIS_URL", mcp_gateway_config.redis_url)

    # ClickUp MCP - use server settings defaults (with nested delimiter pattern)
    clickup_config = settings.mcp.clickup
    _set("MCP__CLICKUP__SERVER_HOST", clickup_config.host)
    _set("MCP__CLICKUP__SERVER_PORT", str(clickup_config.port))
    _set("MCP__CLICKUP__MCP_TRANSPORT", clickup_config.mcp_transport)
    # Token must be provided by env for real runs; default for CI/e2e
    _set(
        "MCP__CLICKUP__API_TOKEN",
        clickup_config.api_token.get_secret_value() if clickup_config.api_token else "e2e-test-token",
    )
    _set("MQ__BACKEND", "redis")

    try:
        yield
    finally:
        for k in list(
            {
                "DATABASE__POSTGRES__DB",
                "DATABASE__POSTGRES__USER",
                "DATABASE__POSTGRES__PASSWORD",
                "MCPGATEWAY__JWT_SECRET",
                "MCPGATEWAY__ADMIN_PASSWORD",
                "MCPGATEWAY__ADMIN_EMAIL",
                "MCPGATEWAY__ADMIN_FULL_NAME",
                "MCPGATEWAY__DB_URL",
                "MCPGATEWAY__REDIS_URL",
                "MCP__CLICKUP__SERVER_HOST",
                "MCP__CLICKUP__SERVER_PORT",
                "MCP__CLICKUP__MCP_TRANSPORT",
                "MCP__CLICKUP__API_TOKEN",
                "MQ__BACKEND",
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
    postgres_config = test_settings.database.postgres
    return f"postgresql+asyncpg://{postgres_config.user}:{postgres_config.password.get_secret_value()}@127.0.0.1:{postgres_config.port}/{postgres_config.db}"


@pytest.fixture(scope="session")
def sync_database_url(compose_stack) -> str:
    """Get sync test database URL from test settings."""
    postgres_config = test_settings.database.postgres
    return f"postgresql://{postgres_config.user}:{postgres_config.password.get_secret_value()}@127.0.0.1:{postgres_config.port}/{postgres_config.db}"


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
