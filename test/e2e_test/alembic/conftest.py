"""Fixtures for Alembic migration tests."""

import os
import time
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from testcontainers.compose import DockerCompose

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@pytest.fixture(scope="session")
def _compose_env():
    """Set up environment variables for docker-compose.e2e.yml."""
    prev = {}

    def _set(k: str, v: str) -> None:
        if k in os.environ:
            prev[k] = os.environ[k]
        os.environ[k] = v

    # PostgreSQL
    _set("POSTGRES_DB", os.getenv("POSTGRES_DB", "ai_dev_test"))
    _set("POSTGRES_USER", os.getenv("POSTGRES_USER", "ai_dev"))
    _set("POSTGRES_PASSWORD", os.getenv("POSTGRES_PASSWORD", "changeme"))

    # MCP Gateway
    _set("MCPGATEWAY_JWT_SECRET", os.getenv("MCPGATEWAY_JWT_SECRET", "my-test-key"))
    _set("MCPGATEWAY_ADMIN_PASSWORD", os.getenv("MCPGATEWAY_ADMIN_PASSWORD", "adminpass"))
    _set("MCPGATEWAY_ADMIN_EMAIL", os.getenv("MCPGATEWAY_ADMIN_EMAIL", "admin@example.com"))
    _set("MCPGATEWAY_ADMIN_FULL_NAME", os.getenv("MCPGATEWAY_ADMIN_FULL_NAME", "Admin User"))
    _set(
        "MCPGATEWAY_DB_URL",
        os.getenv("MCPGATEWAY_DB_URL", "postgresql+psycopg://ai_dev:changeme@postgres:5432/ai_dev_test"),
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
    """Get test database URL."""
    return "postgresql+asyncpg://ai_dev:changeme@127.0.0.1:5432/ai_dev_test"


@pytest.fixture(scope="session")
def sync_database_url(compose_stack) -> str:
    """Get sync database URL."""
    return "postgresql://ai_dev:changeme@127.0.0.1:5432/ai_dev_test"


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
