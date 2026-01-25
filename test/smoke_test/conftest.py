"""
Test infrastructure for smoke tests with real AI model calling and Docker Compose dependencies.

This module provides fixtures and utilities for smoke testing that:
- Uses real AI model connections to OpenAI, Anthropic, Google
- Uses Docker Compose with testcontainers for real database, cache, and external services
- Provides consistent test environment setup with real dependencies
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Optional
from unittest.mock import MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from testcontainers.compose import DockerCompose

# Disable LangSmith during smoke tests to prevent logging issues
os.environ["LANGSMITH_TRACING"] = "false"

# Disable logging to prevent file handle issues during tests
logging.disable(logging.CRITICAL)

from test.settings import test_settings

from gearmeshing_ai.agent_core.abstraction.initialization import setup_agent_abstraction

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Global variable to store the server settings patch
_server_settings_patch = None


@pytest.fixture(scope="session")
def _compose_env():
    """Set up environment variables for docker-compose.e2e.yml."""
    global _server_settings_patch
    prev = {}

    def _set(k: str, v: str) -> None:
        if k in os.environ:
            prev[k] = os.environ[k]
        os.environ[k] = v

    # PostgreSQL - use test settings defaults (with nested delimiter pattern)
    postgres_config = test_settings.database.postgres
    _set("DATABASE__POSTGRES__DB", postgres_config.db)
    _set("DATABASE__POSTGRES__USER", postgres_config.user)
    _set("DATABASE__POSTGRES__PASSWORD", postgres_config.password.get_secret_value())

    # MCP Gateway - use test settings defaults (with nested delimiter pattern)
    if hasattr(test_settings, "mcp") and hasattr(test_settings.mcp, "gateway"):
        mcp_gateway_config = test_settings.mcp.gateway
        _set("MCPGATEWAY__JWT_SECRET", mcp_gateway_config.jwt_secret.get_secret_value())
        _set("MCPGATEWAY__ADMIN_PASSWORD", mcp_gateway_config.admin_password.get_secret_value())
        _set("MCPGATEWAY__ADMIN_EMAIL", mcp_gateway_config.admin_email)
        _set("MCPGATEWAY__ADMIN_FULL_NAME", mcp_gateway_config.admin_full_name)
        _set("MCPGATEWAY__DB_URL", mcp_gateway_config.db_url.get_secret_value())
        _set("MCPGATEWAY__REDIS_URL", mcp_gateway_config.redis_url)

    # Set test database URL to use the PostgreSQL from Docker Compose
    postgres_url = "postgresql+asyncpg://ai_dev:changeme@localhost:5432/ai_dev"
    _set("DATABASE__URL", postgres_url)
    _set("DATABASE__ENABLE_POSTGRES_TESTS", "1")

    # Also patch the server settings to use the same database URL
    from unittest.mock import patch

    import gearmeshing_ai.server.core.config as server_config

    # Create a mock settings object with the correct database URL
    mock_settings = MagicMock()
    mock_settings.database.url = postgres_url
    mock_settings.database.postgres = postgres_config

    # Store the patch globally
    _server_settings_patch = patch.object(server_config, "settings", mock_settings)

    try:
        yield
    finally:
        # Restore original environment variables
        for k in list(os.environ.keys()):
            if k.startswith(("DATABASE__", "MCPGATEWAY__", "MCP__", "MQ__")):
                if k in prev:
                    os.environ[k] = prev[k]
                else:
                    os.environ.pop(k, None)


@pytest.fixture(scope="session")
def compose_stack(_compose_env):
    """Start docker-compose stack for testing."""
    global _server_settings_patch
    compose = DockerCompose(str(PROJECT_ROOT / "test"))

    # Start the server settings patch
    if _server_settings_patch:
        _server_settings_patch.start()

    try:
        compose.start()

        # Wait for services to be ready
        print("Waiting for services to start...")
        time.sleep(10)

        # Check if PostgreSQL is ready using the correct API
        stdout, stderr, exit_code = compose.exec_in_container(
            ["pg_isready", "-U", "ai_dev", "-d", "ai_dev"], service_name="postgres"
        )
        if exit_code != 0:
            print(f"PostgreSQL not ready: {stderr}")
            # Give it more time and try again
            time.sleep(10)
            stdout, stderr, exit_code = compose.exec_in_container(
                ["pg_isready", "-U", "ai_dev", "-d", "ai_dev"], service_name="postgres"
            )
            if exit_code != 0:
                raise RuntimeError(f"PostgreSQL failed to start: {stderr}")

        print("Services are ready!")

        # Run database migrations to create the required tables
        print("Running database migrations...")
        from alembic import command
        from alembic.config import Config

        # Configure Alembic to use our test database
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", "postgresql+asyncpg://ai_dev:changeme@localhost:5432/ai_dev")

        # Run migrations
        command.upgrade(alembic_cfg, "head")
        print("Database migrations completed!")

        # Initialize the AI provider framework
        from gearmeshing_ai.agent_core.abstraction import get_agent_provider

        provider = get_agent_provider()
        provider.set_framework("pydantic_ai")
        print("AI provider framework initialized!")

        yield compose
    finally:
        if _server_settings_patch:
            _server_settings_patch.stop()
        compose.stop()


@pytest.fixture(scope="session")
def database_url(compose_stack) -> str:
    """Get the database URL from the running PostgreSQL container."""
    return "postgresql+asyncpg://ai_dev:changeme@localhost:5432/ai_dev"


@pytest.fixture(scope="session")
def redis_url(compose_stack) -> str:
    """Get the Redis URL from the running Redis container."""
    return "redis://localhost:6379"


@pytest.fixture(scope="session")
def async_engine(database_url):
    """Create async engine for database operations."""
    engine = create_async_engine(database_url, echo=False)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(scope="session")
def async_session_maker(async_engine):
    """Create async session maker."""
    return sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def async_session(async_session_maker):
    """Create async session for database operations."""
    async with async_session_maker() as session:
        yield session


@pytest.fixture(scope="session", autouse=True)
async def initialize_ai_agent_provider() -> AsyncGenerator[Optional[Any], None]:
    """Initialize the AI agent provider for all E2E tests."""
    try:
        # Set up the agent abstraction layer
        provider = setup_agent_abstraction(validate_api_keys=False)
        yield provider
    except Exception as e:
        # If setup fails, tests will be skipped appropriately
        print(f"Warning: Failed to initialize AI agent provider: {e}")
        yield None
    finally:
        # Clean up LangSmith to prevent logging issues
        import os

        # Remove LangSmith environment variables to prevent background thread issues
        for key in ["LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"]:
            if key in os.environ:
                del os.environ[key]


# Markers for different test types


def pytest_configure(config: Any) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "e2e_ai: mark test as E2E AI test (requires real AI API keys)")
    config.addinivalue_line("markers", "openai_only: mark test as OpenAI-only test")
    config.addinivalue_line("markers", "anthropic_only: mark test as Anthropic-only test")
    config.addinivalue_line("markers", "google_only: mark test as Google-only test")
    config.addinivalue_line("markers", "multi_provider: mark test as multi-provider test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add skip conditions."""
    from test.settings import test_settings

    for item in items:
        # Skip OpenAI-only tests if no API key
        if "openai_only" in item.keywords and not test_settings.ai_provider.openai.api_key:
            item.add_marker(pytest.mark.skip(reason="OpenAI API key not configured"))

        # Skip Anthropic-only tests if no API key
        if "anthropic_only" in item.keywords and not test_settings.ai_provider.anthropic.api_key:
            item.add_marker(pytest.mark.skip(reason="Anthropic API key not configured"))

        # Skip Google-only tests if no API key
        if "google_only" in item.keywords and not test_settings.ai_provider.google.api_key:
            item.add_marker(pytest.mark.skip(reason="Google API key not configured"))

        # Skip multi-provider tests if no API keys
        if "multi_provider" in item.keywords:
            has_keys = any(
                [
                    test_settings.ai_provider.openai.api_key,
                    test_settings.ai_provider.anthropic.api_key,
                    test_settings.ai_provider.google.api_key,
                ]
            )
            if not has_keys:
                item.add_marker(pytest.mark.skip(reason="No AI provider API keys configured"))


def pytest_sessionfinish(session, exitstatus):
    """Clean up LangSmith background thread after test session."""
    try:
        # Force cleanup of LangSmith background thread
        import langsmith._internal._background_thread as bg_thread

        if hasattr(bg_thread, "_tracing_thread"):
            thread = bg_thread._tracing_thread
            if thread and thread.is_alive():
                # Wait a moment for the thread to finish
                thread.join(timeout=1.0)
    except Exception:
        # Ignore cleanup errors
        pass

    # Clean up environment variables
    import os

    for key in ["LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"]:
        if key in os.environ:
            del os.environ[key]
