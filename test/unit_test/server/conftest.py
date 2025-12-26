import gc
import logging
import os
import sys
from typing import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import JSON
from sqlalchemy.dialects.sqlite.base import SQLiteTypeCompiler
from sqlalchemy.ext.asyncio import AsyncSession

# Patch SQLite type compiler to handle JSONB
original_process = SQLiteTypeCompiler.process


def patched_process(self, type_, **kw):
    from sqlalchemy.dialects.postgresql import JSONB

    if isinstance(type_, JSONB):
        return self.process(JSON(), **kw)
    return original_process(self, type_, **kw)


SQLiteTypeCompiler.process = patched_process  # type: ignore[method-assign]

# Override database URL for testing before importing app
# Note: The Settings class uses 'database_url' as the field name (no alias)
os.environ["database_url"] = "sqlite+aiosqlite:///:memory:"

# Disable file logging during tests to prevent ResourceWarning
os.environ["ENABLE_FILE_LOGGING"] = "false"

# Remove any cached modules to force reimport with new database_url
for module_name in list(sys.modules.keys()):
    if module_name.startswith("gearmeshing_ai.server") or module_name.startswith("gearmeshing_ai.agent_core"):
        del sys.modules[module_name]

from gearmeshing_ai.agent_core.repos.models import Base
from gearmeshing_ai.server.core.database import get_session
from gearmeshing_ai.server.main import app


@pytest_asyncio.fixture(scope="session", autouse=True)
async def cleanup_event_loop():
    """Cleanup event loop and logging handlers after all tests to prevent ResourceWarning."""
    yield
    # Close all logging handlers to prevent ResourceWarning
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    # Force garbage collection to close any unclosed resources
    gc.collect()


# Use in-memory SQLite for testing
# Note: We use check_same_thread=False for SQLite with async
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(name="session")
async def session_fixture() -> AsyncGenerator[AsyncSession, None]:
    # Import the app's engine which is already configured with SQLite
    from gearmeshing_ai.server.core.database import (
        async_session_maker as app_session_maker,
    )
    from gearmeshing_ai.server.core.database import engine as app_engine

    try:
        # Create tables using Base from agent_core
        # First drop all tables to ensure a clean state, then create fresh
        async with app_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)

        async with app_session_maker() as session:
            yield session
    finally:
        # Clean up - always run even if test fails
        try:
            async with app_engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
        except Exception:
            pass

        # Dispose of the engine to close all connections
        await app_engine.dispose()

        # Force garbage collection
        gc.collect()


@pytest_asyncio.fixture(name="client")
async def client_fixture(session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Fixture providing an async HTTP client for testing."""

    # Override the get_session dependency to use the test session
    async def get_session_override() -> AsyncGenerator[AsyncSession, None]:
        yield session

    app.dependency_overrides[get_session] = get_session_override

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            yield client
    finally:
        app.dependency_overrides.clear()
