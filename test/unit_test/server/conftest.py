import os
from typing import AsyncGenerator
from unittest.mock import patch

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.dialects.postgresql import JSONB

# Patch SQLite JSONB support BEFORE any imports that use the models
from sqlalchemy.dialects.sqlite import base as sqlite_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel.pool import StaticPool

# Patch the JSONB type to work with SQLite
original_jsonb_init = JSONB.__init__


def patched_jsonb_init(self, *args, **kwargs):
    original_jsonb_init(self, *args, **kwargs)
    # Add SQLite compilation support
    if not hasattr(self, "_sqlite_patched"):
        self._sqlite_patched = True


JSONB.__init__ = patched_jsonb_init  # type: ignore[method-assign]

# Patch the SQLiteTypeCompiler to handle JSONB
original_process = sqlite_base.SQLiteTypeCompiler.process


def patched_process(self, type_, **kw):
    if isinstance(type_, JSONB):
        return "JSON"
    if type_.__class__.__name__ == "JSON":
        return "JSON"
    return original_process(self, type_, **kw)


sqlite_base.SQLiteTypeCompiler.process = patched_process  # type: ignore[method-assign]

# Use in-memory SQLite for testing
# Note: We use check_same_thread=False for SQLite with async
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Set test database URL before importing app
os.environ["DATABASE_URL"] = TEST_DATABASE_URL


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """Create a test database engine once per session."""
    from gearmeshing_ai.agent_core.repos.models import Base as AgentCoreBase
    from sqlmodel import SQLModel

    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Import server models to register them with SQLModel
    import gearmeshing_ai.server.models.agent_config  # noqa: F401
    import gearmeshing_ai.server.models.chat_session  # noqa: F401

    # Create tables using both agent_core and server models
    async with engine.begin() as conn:
        # Create agent_core tables
        await conn.run_sync(AgentCoreBase.metadata.create_all)
        # Create server model tables
        await conn.run_sync(SQLModel.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest_asyncio.fixture(name="session")
async def session_fixture(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a new session for each test."""
    async_session_maker = sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session_maker() as session:  # type: ignore[attr-defined]
        yield session


@pytest_asyncio.fixture(name="client")
async def client_fixture(test_engine, session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client with mocked lifespan and overridden dependencies."""
    from gearmeshing_ai.agent_core.repos.sql import build_sql_repos
    from gearmeshing_ai.server.core.database import get_session
    from gearmeshing_ai.server.main import app
    from gearmeshing_ai.server.services.orchestrator import get_orchestrator

    # Create test session factory
    test_async_session_maker = sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

    async def get_session_override() -> AsyncGenerator[AsyncSession, None]:
        yield session

    async def get_orchestrator_override():
        """Override orchestrator to use test session factory."""
        from gearmeshing_ai.server.services.orchestrator import OrchestratorService

        orchestrator = OrchestratorService()
        # Replace the repos with ones using the test session factory
        orchestrator.repos = build_sql_repos(session_factory=test_async_session_maker)
        return orchestrator

    app.dependency_overrides[get_session] = get_session_override
    app.dependency_overrides[get_orchestrator] = get_orchestrator_override

    # Mock the lifespan to prevent database initialization during tests
    async def mock_lifespan(app):
        yield

    with patch("gearmeshing_ai.server.main.lifespan", mock_lifespan):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            yield client

    app.dependency_overrides.clear()
