import os
import uuid
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
    from sqlmodel import SQLModel

    from gearmeshing_ai.core.database.base import Base as AgentCoreBase

    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Import core database entities to register them with SQLModel
    import gearmeshing_ai.core.database.entities.agent_configs  # noqa: F401
    import gearmeshing_ai.core.database.entities.chat_sessions  # noqa: F401
    import gearmeshing_ai.core.database.entities.agent_runs  # noqa: F401
    import gearmeshing_ai.core.database.entities.agent_events  # noqa: F401
    import gearmeshing_ai.core.database.entities.tool_invocations  # noqa: F401
    import gearmeshing_ai.core.database.entities.approvals  # noqa: F401
    import gearmeshing_ai.core.database.entities.checkpoints  # noqa: F401
    import gearmeshing_ai.core.database.entities.policies  # noqa: F401
    import gearmeshing_ai.core.database.entities.usage_ledger  # noqa: F401

    # Create tables using both agent_core and core database entities
    async with engine.begin() as conn:
        # Create agent_core and core database entity tables
        await conn.run_sync(AgentCoreBase.metadata.create_all)
        # Create core database entity tables
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
    from gearmeshing_ai.core.database.repositories.bundle import build_sql_repos_from_session
    from gearmeshing_ai.core.database import get_session
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
        # Replace the repos with ones using the test session
        orchestrator.repos = build_sql_repos_from_session(session=session)
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


@pytest_asyncio.fixture(name="client_with_mocked_runs")
async def client_with_mocked_runs_fixture(test_engine, session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client with mocked orchestrator for runs tests.

    This fixture prevents actual agent execution by mocking the orchestrator's
    create_run, get_run, list_runs, and cancel_run methods.
    """
    from gearmeshing_ai.core.database.repositories.bundle import build_sql_repos_from_session
    from gearmeshing_ai.core.models.domain import AgentRun, AgentRunStatus
    from gearmeshing_ai.core.database import get_session
    from gearmeshing_ai.server.main import app
    from gearmeshing_ai.server.services.orchestrator import get_orchestrator

    # Create test session factory
    test_async_session_maker = sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

    # Store runs in memory for test tracking - shared across all calls within this fixture
    runs_store: dict[str, AgentRun] = {}
    cancelled_runs: set[str] = set()

    async def get_session_override() -> AsyncGenerator[AsyncSession, None]:
        yield session

    async def get_orchestrator_override():
        """Override orchestrator with mocked run methods."""
        from gearmeshing_ai.server.services.orchestrator import OrchestratorService

        orchestrator = OrchestratorService()
        orchestrator.repos = build_sql_repos_from_session(session=session)

        # Mock create_run to avoid actual execution
        async def mock_create_run(run: AgentRun) -> AgentRun:
            run.id = str(uuid.uuid4())
            run.status = AgentRunStatus.pending
            runs_store[run.id] = run
            return run

        # Mock get_run
        async def mock_get_run(run_id: str) -> AgentRun | None:
            if run_id in runs_store:
                run = runs_store[run_id]
                # Return cancelled status if it was cancelled
                if run_id in cancelled_runs:
                    run.status = AgentRunStatus.cancelled
                return run
            return None

        # Mock list_runs
        async def mock_list_runs(tenant_id: str | None = None, limit: int = 100, offset: int = 0) -> list[AgentRun]:
            result = []
            for run_id, run in list(runs_store.items())[offset : offset + limit]:
                if tenant_id is None or run.tenant_id == tenant_id:
                    # Update status if cancelled
                    if run_id in cancelled_runs:
                        run.status = AgentRunStatus.cancelled
                    result.append(run)
            return result

        # Mock cancel_run
        async def mock_cancel_run(run_id: str) -> None:
            if run_id in runs_store:
                cancelled_runs.add(run_id)
                runs_store[run_id].status = AgentRunStatus.cancelled

        # Mock get_run_events
        async def mock_get_run_events(run_id: str, limit: int = 100) -> list:
            return []

        orchestrator.create_run = mock_create_run
        orchestrator.get_run = mock_get_run
        orchestrator.list_runs = mock_list_runs
        orchestrator.cancel_run = mock_cancel_run
        orchestrator.get_run_events = mock_get_run_events

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
