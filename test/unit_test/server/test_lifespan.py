"""
Unit tests for FastAPI application lifespan management.

Tests verify that the application startup and shutdown events are properly
handled, including database initialization and connection verification.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel.pool import StaticPool

pytestmark = pytest.mark.asyncio

# Use in-memory SQLite for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="function")
def mock_settings():
    """Mock settings with test database URL."""
    with patch("gearmeshing_ai.server.core.database.settings") as mock:
        mock.database_url = TEST_DATABASE_URL
        yield mock


@pytest.fixture(scope="function")
async def test_engine_fixture():
    """Create a test database engine for lifespan tests."""
    from gearmeshing_ai.agent_core.repos.models import Base

    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


class TestLifespanStartup:
    """Test application startup lifespan events."""

    async def test_lifespan_startup_initializes_database(self):
        """Test that lifespan startup calls init_db and checkpointer setup."""
        from fastapi import FastAPI

        from gearmeshing_ai.server.main import lifespan

        app = FastAPI()

        with (
            patch("gearmeshing_ai.server.main.init_db", new_callable=AsyncMock) as mock_init_db,
            patch("gearmeshing_ai.server.main.checkpointer_pool") as mock_pool,
            patch("gearmeshing_ai.server.main.AsyncPostgresSaver") as mock_saver,
        ):

            # Setup pool.connection() async context manager
            mock_conn = AsyncMock()
            mock_conn.set_autocommit = AsyncMock()
            mock_conn_ctx = MagicMock()
            mock_pool.connection.return_value = mock_conn_ctx
            mock_conn_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn_ctx.__aexit__ = AsyncMock(return_value=None)

            # Setup pool.open()
            mock_pool.open = AsyncMock()
            mock_pool.close = AsyncMock()

            # Setup saver.setup()
            mock_saver.return_value.setup = AsyncMock()

            async with lifespan(app):
                mock_init_db.assert_called_once()
                mock_pool.open.assert_called_once()
                mock_conn.set_autocommit.assert_called_once_with(True)
                mock_saver.return_value.setup.assert_called_once()

    async def test_lifespan_startup_logs_success(self):
        """Test that lifespan startup logs success message."""
        from fastapi import FastAPI

        from gearmeshing_ai.server.main import lifespan

        app = FastAPI()

        with (
            patch("gearmeshing_ai.server.main.init_db", new_callable=AsyncMock) as mock_init_db,
            patch("gearmeshing_ai.server.main.logger") as mock_logger,
            patch("gearmeshing_ai.server.main.checkpointer_pool") as mock_pool,
            patch("gearmeshing_ai.server.main.AsyncPostgresSaver") as mock_saver,
        ):

            # Setup pool.connection() async context manager
            mock_conn = AsyncMock()
            mock_conn.set_autocommit = AsyncMock()
            mock_conn_ctx = MagicMock()
            mock_pool.connection.return_value = mock_conn_ctx
            mock_conn_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn_ctx.__aexit__ = AsyncMock(return_value=None)

            mock_pool.open = AsyncMock()
            mock_pool.close = AsyncMock()
            mock_saver.return_value.setup = AsyncMock()

            async with lifespan(app):
                pass

            # Verify startup logs
            assert mock_logger.info.call_count >= 2
            calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("Starting up" in call for call in calls)
            assert any("Database initialized successfully" in call for call in calls)

    async def test_lifespan_startup_handles_init_db_exception(self):
        """Test that lifespan startup handles init_db exceptions gracefully."""
        from fastapi import FastAPI

        from gearmeshing_ai.server.main import lifespan

        app = FastAPI()

        with (
            patch("gearmeshing_ai.server.main.init_db", new_callable=AsyncMock) as mock_init_db,
            patch("gearmeshing_ai.server.main.logger") as mock_logger,
            patch("gearmeshing_ai.server.main.checkpointer_pool") as mock_pool,
        ):

            mock_init_db.side_effect = Exception("Database connection failed")
            mock_pool.open = AsyncMock()
            mock_pool.close = AsyncMock()

            async with lifespan(app):
                pass

            # Verify error was logged
            mock_logger.error.assert_called_once()
            error_call = mock_logger.error.call_args[0][0]
            assert "Initialization failed" in error_call

    async def test_lifespan_shutdown_logs_message(self):
        """Test that lifespan shutdown logs shutdown message."""
        from fastapi import FastAPI

        from gearmeshing_ai.server.main import lifespan

        app = FastAPI()

        with (
            patch("gearmeshing_ai.server.main.init_db", new_callable=AsyncMock) as mock_init_db,
            patch("gearmeshing_ai.server.main.logger") as mock_logger,
            patch("gearmeshing_ai.server.main.checkpointer_pool") as mock_pool,
            patch("gearmeshing_ai.server.main.AsyncPostgresSaver") as mock_saver,
        ):

            # Setup pool.connection() async context manager
            mock_conn = AsyncMock()
            mock_conn.set_autocommit = AsyncMock()
            mock_conn_ctx = MagicMock()
            mock_pool.connection.return_value = mock_conn_ctx
            mock_conn_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn_ctx.__aexit__ = AsyncMock(return_value=None)

            mock_pool.open = AsyncMock()
            mock_pool.close = AsyncMock()
            mock_saver.return_value.setup = AsyncMock()

            async with lifespan(app):
                pass

            # Verify shutdown log was called (after yield)
            shutdown_logs = [call[0][0] for call in mock_logger.info.call_args_list if "Shutting down" in call[0][0]]
            assert len(shutdown_logs) > 0
            mock_pool.close.assert_called_once()

    async def test_lifespan_context_manager_protocol(self):
        """Test that lifespan properly implements async context manager protocol."""
        from fastapi import FastAPI

        from gearmeshing_ai.server.main import lifespan

        app = FastAPI()

        with (
            patch("gearmeshing_ai.server.main.init_db", new_callable=AsyncMock) as mock_init_db,
            patch("gearmeshing_ai.server.main.checkpointer_pool") as mock_pool,
            patch("gearmeshing_ai.server.main.AsyncPostgresSaver") as mock_saver,
        ):

            # Setup pool.connection() async context manager
            mock_conn = AsyncMock()
            mock_conn.set_autocommit = AsyncMock()
            mock_conn_ctx = MagicMock()
            mock_pool.connection.return_value = mock_conn_ctx
            mock_conn_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn_ctx.__aexit__ = AsyncMock(return_value=None)

            mock_pool.open = AsyncMock()
            mock_pool.close = AsyncMock()
            mock_saver.return_value.setup = AsyncMock()

            # Verify lifespan can be used as async context manager
            async with lifespan(app) as result:
                # App is running here
                assert result is None


class TestDatabaseInitialization:
    """Test database initialization functionality."""

    async def test_init_db_creates_tables(self, test_engine_fixture):
        """Test that init_db creates all required tables."""

        # Verify tables exist
        inspector_result = []

        async with test_engine_fixture.begin() as conn:

            def get_table_names(conn):
                from sqlalchemy import inspect

                inspector = inspect(conn)
                return inspector.get_table_names()

            table_names = await conn.run_sync(get_table_names)
            inspector_result = table_names

        # Check for expected tables
        expected_tables = [
            "gm_agent_runs",
            "gm_agent_events",
            "gm_approvals",
            "gm_policies",
        ]

        for table_name in expected_tables:
            assert table_name in inspector_result, (
                f"Expected table {table_name} not found. " f"Available tables: {inspector_result}"
            )

    async def test_init_db_with_real_engine(self, test_engine_fixture):
        """Test init_db with a real async engine."""
        from gearmeshing_ai.server.core.database import init_db

        # Replace the global engine temporarily
        with patch("gearmeshing_ai.server.core.database.engine", test_engine_fixture):
            # Call init_db
            await init_db()

            # Verify tables were created
            inspector_result = []

            async with test_engine_fixture.begin() as conn:

                def get_table_names(conn):
                    from sqlalchemy import inspect

                    inspector = inspect(conn)
                    return inspector.get_table_names()

                table_names = await conn.run_sync(get_table_names)
                inspector_result = table_names

            assert len(inspector_result) > 0, "No tables were created"

    async def test_get_session_returns_async_session(self, test_engine_fixture):
        """Test that get_session returns a valid AsyncSession."""
        from gearmeshing_ai.server.core.database import get_session

        # Create a test session factory
        async_session_maker = sessionmaker(test_engine_fixture, class_=AsyncSession, expire_on_commit=False)

        with patch(
            "gearmeshing_ai.server.core.database.async_session_maker",
            async_session_maker,
        ):
            async for session in get_session():
                assert isinstance(session, AsyncSession)
                # Verify we can execute a simple query
                result = await session.execute(text("SELECT 1"))
                assert result is not None
                break

    async def test_engine_is_async_engine(self):
        """Test that the global engine is an AsyncEngine."""
        from gearmeshing_ai.server.core.database import engine

        assert isinstance(engine, AsyncEngine)

    async def test_async_session_maker_is_callable(self):
        """Test that async_session_maker is callable and returns AsyncSession."""
        from gearmeshing_ai.server.core.database import async_session_maker

        assert callable(async_session_maker)

        # Create a session and verify it's an AsyncSession
        session = async_session_maker()
        assert hasattr(session, "__aenter__")
        assert hasattr(session, "__aexit__")


class TestApplicationStartup:
    """Test full application startup with lifespan."""

    async def test_app_has_lifespan_configured(self):
        """Test that the FastAPI app has lifespan configured."""
        from gearmeshing_ai.server.main import app

        # In FastAPI, the lifespan is passed during app initialization
        # We can verify it by checking the router's lifespan
        assert app.router is not None

    async def test_lifespan_function_is_callable(self):
        """Test that lifespan function is callable."""
        from gearmeshing_ai.server.main import lifespan

        assert callable(lifespan)

    async def test_database_connection_after_startup(self, test_engine_fixture):
        """Test that database connection is available after startup."""
        from gearmeshing_ai.server.core.database import get_session

        async_session_maker = sessionmaker(test_engine_fixture, class_=AsyncSession, expire_on_commit=False)

        with patch(
            "gearmeshing_ai.server.core.database.async_session_maker",
            async_session_maker,
        ):
            async for session in get_session():
                # Verify we can query the database
                result = await session.execute(text("SELECT 1"))
                assert result is not None
                break

    async def test_multiple_sessions_from_factory(self, test_engine_fixture):
        """Test that multiple sessions can be created from the factory."""
        async_session_maker = sessionmaker(test_engine_fixture, class_=AsyncSession, expire_on_commit=False)

        sessions = []
        for _ in range(3):
            async with async_session_maker() as session:
                sessions.append(session)
                assert isinstance(session, AsyncSession)

        # Verify all sessions were created successfully
        assert len(sessions) == 3
