"""
Database Connection and Session Management.

This module sets up the asynchronous SQLAlchemy engine and session factory.
It provides utilities for dependency injection of database sessions.
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

# Import Base here to ensure it's available for table creation
from gearmeshing_ai.agent_core.repos.models import Base  # noqa: E402, F401
from gearmeshing_ai.agent_core.repos.sql import create_engine, create_sessionmaker
from gearmeshing_ai.server.core.config import settings

# Create global engine and session factory using agent_core utilities
# Create Async Engine
# echo=True can be enabled for debugging SQL queries
"""
engine:
    The global SQLAlchemy AsyncEngine instance.
    Configured with the connection URL from settings and optimized for async usage.
"""
engine = create_engine(settings.database_url)

# Create Session Factory
"""
async_session_maker:
    A global factory for creating new AsyncSession instances.
    Bound to the `engine` and configured to NOT expire on commit (typical for async).
"""
async_session_maker = create_sessionmaker(engine)

# Create Connection Pool for LangGraph Checkpointer
"""
checkpointer_pool:
    A global connection pool for the LangGraph AsyncPostgresSaver.
    We use psycopg_pool directly as required by langgraph-checkpoint-postgres.
"""
from psycopg_pool import AsyncConnectionPool
# Convert sqlalchemy URL to psycopg URL (remove +asyncpg if present)
# sqlalchemy: postgresql+asyncpg://user:pass@host/db
# psycopg: postgresql://user:pass@host/db
psycopg_url = settings.database_url.replace("+asyncpg", "")
checkpointer_pool = AsyncConnectionPool(conninfo=psycopg_url, open=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency generator for database sessions.

    Yields:
        AsyncSession: An asynchronous SQLAlchemy session.
    """
    async with async_session_maker() as session:
        yield session


async def init_db():
    """
    Initialize the database.

    Creates all tables defined in the authoritative agent_core ORM metadata.
    NOTE: In production, Alembic migrations should be used instead of this function.
    
    This function is now a no-op since Alembic migrations handle table creation.
    The db-migrate service runs migrations before the application starts.
    """
    # In production, Alembic migrations (run by db-migrate service) handle all DDL.
    # This function is kept for backward compatibility but does nothing.
    # Attempting to call create_all() here would fail with "CREATE INDEX CONCURRENTLY
    # cannot run inside a transaction block" since Alembic already created indexes.
    pass
