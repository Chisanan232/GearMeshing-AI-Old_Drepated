"""
Global database session and engine management.

This module manages the global AsyncEngine and async_sessionmaker instances
that are used throughout the application for database access.
"""

from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gearmeshing_ai.server.core.config import settings

from .utils import create_engine, create_sessionmaker

# Create global engine and session factory
engine = create_engine(settings.database.url)
async_session_maker = create_sessionmaker(engine)


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
