"""
Database Connection and Session Management.

This module sets up the asynchronous SQLAlchemy engine and session factory.
It provides utilities for dependency injection of database sessions.
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession

from gearmeshing_ai.agent_core.repos.sql import create_engine, create_sessionmaker
from gearmeshing_ai.server.core.config import settings
# Import Base here to ensure it's available for table creation
from gearmeshing_ai.agent_core.repos.models import Base  # noqa: E402, F401

# Create global engine and session factory using agent_core utilities
# Create Async Engine
# echo=True can be enabled for debugging SQL queries
"""
engine:
    The global SQLAlchemy AsyncEngine instance.
    Configured with the connection URL from settings and optimized for async usage.
"""
engine = create_engine(settings.DATABASE_URL)

# Create Session Factory
"""
async_session_maker:
    A global factory for creating new AsyncSession instances.
    Bound to the `engine` and configured to NOT expire on commit (typical for async).
"""
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
    """
    # In production, we use Alembic.
    # For dev/testing, this can create tables if they don't exist.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
