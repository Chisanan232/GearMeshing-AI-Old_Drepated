"""
Database Connection and Session Management.

This module sets up the asynchronous SQLAlchemy engine and session factory.
It provides utilities for dependency injection of database sessions.
"""
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

from gearmeshing_ai.server.core.config import settings

# Create Async Engine
# echo=True can be enabled for debugging SQL queries
"""
engine:
    The global SQLAlchemy AsyncEngine instance.
    Configured with the connection URL from settings and optimized for async usage.
"""
engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)

# Create Session Factory
"""
async_session_maker:
    A global factory for creating new AsyncSession instances.
    Bound to the `engine` and configured to NOT expire on commit (typical for async).
"""
async_session_maker = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

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

    Creates all tables defined in SQLModel metadata.
    NOTE: In production, Alembic migrations should be used instead of this function.
    """
    # In production, we use Alembic. 
    # For dev/testing, this can create tables if they don't exist.
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

