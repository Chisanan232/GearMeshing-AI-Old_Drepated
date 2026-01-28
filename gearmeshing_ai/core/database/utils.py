"""
Database utility functions for engine and session management.

This module provides the core utility functions for creating database engines,
session factories, and repository bundles. Built with async SQLAlchemy to match
the performance requirements of the agent runtime system.

Functions:
- create_engine: Creates async SQLAlchemy engine with URL normalization
- create_sessionmaker: Creates async session factory with safe defaults
- create_all: Creates all tables from ORM metadata (for tests/dev)
- build_sql_repos: Builds complete repository bundle for dependency injection
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .base import Base


def create_engine(db_url: str) -> AsyncEngine:
    """Create an async SQLAlchemy engine.

    The helper normalizes Postgres URLs to ensure the async driver is used.
    For example, it rewrites ``postgresql://`` and other variants to
    ``postgresql+asyncpg://``.
    
    Args:
        db_url: Database connection URL
        
    Returns:
        Configured AsyncEngine instance
    """
    url = re.sub(r"^postgres(?:ql)?(?:\+[a-z0-9_]+)?://", "postgresql+asyncpg://", db_url, count=1)
    return create_async_engine(url, pool_pre_ping=True)


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create an ``async_sessionmaker`` with safe defaults for this project.
    
    Args:
        engine: Async SQLAlchemy engine
        
    Returns:
        Configured async session factory
    """
    return async_sessionmaker(engine, expire_on_commit=False)


async def create_all(engine: AsyncEngine) -> None:
    """Create all tables for the current ORM metadata.

    This is mainly intended for tests and local development.
    Production should use Alembic migrations instead.
    
    Args:
        engine: Async SQLAlchemy engine
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def _utc_now_naive() -> datetime:
    """Get current UTC datetime as naive datetime.
    
    Returns:
        Current UTC datetime without timezone info
    """
    return datetime.now(timezone.utc)
