"""
Centralized database layer for GearMeshing AI.

This package provides a unified location for all database entities and repositories,
organized by business domain and table relationships.

Structure:
- entities/: Database entity models organized by table/business logic
- repositories/: Data access layer organized by table/business logic
- schemas/: API schema models for request/response serialization
- session.py: Global engine and session factory management
- utils.py: Database utility functions (engine, session, repo bundle)
"""

from .base import Base
from .session import (
    async_session_maker,
    engine,
    get_session,
    init_db,
)
from .utils import (
    create_all,
    create_engine,
    create_sessionmaker,
)

__all__ = [
    "Base",
    "async_session_maker",
    "create_all",
    "create_engine",
    "create_sessionmaker",
    "engine",
    "get_session",
    "init_db",
]
