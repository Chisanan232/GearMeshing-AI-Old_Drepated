"""
Centralized database layer for GearMeshing AI.

This package provides a unified location for all database entities and repositories,
organized by business domain and table relationships.

Structure:
- entities/: Database entity models organized by table/business logic
- repositories/: Data access layer organized by table/business logic
- async_base.py: Async base repository interfaces and utilities
- async_repositories_complete.py: Complete async repository implementations
- utils.py: Database utility functions (engine, session, repo bundle)
"""

from .base import Base
from .utils import (
    create_all,
    create_engine,
    create_sessionmaker,
)

__all__ = [
    "Base",
    "create_all",
    "create_engine",
    "create_sessionmaker",
]