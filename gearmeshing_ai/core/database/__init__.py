"""
Centralized database layer for GearMeshing AI.

This package provides a unified location for all database entities and repositories,
organized by business domain and table relationships.

Structure:
- entities/: Database entity models organized by table/business logic
- repositories/: Data access layer organized by table/business logic
"""

from .base import Base

__all__ = ["Base"]