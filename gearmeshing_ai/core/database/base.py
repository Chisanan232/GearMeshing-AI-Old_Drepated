"""
Base database models and utilities.

This module provides the foundational database components used across
all entities in the centralized database layer using SQLModel.
"""

from __future__ import annotations

from sqlmodel import SQLModel


class Base(SQLModel):
    """Base class for all SQLModel entities."""
    
    class Config:
        """Pydantic configuration for all models."""
        arbitrary_types_allowed = True
