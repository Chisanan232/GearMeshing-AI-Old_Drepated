"""Pydantic base schema utilities for agent core models.

This module re-exports BaseSchema from the centralized core.models package
for backward compatibility. New code should import directly from core.models.
"""

from __future__ import annotations

from gearmeshing_ai.core.models.base import BaseSchema

__all__ = ["BaseSchema"]
