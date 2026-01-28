"""Domain models for agent configuration.

This module re-exports configuration models from the centralized core.models package
for backward compatibility. New code should import directly from core.models.
"""

from __future__ import annotations

from gearmeshing_ai.core.models.config import ModelConfig, RoleConfig

__all__ = ["ModelConfig", "RoleConfig"]
