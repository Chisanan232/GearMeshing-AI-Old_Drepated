"""Pydantic base schema utilities for MCP info provider models.

Provides a common :class:`BaseSchema` that enforces aliasing and extra-field
policy for all DTOs and domain models under
``gearmeshing_ai.info_provider.mcp.schemas``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


def _to_camel(s: str) -> str:
    """Convert snake_case to camelCase for JSON aliasing."""
    parts = s.split("_")
    return parts[0] + "".join(p.capitalize() or "_" for p in parts[1:])


class BaseSchema(BaseModel):
    """Shared base for all Pydantic models in the MCP info provider layer.

    - Sets strict handling for extra fields
    - Enables ``populate_by_name`` for using either snake_case or camelCase
    - Uses a snake->camel alias generator for JSON interop
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        alias_generator=_to_camel,  # snake_case -> camelCase aliases
    )
