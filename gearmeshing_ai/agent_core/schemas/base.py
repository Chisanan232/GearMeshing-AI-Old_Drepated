"""Pydantic base schema utilities for agent core models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    """
    Base Pydantic model for all domain schemas.

    Configures common Pydantic behaviors:
    - ``populate_by_name=True``: Allow initialization by alias or field name.
    - ``extra="forbid"``: Prevent unknown fields from slipping into the model, ensuring strict validation.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )
