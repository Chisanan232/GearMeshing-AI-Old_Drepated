"""
Policy entity models.

This module contains the database entity for tenant-specific policy
configurations. Policies control agent behavior, capabilities, and
risk thresholds at the tenant level.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from sqlmodel import JSON, Field

from ..base import Base


class Policy(Base):
    """Entity for tenant policy configurations.

    Stores tenant-specific policy configurations that control
    agent behavior, risk thresholds, and capability access.

    Table: gm_policies
    """

    __tablename__ = "gm_policies"

    # Primary identifiers
    id: str = Field(primary_key=True, max_length=64)
    tenant_id: str = Field(max_length=128, unique=True, index=True)

    # Policy configuration
    config: Dict[str, Any] = Field(sa_type=JSON)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def __repr__(self) -> str:
        return f"Policy(id={self.id}, tenant_id={self.tenant_id})"
