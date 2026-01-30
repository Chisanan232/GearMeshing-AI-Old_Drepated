"""
Approval entity models.

This module contains the database entity for approval workflow management.
Approvals are required for high-risk operations and provide audit trails
for human-in-the-loop decision making.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Field, Text

from ..base import Base


class Approval(Base):
    """Entity for approval workflow management.

    Stores approval requests and their resolutions for high-risk
    agent operations requiring human oversight.

    Table: gm_approvals
    """

    __tablename__ = "gm_approvals"

    # Primary identifiers
    id: str = Field(primary_key=True, max_length=64)
    run_id: str = Field(max_length=64, index=True)

    # Approval request details
    risk: str = Field(max_length=16, index=True)
    capability: str = Field(max_length=64, index=True)
    reason: str = Field(sa_type=Text)

    # Request timing
    requested_at: datetime = Field(index=True)
    expires_at: Optional[datetime] = Field(default=None)

    # Resolution details
    decision: Optional[str] = Field(default=None, max_length=16, index=True)
    decided_at: Optional[datetime] = Field(default=None)
    decided_by: Optional[str] = Field(default=None, max_length=128)

    def __repr__(self) -> str:
        return f"Approval(id={self.id}, risk={self.risk}, decision={self.decision})"
