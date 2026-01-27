"""
Usage ledger entity models.

This module contains the database entity for token and cost accounting.
The usage ledger provides append-only accounting of resource consumption
per agent run for billing and monitoring purposes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel

from ..base import Base


class UsageLedger(Base):
    """Entity for token and cost accounting.
    
    Append-only ledger recording token usage and costs for each
    agent run, enabling billing and usage analytics.
    
    Table: gm_usage_ledger
    """
    
    __tablename__ = "gm_usage_ledger"
    
    # Primary identifiers
    id: str = Field(primary_key=True, max_length=64)
    run_id: str = Field(max_length=64, index=True)
    tenant_id: Optional[str] = Field(default=None, max_length=128, index=True)
    
    # Model information
    provider: str = Field(max_length=64, index=True)
    model: str = Field(max_length=128, index=True)
    
    # Token usage
    prompt_tokens: int = Field()
    completion_tokens: int = Field()
    total_tokens: int = Field()
    
    # Cost information
    cost_usd: Optional[float] = Field(default=None)
    
    # Timestamp
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    
    def __repr__(self) -> str:
        return f"UsageLedger(id={self.id}, run_id={self.run_id}, total_tokens={self.total_tokens})"
