"""
Checkpoint entity models.

This module contains the database entity for LangGraph state checkpoints.
Checkpoints enable pause/resume functionality by storing serialized
agent state at key points during execution.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from sqlmodel import Field, JSON, SQLModel

from ..base import Base


class Checkpoint(Base):
    """Entity for LangGraph state checkpoints.
    
    Stores serialized state required for pause/resume functionality
    in agent execution workflows.
    
    Table: gm_checkpoints
    """
    
    __tablename__ = "gm_checkpoints"
    
    # Primary identifiers
    id: str = Field(primary_key=True, max_length=64)
    run_id: str = Field(max_length=64, index=True)
    
    # Checkpoint details
    node: str = Field(max_length=128, index=True)
    state: Dict[str, Any] = Field(sa_type=JSON)
    
    # Timestamp
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    
    def __repr__(self) -> str:
        return f"Checkpoint(id={self.id}, run_id={self.run_id}, node={self.node})"
