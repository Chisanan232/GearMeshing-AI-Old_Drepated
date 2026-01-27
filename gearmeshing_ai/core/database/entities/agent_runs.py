"""
Agent run entity models.

This module contains the database entity for agent run lifecycle and metadata.
Each agent run represents a complete execution session with its status,
objective, and runtime configuration.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel

from ..base import Base


class AgentRunBase(Base):
    """Base fields for agent run entity."""
    
    # Runtime configuration
    role: str = Field(description="Agent role for this run")
    autonomy_profile: str = Field(description="Autonomy profile for decision making")
    
    # Objective and completion criteria
    objective: str = Field(description="User objective being solved")
    done_when: Optional[str] = Field(default=None, description="Completion criteria")
    
    # Version tracking
    prompt_provider_version: Optional[str] = Field(default=None, description="Prompt provider version")
    
    # Status
    status: str = Field(description="Current run status")


class AgentRun(AgentRunBase, table=True):
    """Entity for agent run lifecycle and metadata.
    
    Stores the complete lifecycle of an agent run including status,
    objective, role configuration, and timestamps.
    
    Table: gm_agent_runs
    """
    
    __tablename__ = "gm_agent_runs"
    
    # Primary identifiers
    id: str = Field(primary_key=True, max_length=64)
    tenant_id: Optional[str] = Field(default=None, max_length=128, index=True)
    workspace_id: Optional[str] = Field(default=None, max_length=128, index=True)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"onupdate": datetime.utcnow})
    
    def __repr__(self) -> str:
        return f"AgentRun(id={self.id}, role={self.role}, status={self.status})"


# For compatibility with existing code that expects the class name
RunRow = AgentRun
