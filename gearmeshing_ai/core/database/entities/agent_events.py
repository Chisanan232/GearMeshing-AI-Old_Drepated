"""
Agent event entity models.

This module contains the database entity for agent event streaming.
Events form an append-only timeline for each agent run, providing
auditability and debugging capabilities.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

from sqlmodel import Field, SQLModel

from ..base import Base


class AgentEventBase(Base):
    """Base fields for agent event entity."""
    
    # Event metadata
    type: str = Field(description="Event type identifier")
    correlation_id: Optional[str] = Field(default=None, description="Correlation ID for event grouping")
    payload: str = Field(default="{}", description="JSON event payload data")


class AgentEvent(AgentEventBase, table=True):
    """Entity for agent event streaming.
    
    Stores append-only events for agent runs with structured payloads.
    Each event represents a significant occurrence in the agent lifecycle.
    
    Table: gm_agent_events
    """
    
    __tablename__ = "gm_agent_events"
    
    # Primary identifiers
    id: str = Field(primary_key=True, max_length=64)
    run_id: str = Field(max_length=64, index=True, description="Associated agent run ID")
    
    # Timestamp
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    
    def get_payload_dict(self) -> Dict[str, Any]:
        """Get payload as a dictionary."""
        try:
            return json.loads(self.payload) if self.payload else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def set_payload_dict(self, payload_dict: Dict[str, Any]) -> None:
        """Set payload from a dictionary."""
        self.payload = json.dumps(payload_dict)
    
    def __repr__(self) -> str:
        return f"AgentEvent(id={self.id}, run_id={self.run_id}, type={self.type})"


# For compatibility with existing code
EventRow = AgentEvent
