"""
Tool invocation entity models.

This module contains the database entity for tool execution records.
Tool invocations provide an auditable record of side-effecting operations
performed by agents during execution.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict

from sqlmodel import Field, SQLModel

from ..base import Base


class ToolInvocation(Base, table=True):
    """Entity for tool execution records.
    
    Records side-effecting tool invocations made by agent actions,
    including arguments, results, and risk assessment.
    
    Table: gm_tool_invocations
    """
    
    __tablename__ = "gm_tool_invocations"
    
    # Primary identifiers
    id: str = Field(primary_key=True, max_length=64)
    run_id: str = Field(index=True, max_length=64)
    
    # Tool execution details
    server_id: str = Field(index=True, max_length=128)
    tool_name: str = Field(index=True, max_length=256)
    
    # Execution data (stored as JSON strings for SQLModel compatibility)
    args: str = Field(default="{}")
    ok: bool = Field(index=True)
    result: str = Field(default="{}")
    
    # Risk assessment
    risk: str = Field(index=True, max_length=16)
    
    # Timestamp
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    
    # Helper methods for JSON serialization
    def get_args_dict(self) -> Dict[str, Any]:
        """Get args as dictionary."""
        try:
            return json.loads(self.args) if self.args else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def set_args_dict(self, args_dict: Dict[str, Any]) -> None:
        """Set args from dictionary."""
        self.args = json.dumps(args_dict) if args_dict else "{}"
    
    def get_result_dict(self) -> Dict[str, Any]:
        """Get result as dictionary."""
        try:
            return json.loads(self.result) if self.result else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def set_result_dict(self, result_dict: Dict[str, Any]) -> None:
        """Set result from dictionary."""
        self.result = json.dumps(result_dict) if result_dict else "{}"
    
    def __repr__(self) -> str:
        return f"ToolInvocation(id={self.id}, tool={self.tool_name}, ok={self.ok})"
