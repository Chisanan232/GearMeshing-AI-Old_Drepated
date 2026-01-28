"""
Agent configuration entity models.

This module contains the database entity for agent configuration management.
Agent configurations define roles, model settings, capabilities, and tool
access for different agent types.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlmodel import Field, SQLModel

from ..base import Base

if TYPE_CHECKING:
    from gearmeshing_ai.core.models.config import ModelConfig, RoleConfig


class AgentConfigBase(Base):
    """Base fields for agent configuration."""
    
    # Basic role information
    role_name: str = Field(description="Role identifier (e.g., 'dev', 'planner')")
    display_name: str = Field(description="Human-readable role name")
    description: str = Field(description="Role description")
    system_prompt_key: str = Field(description="Key for system prompt lookup")
    
    # Model configuration
    model_provider: str = Field(description="LLM provider (openai, anthropic, google)")
    model_name: str = Field(description="Model identifier (e.g., 'gpt-4o')")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(default=4096, ge=1, description="Max tokens per response")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    
    # Capabilities and tools (stored as JSON strings for SQLModel compatibility)
    capabilities: str = Field(default="[]", description="JSON array of capability names")
    tools: str = Field(default="[]", description="JSON array of tool names")
    autonomy_profiles: str = Field(default="[]", description="JSON array of autonomy profile names")
    
    # Additional metadata
    done_when: Optional[str] = Field(default=None, description="Completion criteria")
    is_active: bool = Field(default=True, description="Whether this role is active")
    tenant_id: Optional[str] = Field(default=None, description="Tenant-specific override")


class AgentConfig(AgentConfigBase, table=True):
    """Persistent agent configuration in database.
    
    Stores complete agent role configurations including model settings,
    capabilities, and tool access permissions.
    
    Table: agent_configs
    """
    
    __tablename__ = "agent_configs"
    __table_args__ = ({"extend_existing": True},)
    
    # Primary key
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"onupdate": datetime.utcnow})
    
    def get_capabilities_list(self) -> List[str]:
        """Get capabilities as a list."""
        try:
            return json.loads(self.capabilities) if self.capabilities else []
        except (json.JSONDecodeError, TypeError):
            return []
    
    def set_capabilities_list(self, capabilities: List[str]) -> None:
        """Set capabilities from a list."""
        self.capabilities = json.dumps(capabilities)
    
    def get_tools_list(self) -> List[str]:
        """Get tools as a list."""
        try:
            return json.loads(self.tools) if self.tools else []
        except (json.JSONDecodeError, TypeError):
            return []
    
    def set_tools_list(self, tools: List[str]) -> None:
        """Set tools from a list."""
        self.tools = json.dumps(tools)
    
    def get_autonomy_profiles_list(self) -> List[str]:
        """Get autonomy profiles as a list."""
        try:
            return json.loads(self.autonomy_profiles) if self.autonomy_profiles else []
        except (json.JSONDecodeError, TypeError):
            return []
    
    def set_autonomy_profiles_list(self, autonomy_profiles: List[str]) -> None:
        """Set autonomy profiles from a list."""
        self.autonomy_profiles = json.dumps(autonomy_profiles)
    
    def __repr__(self) -> str:
        return f"AgentConfig(role={self.role_name}, provider={self.model_provider}, model={self.model_name})"
