"""
Database models for AI agent configuration persistence.

This module defines SQLModel-based models for storing agent configurations,
roles, and related settings in the database instead of YAML files.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlmodel import Column, DateTime, Field, SQLModel, JSON

if TYPE_CHECKING:
    from gearmeshing_ai.agent_core.schemas.config import ModelConfig, RoleConfig


class AgentConfigBase(SQLModel):
    """Base fields for agent configuration."""

    role_name: str = Field(index=True, description="Role identifier (e.g., 'dev', 'planner')")
    display_name: str = Field(description="Human-readable role name")
    description: str = Field(description="Role description")
    system_prompt_key: str = Field(description="Key for system prompt lookup")
    
    # Model configuration
    model_provider: str = Field(description="LLM provider (openai, anthropic, google)")
    model_name: str = Field(description="Model identifier (e.g., 'gpt-4o')")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    
    # Capabilities and tools
    capabilities: str = Field(default="[]", description="JSON array of capability names")
    tools: str = Field(default="[]", description="JSON array of tool names")
    autonomy_profiles: str = Field(default="[]", description="JSON array of autonomy profile names")
    
    # Additional metadata
    done_when: Optional[str] = Field(default=None, description="Completion criteria")
    is_active: bool = Field(default=True, description="Whether this role is active")
    tenant_id: Optional[str] = Field(default=None, index=True, description="Tenant-specific override")


class AgentConfig(AgentConfigBase, table=True):
    """Persistent agent configuration in database."""

    __tablename__ = "agent_configs"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime, nullable=False),
        description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime, nullable=False),
        description="Last update timestamp"
    )

    def to_model_config(self) -> ModelConfig:
        """Convert AgentConfig to ModelConfig domain model.

        Returns:
            ModelConfig domain model with provider and parameters.
        """
        from gearmeshing_ai.agent_core.schemas.config import ModelConfig

        return ModelConfig(
            provider=self.model_provider,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )

    def to_role_config(self) -> RoleConfig:
        """Convert AgentConfig to RoleConfig domain model.

        Returns:
            RoleConfig domain model with complete role settings.
        """
        from gearmeshing_ai.agent_core.schemas.config import RoleConfig

        capabilities: list[str] = json.loads(self.capabilities) if self.capabilities else []
        tools: list[str] = json.loads(self.tools) if self.tools else []
        autonomy_profiles: list[str] = json.loads(self.autonomy_profiles) if self.autonomy_profiles else []

        model_config: ModelConfig = self.to_model_config()

        return RoleConfig(
            role_name=self.role_name,
            display_name=self.display_name,
            description=self.description,
            system_prompt_key=self.system_prompt_key,
            model=model_config,
            capabilities=capabilities,
            tools=tools,
            autonomy_profiles=autonomy_profiles,
            done_when=self.done_when,
        )


class AgentConfigRead(AgentConfigBase):
    """Schema for reading agent configuration."""

    id: int
    created_at: datetime
    updated_at: datetime


class AgentConfigCreate(AgentConfigBase):
    """Schema for creating agent configuration."""

    pass


class AgentConfigUpdate(SQLModel):
    """Schema for updating agent configuration."""

    display_name: Optional[str] = None
    description: Optional[str] = None
    system_prompt_key: Optional[str] = None
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    capabilities: Optional[str] = None
    tools: Optional[str] = None
    autonomy_profiles: Optional[str] = None
    done_when: Optional[str] = None
    is_active: Optional[bool] = None
