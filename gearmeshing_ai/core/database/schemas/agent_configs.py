"""
Schema models for agent configuration API requests and responses.

These schemas are used for API serialization/deserialization and are separate
from the entity models to allow independent evolution of API contracts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class AgentConfigBase(BaseModel):
    """Base fields for agent configuration schema."""

    role_name: str = Field(description="Role identifier (e.g., 'dev', 'planner')")
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
    tenant_id: Optional[str] = Field(default=None, description="Tenant-specific override")


class AgentConfigRead(AgentConfigBase):
    """Schema for reading agent configuration."""

    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AgentConfigCreate(AgentConfigBase):
    """Schema for creating agent configuration."""

    pass


class AgentConfigUpdate(BaseModel):
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
