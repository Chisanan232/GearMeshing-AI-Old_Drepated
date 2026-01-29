"""
Agent configuration I/O models for API requests and responses.

This module contains Pydantic-based I/O schemas for agent configuration
API endpoints. These models define the contract between the API and clients
for creating, reading, and updating agent configurations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class AgentConfigRead(BaseModel):
    """Schema for reading agent configuration from API."""

    id: int
    role_name: str = Field(description="Role identifier (e.g., 'dev', 'planner')")
    display_name: str = Field(description="Human-readable role name")
    description: str = Field(description="Role description")
    system_prompt_key: str = Field(description="Key for system prompt lookup")
    model_provider: str = Field(description="LLM provider (openai, anthropic, google)")
    model_name: str = Field(description="Model identifier (e.g., 'gpt-4o')")
    temperature: float = Field(description="Model temperature")
    max_tokens: int = Field(description="Max tokens per response")
    top_p: float = Field(description="Nucleus sampling parameter")
    capabilities: str = Field(description="JSON array of capability names")
    tools: str = Field(description="JSON array of tool names")
    autonomy_profiles: str = Field(description="JSON array of autonomy profile names")
    done_when: Optional[str] = Field(default=None, description="Completion criteria")
    is_active: bool = Field(description="Whether this role is active")
    tenant_id: Optional[str] = Field(default=None, description="Tenant-specific override")
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AgentConfigCreate(BaseModel):
    """Schema for creating agent configuration via API."""

    role_name: str = Field(description="Role identifier (e.g., 'dev', 'planner')")
    display_name: str = Field(description="Human-readable role name")
    description: str = Field(description="Role description")
    system_prompt_key: str = Field(description="Key for system prompt lookup")
    model_provider: str = Field(description="LLM provider (openai, anthropic, google)")
    model_name: str = Field(description="Model identifier (e.g., 'gpt-4o')")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    capabilities: str = Field(default="[]", description="JSON array of capability names")
    tools: str = Field(default="[]", description="JSON array of tool names")
    autonomy_profiles: str = Field(default="[]", description="JSON array of autonomy profile names")
    done_when: Optional[str] = Field(default=None, description="Completion criteria")
    is_active: bool = Field(default=True, description="Whether this role is active")
    tenant_id: Optional[str] = Field(default=None, description="Tenant-specific override")


class AgentConfigUpdate(BaseModel):
    """Schema for updating agent configuration via API."""

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
