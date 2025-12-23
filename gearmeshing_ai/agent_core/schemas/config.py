"""Domain models for agent configuration."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model configuration with provider and parameters."""

    provider: str = Field(..., description="Model provider (openai, anthropic, google)")
    model: str = Field(..., description="Model name (e.g., gpt-4o, claude-3-5-sonnet)")
    temperature: float = Field(default=0.7, description="Temperature (0.0-2.0)")
    max_tokens: int = Field(default=4096, description="Maximum output tokens")
    top_p: float = Field(default=0.9, description="Top-p sampling (0.0-1.0)")


class RoleConfig(BaseModel):
    """Complete role configuration."""

    role_name: str = Field(..., description="Role identifier")
    display_name: str = Field(..., description="Human-readable role name")
    description: Optional[str] = Field(default=None, description="Role description")
    system_prompt_key: Optional[str] = Field(default=None, description="System prompt key")
    model: ModelConfig = Field(..., description="Model settings for this role")
    capabilities: list[str] = Field(default_factory=list, description="Available capabilities")
    tools: list[str] = Field(default_factory=list, description="Available tools")
    autonomy_profiles: list[str] = Field(default_factory=list, description="Supported autonomy profiles")
    done_when: Optional[str] = Field(default=None, description="Completion criteria")
