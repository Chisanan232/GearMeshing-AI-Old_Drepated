"""AI agent configuration source with separated provider integrations.

This module defines the AgentConfigSource class that orchestrates model and prompt
providers to create complete AI agent configurations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from gearmeshing_ai.agent_core.abstraction.base import AIAgentConfig
from gearmeshing_ai.core.models.config import ModelConfig


class AgentConfigSource(BaseModel):
    """
    Orchestrated agent configuration using separated model and prompt providers.

    This class provides a high-level interface for creating AI agent configurations
    by combining model settings from the ModelProvider system with system prompts
    from the existing PromptProvider system.

    Attributes:
        model_config_key: Key for model configuration (e.g., 'gpt4_default', 'claude_sonnet').
        tenant_id: Optional tenant identifier for multi-tenant model configurations.
        prompt_key: Key for system prompt (e.g., 'dev/system', 'pm/system').
        locale: Language locale for prompt resolution.
        prompt_tenant_id: Optional tenant identifier for prompt-specific multi-tenancy.
        overrides: Optional runtime overrides for agent configuration.
    """

    model_config_key: str = Field(..., description="Key for model configuration")
    tenant_id: Optional[str] = Field(default=None, description="Tenant identifier for model config")
    prompt_key: str = Field(..., description="Key for system prompt")
    locale: str = Field(default="en", description="Language locale for prompts")
    prompt_tenant_id: Optional[str] = Field(default=None, description="Tenant identifier for prompts")
    overrides: Optional[Dict[str, Any]] = Field(default=None, description="Runtime configuration overrides")

    def get_model_config(self) -> ModelConfig:
        """
        Get model configuration using the ModelProvider system.

        Returns:
            ModelConfig: The resolved model configuration.

        Raises:
            KeyError: If the model configuration key is not found.
        """
        from gearmeshing_ai.info_provider.model import load_model_provider

        model_provider = load_model_provider()
        return model_provider.get(self.model_config_key, self.tenant_id)

    def get_system_prompt(self) -> str:
        """
        Get system prompt using the existing PromptProvider system.

        Returns:
            str: The resolved system prompt text.

        Raises:
            KeyError: If the prompt key is not found.
        """
        from gearmeshing_ai.info_provider.prompt import load_prompt_provider

        prompt_provider = load_prompt_provider()
        return prompt_provider.get(name=self.prompt_key, locale=self.locale, tenant=self.prompt_tenant_id)

    def to_agent_config(self, framework: str, name: Optional[str] = None) -> AIAgentConfig:
        """
        Convert this configuration source to a complete AIAgentConfig.

        Args:
            framework: The AI framework to use (e.g., 'pydantic_ai', 'langchain').
            name: Optional name for the agent. If not provided, auto-generated.

        Returns:
            AIAgentConfig: Complete agent configuration ready for agent creation.
        """
        # Get model configuration
        model_config = self.get_model_config()

        # Get system prompt
        system_prompt = self.get_system_prompt()

        # Generate name if not provided
        if name is None:
            name = f"agent_{self.model_config_key}_{self.prompt_key}"
            if self.tenant_id:
                name += f"_{self.tenant_id}"

        # Filter out model parameter overrides from metadata
        model_params = {"temperature", "max_tokens", "top_p"}
        metadata = {k: v for k, v in (self.overrides or {}).items() if k not in model_params}

        # Build agent configuration
        agent_config = AIAgentConfig(
            name=name,
            framework=framework,
            model=model_config.model,
            system_prompt=system_prompt,
            temperature=(
                self.overrides.get("temperature")
                if self.overrides and "temperature" in self.overrides
                else model_config.temperature
            ),
            max_tokens=(
                self.overrides.get("max_tokens")
                if self.overrides and "max_tokens" in self.overrides
                else model_config.max_tokens
            ),
            top_p=self.overrides.get("top_p") if self.overrides and "top_p" in self.overrides else model_config.top_p,
            tools=[],  # Tools can be added separately if needed
            metadata=metadata,
        )

        return agent_config
