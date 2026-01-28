"""Unit tests for AgentConfigSource.

This module provides comprehensive tests for the AgentConfigSource class,
including provider integration and configuration resolution.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gearmeshing_ai.agent_core.abstraction.base import AIAgentConfig
from gearmeshing_ai.agent_core.abstraction.config_source import AgentConfigSource
from gearmeshing_ai.core.models.config import ModelConfig


class TestAgentConfigSource:
    """Test cases for AgentConfigSource."""

    def test_init_minimal(self):
        """Test initialization with minimal parameters."""
        source = AgentConfigSource(model_config_key="gpt4_default", prompt_key="dev/system")

        assert source.model_config_key == "gpt4_default"
        assert source.prompt_key == "dev/system"
        assert source.locale == "en"
        assert source.tenant_id is None
        assert source.prompt_tenant_id is None
        assert source.overrides is None

    def test_init_full(self):
        """Test initialization with all parameters."""
        overrides = {"temperature": 0.1, "max_tokens": 8000}
        source = AgentConfigSource(
            model_config_key="claude_sonnet",
            tenant_id="acme_corp",
            prompt_key="pm/system",
            locale="fr",
            prompt_tenant_id="acme_corp",
            overrides=overrides,
        )

        assert source.model_config_key == "claude_sonnet"
        assert source.tenant_id == "acme_corp"
        assert source.prompt_key == "pm/system"
        assert source.locale == "fr"
        assert source.prompt_tenant_id == "acme_corp"
        assert source.overrides == overrides

    @patch("gearmeshing_ai.info_provider.model.load_model_provider")
    def test_get_model_config(self, mock_load_provider):
        """Test getting model configuration."""
        mock_provider = MagicMock()
        expected_config = ModelConfig(provider="openai", model="gpt-4o", temperature=0.7, max_tokens=4096, top_p=0.9)
        mock_provider.get.return_value = expected_config
        mock_load_provider.return_value = mock_provider

        source = AgentConfigSource(model_config_key="gpt4_default", prompt_key="dev/system")

        result = source.get_model_config()

        assert result == expected_config
        mock_provider.get.assert_called_once_with("gpt4_default", None)
        mock_load_provider.assert_called_once()

    @patch("gearmeshing_ai.info_provider.model.load_model_provider")
    def test_get_model_config_with_tenant(self, mock_load_provider):
        """Test getting model configuration with tenant."""
        mock_provider = MagicMock()
        expected_config = ModelConfig(
            provider="anthropic", model="claude-3-5-sonnet", temperature=0.5, max_tokens=8192, top_p=0.9
        )
        mock_provider.get.return_value = expected_config
        mock_load_provider.return_value = mock_provider

        source = AgentConfigSource(model_config_key="claude_sonnet", tenant_id="acme_corp", prompt_key="dev/system")

        result = source.get_model_config()

        assert result == expected_config
        mock_provider.get.assert_called_once_with("claude_sonnet", "acme_corp")

    @patch("gearmeshing_ai.info_provider.model.load_model_provider")
    def test_get_model_config_not_found(self, mock_load_provider):
        """Test getting model configuration when not found."""
        mock_provider = MagicMock()
        mock_provider.get.side_effect = KeyError("Model config not found")
        mock_load_provider.return_value = mock_provider

        source = AgentConfigSource(model_config_key="nonexistent", prompt_key="dev/system")

        with pytest.raises(KeyError, match="Model config not found"):
            source.get_model_config()

    @patch("gearmeshing_ai.info_provider.prompt.load_prompt_provider")
    def test_get_system_prompt(self, mock_load_provider):
        """Test getting system prompt."""
        mock_provider = MagicMock()
        expected_prompt = "You are a senior software engineer..."
        mock_provider.get.return_value = expected_prompt
        mock_load_provider.return_value = mock_provider

        source = AgentConfigSource(model_config_key="gpt4_default", prompt_key="dev/system")

        result = source.get_system_prompt()

        assert result == expected_prompt
        mock_provider.get.assert_called_once_with(name="dev/system", locale="en", tenant=None)
        mock_load_provider.assert_called_once()

    @patch("gearmeshing_ai.info_provider.prompt.load_prompt_provider")
    def test_get_system_prompt_with_locale_and_tenant(self, mock_load_provider):
        """Test getting system prompt with locale and tenant."""
        mock_provider = MagicMock()
        expected_prompt = "Vous êtes un ingénieur logiciel senior..."
        mock_provider.get.return_value = expected_prompt
        mock_load_provider.return_value = mock_provider

        source = AgentConfigSource(
            model_config_key="gpt4_default", prompt_key="pm/system", locale="fr", prompt_tenant_id="acme_corp"
        )

        result = source.get_system_prompt()

        assert result == expected_prompt
        mock_provider.get.assert_called_once_with(name="pm/system", locale="fr", tenant="acme_corp")

    @patch("gearmeshing_ai.info_provider.prompt.load_prompt_provider")
    def test_get_system_prompt_not_found(self, mock_load_provider):
        """Test getting system prompt when not found."""
        mock_provider = MagicMock()
        mock_provider.get.side_effect = KeyError("Prompt not found")
        mock_load_provider.return_value = mock_provider

        source = AgentConfigSource(model_config_key="gpt4_default", prompt_key="nonexistent")

        with pytest.raises(KeyError, match="Prompt not found"):
            source.get_system_prompt()

    @patch("gearmeshing_ai.info_provider.prompt.load_prompt_provider")
    @patch("gearmeshing_ai.info_provider.model.load_model_provider")
    def test_to_agent_config(self, mock_load_model, mock_load_prompt):
        """Test converting to AIAgentConfig."""
        # Setup model provider mock
        mock_model_provider = MagicMock()
        model_config = ModelConfig(provider="openai", model="gpt-4o", temperature=0.7, max_tokens=4096, top_p=0.9)
        mock_model_provider.get.return_value = model_config
        mock_load_model.return_value = mock_model_provider

        # Setup prompt provider mock
        mock_prompt_provider = MagicMock()
        system_prompt = "You are a senior software engineer..."
        mock_prompt_provider.get.return_value = system_prompt
        mock_load_prompt.return_value = mock_prompt_provider

        source = AgentConfigSource(
            model_config_key="gpt4_default", prompt_key="dev/system", overrides={"custom_param": "value"}
        )

        agent_config = source.to_agent_config(framework="pydantic_ai")

        assert isinstance(agent_config, AIAgentConfig)
        assert agent_config.name == "agent_gpt4_default_dev/system"
        assert agent_config.framework == "pydantic_ai"
        assert agent_config.model == "gpt-4o"
        assert agent_config.system_prompt == system_prompt
        assert agent_config.temperature == 0.7
        assert agent_config.max_tokens == 4096
        assert agent_config.top_p == 0.9
        assert agent_config.tools == []
        assert agent_config.metadata == {"custom_param": "value"}

    @patch("gearmeshing_ai.info_provider.prompt.load_prompt_provider")
    @patch("gearmeshing_ai.info_provider.model.load_model_provider")
    def test_to_agent_config_with_tenant(self, mock_load_model, mock_load_prompt):
        """Test converting to AIAgentConfig with tenant."""
        # Setup mocks
        mock_model_provider = MagicMock()
        model_config = ModelConfig(
            provider="anthropic", model="claude-3-5-sonnet", temperature=0.5, max_tokens=8192, top_p=0.9
        )
        mock_model_provider.get.return_value = model_config
        mock_load_model.return_value = mock_model_provider

        mock_prompt_provider = MagicMock()
        system_prompt = "You are a product manager..."
        mock_prompt_provider.get.return_value = system_prompt
        mock_load_prompt.return_value = mock_prompt_provider

        source = AgentConfigSource(model_config_key="claude_sonnet", tenant_id="acme_corp", prompt_key="pm/system")

        agent_config = source.to_agent_config(framework="pydantic_ai")

        assert agent_config.name == "agent_claude_sonnet_pm/system_acme_corp"
        assert agent_config.model == "claude-3-5-sonnet"
        assert agent_config.temperature == 0.5

    @patch("gearmeshing_ai.info_provider.prompt.load_prompt_provider")
    @patch("gearmeshing_ai.info_provider.model.load_model_provider")
    def test_to_agent_config_custom_name(self, mock_load_model, mock_load_prompt):
        """Test converting to AIAgentConfig with custom name."""
        # Setup mocks
        mock_model_provider = MagicMock()
        model_config = ModelConfig(provider="google", model="gemini-pro", temperature=0.7, max_tokens=2048, top_p=0.9)
        mock_model_provider.get.return_value = model_config
        mock_load_model.return_value = mock_model_provider

        mock_prompt_provider = MagicMock()
        system_prompt = "You are a QA engineer..."
        mock_prompt_provider.get.return_value = system_prompt
        mock_load_prompt.return_value = mock_prompt_provider

        source = AgentConfigSource(model_config_key="gemini_pro", prompt_key="qa/system")

        agent_config = source.to_agent_config(framework="langchain", name="custom_agent")

        assert agent_config.name == "custom_agent"
        assert agent_config.framework == "langchain"
        assert agent_config.model == "gemini-pro"

    @patch("gearmeshing_ai.info_provider.prompt.load_prompt_provider")
    @patch("gearmeshing_ai.info_provider.model.load_model_provider")
    def test_to_agent_config_no_overrides(self, mock_load_model, mock_load_prompt):
        """Test converting to AIAgentConfig without overrides."""
        # Setup mocks
        mock_model_provider = MagicMock()
        model_config = ModelConfig(provider="openai", model="gpt-4o", temperature=0.7, max_tokens=4096, top_p=0.9)
        mock_model_provider.get.return_value = model_config
        mock_load_model.return_value = mock_model_provider

        mock_prompt_provider = MagicMock()
        system_prompt = "You are a developer..."
        mock_prompt_provider.get.return_value = system_prompt
        mock_load_prompt.return_value = mock_prompt_provider

        source = AgentConfigSource(
            model_config_key="gpt4_default",
            prompt_key="dev/system",
            # No overrides
        )

        agent_config = source.to_agent_config(framework="pydantic_ai")

        assert agent_config.metadata == {}
