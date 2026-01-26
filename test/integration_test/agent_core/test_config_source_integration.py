"""Integration tests for AgentConfigSource and AIAgentProvider.

This module provides integration tests that verify the complete workflow
from configuration sources to agent creation using both model and prompt providers.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gearmeshing_ai.agent_core.abstraction.base import AIAgentBase, AIAgentConfig
from gearmeshing_ai.agent_core.abstraction.config_source import AgentConfigSource
from gearmeshing_ai.agent_core.abstraction.provider import AIAgentProvider


class TestAgentConfigSourceIntegration:
    """Integration tests for AgentConfigSource with real providers."""

    @pytest.mark.asyncio
    async def test_create_agent_from_config_source_hardcoded(self):
        """Test creating agent using hardcoded model and prompt providers."""
        # Setup provider
        provider = AIAgentProvider()
        mock_factory = MagicMock()
        mock_agent = MagicMock(spec=AIAgentBase)
        mock_factory.create = AsyncMock(return_value=mock_agent)
        mock_factory.is_registered.return_value = True
        provider._factory = mock_factory
        provider.set_framework("pydantic_ai")

        # Create config source
        config_source = AgentConfigSource(model_config_key="gpt4_default", prompt_key="dev/system")

        # Create agent
        agent = await provider.create_agent_from_config_source(config_source)

        # Verify
        assert agent == mock_agent
        mock_factory.create.assert_called_once()

        # Check the config passed to factory
        call_args = mock_factory.create.call_args
        agent_config = call_args[0][0]  # First positional argument

        assert isinstance(agent_config, AIAgentConfig)
        assert agent_config.framework == "pydantic_ai"
        assert agent_config.model == "gpt-4o"
        assert (
            agent_config.system_prompt
            == "You are a senior software engineer. Prefer small, safe changes, explicit assumptions, and tests."
        )
        assert agent_config.temperature == 0.7
        assert agent_config.max_tokens == 4096
        assert agent_config.top_p == 0.9

    @pytest.mark.asyncio
    async def test_create_agent_from_config_source_with_overrides(self):
        """Test creating agent with runtime overrides."""
        # Setup provider
        provider = AIAgentProvider()
        mock_factory = MagicMock()
        mock_agent = MagicMock(spec=AIAgentBase)
        mock_factory.create = AsyncMock(return_value=mock_agent)
        mock_factory.is_registered.return_value = True
        provider._factory = mock_factory
        provider.set_framework("pydantic_ai")

        # Create config source with overrides
        config_source = AgentConfigSource(
            model_config_key="gpt4_precise",
            prompt_key="qa/system",
            overrides={"temperature": 0.1, "max_tokens": 8000, "custom_setting": "value"},
        )

        # Create agent
        agent = await provider.create_agent_from_config_source(config_source)

        # Verify overrides are applied
        call_args = mock_factory.create.call_args
        agent_config = call_args[0][0]

        assert agent_config.temperature == 0.1  # Override applied
        assert agent_config.max_tokens == 8000  # Override applied
        assert agent_config.metadata == {"custom_setting": "value"}  # Custom setting

    @pytest.mark.asyncio
    async def test_create_agent_from_config_source_with_tenant(self):
        """Test creating agent with tenant-specific configuration."""
        # Setup provider
        provider = AIAgentProvider()
        mock_factory = MagicMock()
        mock_agent = MagicMock(spec=AIAgentBase)
        mock_factory.create = AsyncMock(return_value=mock_agent)
        mock_factory.is_registered.return_value = True
        provider._factory = mock_factory
        provider.set_framework("pydantic_ai")

        # Create config source with tenant
        config_source = AgentConfigSource(
            model_config_key="claude_sonnet",
            tenant_id="acme_corp",
            prompt_key="pm/system",
            prompt_tenant_id="acme_corp",
        )

        # Create agent
        agent = await provider.create_agent_from_config_source(config_source)

        # Verify tenant is used in name generation
        call_args = mock_factory.create.call_args
        agent_config = call_args[0][0]

        assert agent_config.name == "agent_claude_sonnet_pm/system_acme_corp"

    @pytest.mark.asyncio
    async def test_create_agent_from_config_source_custom_name(self):
        """Test creating agent with custom name."""
        # Setup provider
        provider = AIAgentProvider()
        mock_factory = MagicMock()
        mock_agent = MagicMock(spec=AIAgentBase)
        mock_factory.create = AsyncMock(return_value=mock_agent)
        mock_factory.is_registered.return_value = True
        provider._factory = mock_factory
        provider.set_framework("langchain")

        # Create config source
        config_source = AgentConfigSource(model_config_key="gemini_pro", prompt_key="dev/system")

        # Create agent with custom name
        agent = await provider.create_agent_from_config_source(config_source, use_cache=False)

        # Verify framework and cache parameter
        call_args = mock_factory.create.call_args
        agent_config = call_args[0][0]
        use_cache = call_args[1]["use_cache"]

        assert agent_config.framework == "langchain"  # Set by provider
        assert use_cache is False

    @pytest.mark.asyncio
    async def test_create_agent_from_config_source_provider_not_initialized(self):
        """Test error handling when provider is not initialized."""
        provider = AIAgentProvider()
        # Don't initialize factory

        config_source = AgentConfigSource(model_config_key="gpt4_default", prompt_key="dev/system")

        with pytest.raises(RuntimeError, match="Factory not initialized"):
            await provider.create_agent_from_config_source(config_source)

    @pytest.mark.asyncio
    async def test_create_agent_from_config_source_framework_not_set(self):
        """Test error handling when framework is not set."""
        provider = AIAgentProvider()
        mock_factory = MagicMock()
        provider._factory = mock_factory
        # Don't set framework

        config_source = AgentConfigSource(model_config_key="gpt4_default", prompt_key="dev/system")

        with pytest.raises(RuntimeError, match="Framework not set"):
            await provider.create_agent_from_config_source(config_source)

    @pytest.mark.asyncio
    async def test_create_agent_from_config_source_model_config_not_found(self):
        """Test error handling when model config is not found."""
        # Setup provider
        provider = AIAgentProvider()
        mock_factory = MagicMock()
        mock_factory.is_registered.return_value = True
        provider._factory = mock_factory
        provider.set_framework("pydantic_ai")

        # Create config source with non-existent model config
        config_source = AgentConfigSource(model_config_key="nonexistent_model", prompt_key="dev/system")

        with pytest.raises(KeyError, match="model config not found"):
            await provider.create_agent_from_config_source(config_source)

    @pytest.mark.asyncio
    async def test_create_agent_from_config_source_prompt_not_found(self):
        """Test error handling when prompt is not found."""
        # Setup provider
        provider = AIAgentProvider()
        mock_factory = MagicMock()
        mock_factory.is_registered.return_value = True
        provider._factory = mock_factory
        provider.set_framework("pydantic_ai")

        # Create config source with non-existent prompt
        config_source = AgentConfigSource(model_config_key="gpt4_default", prompt_key="nonexistent_prompt")

        with pytest.raises(KeyError, match="prompt not found"):
            await provider.create_agent_from_config_source(config_source)


class TestAgentConfigSourceWorkflow:
    """Test complete workflow scenarios."""

    @pytest.mark.asyncio
    async def test_developer_agent_workflow(self):
        """Test complete workflow for creating a developer agent."""
        # Setup provider
        provider = AIAgentProvider()
        mock_factory = MagicMock()
        mock_agent = MagicMock(spec=AIAgentBase)
        mock_factory.create = AsyncMock(return_value=mock_agent)
        mock_factory.is_registered.return_value = True
        provider._factory = mock_factory
        provider.set_framework("pydantic_ai")

        # Create developer agent configuration
        config_source = AgentConfigSource(
            model_config_key="gpt4_precise",  # Use precise settings for development
            prompt_key="dev/system",  # Use developer system prompt
            overrides={
                "temperature": 0.1,  # Even more precise for code generation
                "max_tokens": 8000,  # Allow longer responses
                "code_style": "pep8",  # Custom development setting
                "testing_framework": "pytest",  # Testing preference
            },
        )

        # Create agent
        agent = await provider.create_agent_from_config_source(config_source)

        # Verify the complete configuration
        call_args = mock_factory.create.call_args
        agent_config = call_args[0][0]

        assert agent_config.name == "agent_gpt4_precise_dev/system"
        assert agent_config.model == "gpt-4o"
        assert (
            agent_config.system_prompt
            == "You are a senior software engineer. Prefer small, safe changes, explicit assumptions, and tests."
        )
        assert agent_config.temperature == 0.1  # Override applied
        assert agent_config.max_tokens == 8000  # Override applied
        assert agent_config.metadata == {"code_style": "pep8", "testing_framework": "pytest"}

    @pytest.mark.asyncio
    async def test_qa_agent_workflow(self):
        """Test complete workflow for creating a QA agent."""
        # Setup provider
        provider = AIAgentProvider()
        mock_factory = MagicMock()
        mock_agent = MagicMock(spec=AIAgentBase)
        mock_factory.create = AsyncMock(return_value=mock_agent)
        mock_factory.is_registered.return_value = True
        provider._factory = mock_factory
        provider.set_framework("pydantic_ai")

        # Create QA agent configuration
        config_source = AgentConfigSource(
            model_config_key="claude_haiku",  # Fast model for quick checks
            prompt_key="qa/system",  # QA-specific prompt
            locale="en",
            overrides={
                "focus_areas": ["edge_cases", "regressions", "observability"],
                "test_types": ["unit", "integration", "e2e"],
                "quality_gates": True,
            },
        )

        # Create agent
        agent = await provider.create_agent_from_config_source(config_source)

        # Verify QA-specific configuration
        call_args = mock_factory.create.call_args
        agent_config = call_args[0][0]

        assert agent_config.name == "agent_claude_haiku_qa/system"
        assert agent_config.model == "claude-3-5-haiku-20241022"
        assert (
            agent_config.system_prompt
            == "You are a meticulous QA engineer. Think in terms of edge cases, regressions, and observability."
        )
        assert agent_config.temperature == 0.5  # From claude_haiku config
        assert agent_config.metadata == {
            "focus_areas": ["edge_cases", "regressions", "observability"],
            "test_types": ["unit", "integration", "e2e"],
            "quality_gates": True,
        }

    @pytest.mark.asyncio
    async def test_multilingual_agent_workflow(self):
        """Test workflow for creating a multilingual agent."""
        # Setup provider
        provider = AIAgentProvider()
        mock_factory = MagicMock()
        mock_agent = MagicMock(spec=AIAgentBase)
        mock_factory.create = AsyncMock(return_value=mock_agent)
        mock_factory.is_registered.return_value = True
        provider._factory = mock_factory
        provider.set_framework("pydantic_ai")

        # Create English agent configuration (since builtin provider only supports English)
        config_source = AgentConfigSource(
            model_config_key="gpt4_creative",  # Creative model for nuanced language
            prompt_key="dev/system",  # Use existing prompt (would be translated in real system)
            locale="en",  # Use English since builtin provider only supports English
            overrides={"language": "en-US", "formality": "formal", "cultural_context": "western"},
        )

        # Create agent
        agent = await provider.create_agent_from_config_source(config_source)

        # Verify multilingual configuration
        call_args = mock_factory.create.call_args
        agent_config = call_args[0][0]

        assert agent_config.name == "agent_gpt4_creative_dev/system"
        assert agent_config.temperature == 1.0  # From gpt4_creative config
        assert agent_config.metadata == {"language": "en-US", "formality": "formal", "cultural_context": "western"}

    @pytest.mark.asyncio
    async def test_tenant_isolated_workflow(self):
        """Test workflow with tenant-specific configurations."""
        # Setup provider
        provider = AIAgentProvider()
        mock_factory = MagicMock()
        mock_agent = MagicMock(spec=AIAgentBase)
        mock_factory.create = AsyncMock(return_value=mock_agent)
        mock_factory.is_registered.return_value = True
        provider._factory = mock_factory
        provider.set_framework("pydantic_ai")

        # Create tenant-specific agent
        config_source = AgentConfigSource(
            model_config_key="gpt4_default",
            tenant_id="enterprise_client",
            prompt_key="dev/system",
            prompt_tenant_id="enterprise_client",
            overrides={"compliance_mode": "strict", "data_privacy": "enterprise", "audit_logging": True},
        )

        # Create agent
        agent = await provider.create_agent_from_config_source(config_source)

        # Verify tenant isolation
        call_args = mock_factory.create.call_args
        agent_config = call_args[0][0]

        assert agent_config.name == "agent_gpt4_default_dev/system_enterprise_client"
        assert agent_config.metadata == {
            "compliance_mode": "strict",
            "data_privacy": "enterprise",
            "audit_logging": True,
        }
