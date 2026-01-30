"""
Unit tests for AgentConfig model conversion methods.

Tests the to_model_config() and to_role_config() utility functions
that convert database entities to domain models.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Generator

import pytest
from sqlmodel import Session, create_engine

from gearmeshing_ai.core.database.entities.agent_configs import AgentConfig
from gearmeshing_ai.core.models.config import ModelConfig, RoleConfig
from gearmeshing_ai.core.models.io.agent_configs import (
    AgentConfigCreate,
    AgentConfigRead,
    AgentConfigUpdate,
)


@pytest.fixture
def in_memory_db() -> Generator[Session, None, None]:
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    AgentConfig.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def sample_agent_config() -> AgentConfig:
    """Create a sample agent configuration for testing."""
    return AgentConfig(
        role_name="dev",
        display_name="Developer",
        description="AI developer assistant",
        system_prompt_key="dev_system_prompt",
        model_provider="openai",
        model_name="gpt-4o",
        temperature=0.7,
        max_tokens=4096,
        top_p=0.9,
        capabilities='["code_generation", "debugging"]',
        tools='["file_editor", "terminal"]',
        autonomy_profiles='["supervised", "semi_autonomous"]',
        done_when="Task completed",
        is_active=True,
        tenant_id=None,
    )


class TestAgentConfigToModelConfig:
    """Tests for AgentConfig.to_model_config() method."""

    def test_to_model_config_basic(self, sample_agent_config: AgentConfig) -> None:
        """Test basic conversion to ModelConfig."""
        model_config: ModelConfig = sample_agent_config.to_model_config()

        assert isinstance(model_config, ModelConfig)
        assert model_config.provider == "openai"
        assert model_config.model == "gpt-4o"
        assert model_config.temperature == 0.7
        assert model_config.max_tokens == 4096
        assert model_config.top_p == 0.9

    def test_to_model_config_with_different_provider(self, sample_agent_config: AgentConfig) -> None:
        """Test conversion with different LLM provider."""
        sample_agent_config.model_provider = "anthropic"
        sample_agent_config.model_name = "claude-3-5-sonnet"

        model_config: ModelConfig = sample_agent_config.to_model_config()

        assert model_config.provider == "anthropic"
        assert model_config.model == "claude-3-5-sonnet"

    def test_to_model_config_with_custom_parameters(self, sample_agent_config: AgentConfig) -> None:
        """Test conversion with custom temperature and token settings."""
        sample_agent_config.temperature = 1.5
        sample_agent_config.max_tokens = 8192
        sample_agent_config.top_p = 0.95

        model_config: ModelConfig = sample_agent_config.to_model_config()

        assert model_config.temperature == 1.5
        assert model_config.max_tokens == 8192
        assert model_config.top_p == 0.95

    def test_to_model_config_returns_new_instance(self, sample_agent_config: AgentConfig) -> None:
        """Test that to_model_config() returns a new instance."""
        model_config1: ModelConfig = sample_agent_config.to_model_config()
        model_config2: ModelConfig = sample_agent_config.to_model_config()

        assert model_config1 is not model_config2
        assert model_config1 == model_config2


class TestAgentConfigToRoleConfig:
    """Tests for AgentConfig.to_role_config() method."""

    def test_to_role_config_basic(self, sample_agent_config: AgentConfig) -> None:
        """Test basic conversion to RoleConfig."""
        role_config: RoleConfig = sample_agent_config.to_role_config()

        assert isinstance(role_config, RoleConfig)
        assert role_config.role_name == "dev"
        assert role_config.display_name == "Developer"
        assert role_config.description == "AI developer assistant"
        assert role_config.system_prompt_key == "dev_system_prompt"
        assert role_config.done_when == "Task completed"

    def test_to_role_config_model_config(self, sample_agent_config: AgentConfig) -> None:
        """Test that RoleConfig contains correct ModelConfig."""
        role_config: RoleConfig = sample_agent_config.to_role_config()

        assert isinstance(role_config.model, ModelConfig)
        assert role_config.model.provider == "openai"
        assert role_config.model.model == "gpt-4o"
        assert role_config.model.temperature == 0.7
        assert role_config.model.max_tokens == 4096
        assert role_config.model.top_p == 0.9

    def test_to_role_config_capabilities_parsing(self, sample_agent_config: AgentConfig) -> None:
        """Test that capabilities JSON is correctly parsed."""
        role_config: RoleConfig = sample_agent_config.to_role_config()

        assert isinstance(role_config.capabilities, list)
        assert len(role_config.capabilities) == 2
        assert "code_generation" in role_config.capabilities
        assert "debugging" in role_config.capabilities

    def test_to_role_config_tools_parsing(self, sample_agent_config: AgentConfig) -> None:
        """Test that tools JSON is correctly parsed."""
        role_config: RoleConfig = sample_agent_config.to_role_config()

        assert isinstance(role_config.tools, list)
        assert len(role_config.tools) == 2
        assert "file_editor" in role_config.tools
        assert "terminal" in role_config.tools

    def test_to_role_config_autonomy_profiles_parsing(self, sample_agent_config: AgentConfig) -> None:
        """Test that autonomy_profiles JSON is correctly parsed."""
        role_config: RoleConfig = sample_agent_config.to_role_config()

        assert isinstance(role_config.autonomy_profiles, list)
        assert len(role_config.autonomy_profiles) == 2
        assert "supervised" in role_config.autonomy_profiles
        assert "semi_autonomous" in role_config.autonomy_profiles

    def test_to_role_config_empty_json_arrays(self, sample_agent_config: AgentConfig) -> None:
        """Test handling of empty JSON arrays."""
        sample_agent_config.capabilities = "[]"
        sample_agent_config.tools = "[]"
        sample_agent_config.autonomy_profiles = "[]"

        role_config: RoleConfig = sample_agent_config.to_role_config()

        assert role_config.capabilities == []
        assert role_config.tools == []
        assert role_config.autonomy_profiles == []

    def test_to_role_config_null_json_fields(self, sample_agent_config: AgentConfig) -> None:
        """Test handling of null/empty JSON fields."""
        sample_agent_config.capabilities = ""
        sample_agent_config.tools = ""
        sample_agent_config.autonomy_profiles = ""

        role_config: RoleConfig = sample_agent_config.to_role_config()

        assert role_config.capabilities == []
        assert role_config.tools == []
        assert role_config.autonomy_profiles == []

    def test_to_role_config_with_optional_fields(self, sample_agent_config: AgentConfig) -> None:
        """Test RoleConfig with optional fields set to None."""
        sample_agent_config.description = ""
        sample_agent_config.done_when = None

        role_config: RoleConfig = sample_agent_config.to_role_config()

        assert role_config.description == ""
        assert role_config.done_when is None

    def test_to_role_config_returns_new_instance(self, sample_agent_config: AgentConfig) -> None:
        """Test that to_role_config() returns a new instance."""
        role_config1: RoleConfig = sample_agent_config.to_role_config()
        role_config2: RoleConfig = sample_agent_config.to_role_config()

        assert role_config1 is not role_config2
        assert role_config1 == role_config2

    def test_to_role_config_complex_json(self, sample_agent_config: AgentConfig) -> None:
        """Test parsing of complex JSON arrays."""
        complex_capabilities = json.dumps(
            [
                "code_generation",
                "debugging",
                "refactoring",
                "documentation",
            ]
        )
        sample_agent_config.capabilities = complex_capabilities

        role_config: RoleConfig = sample_agent_config.to_role_config()

        assert len(role_config.capabilities) == 4
        assert "refactoring" in role_config.capabilities
        assert "documentation" in role_config.capabilities


class TestAgentConfigModels:
    """Tests for AgentConfig model variants."""

    def test_agent_config_create_schema(self) -> None:
        """Test AgentConfigCreate schema."""
        config_data = {
            "role_name": "planner",
            "display_name": "Planner",
            "description": "AI planning agent",
            "system_prompt_key": "planner_system_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "temperature": 0.5,
            "max_tokens": 2048,
            "top_p": 0.8,
        }
        config: AgentConfigCreate = AgentConfigCreate(**config_data)

        assert config.role_name == "planner"
        assert config.model_provider == "openai"

    def test_agent_config_update_schema(self) -> None:
        """Test AgentConfigUpdate schema with partial fields."""
        update_data = {
            "temperature": 0.9,
            "max_tokens": 8192,
        }
        update: AgentConfigUpdate = AgentConfigUpdate(**update_data)

        assert update.temperature == 0.9
        assert update.max_tokens == 8192
        assert update.display_name is None

    def test_agent_config_read_schema(self, sample_agent_config: AgentConfig) -> None:
        """Test AgentConfigRead schema includes timestamps."""
        sample_agent_config.id = 1
        sample_agent_config.created_at = datetime.utcnow()
        sample_agent_config.updated_at = datetime.utcnow()

        read_config: AgentConfigRead = AgentConfigRead.model_validate(sample_agent_config)

        assert read_config.id == 1
        assert read_config.created_at is not None
        assert read_config.updated_at is not None
