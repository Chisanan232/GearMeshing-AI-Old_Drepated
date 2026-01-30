"""Unit tests for agent configuration entity model.

Tests SQLModel validation, field constraints, JSON helper methods,
and business logic for the AgentConfig entity.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from gearmeshing_ai.core.database.entities.agent_configs import (
    AgentConfig,
    AgentConfigBase,
)


class TestAgentConfigBase:
    """Tests for AgentConfigBase model validation."""

    def test_agent_config_base_valid_data(self, sample_agent_config_data):
        """Test AgentConfigBase with valid data."""
        config = AgentConfigBase(**sample_agent_config_data)

        assert config.role_name == "developer"
        assert config.display_name == "Software Developer"
        assert config.model_provider == "openai"
        assert config.model_name == "gpt-4o"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.is_active is True

    def test_agent_config_base_temperature_validation(self, sample_agent_config_data):
        """Test temperature field validation constraints."""
        # Test valid temperature range
        for temp in [0.0, 0.5, 1.0, 2.0]:
            data = sample_agent_config_data.copy()
            data["temperature"] = temp
            config = AgentConfigBase(**data)
            assert config.temperature == temp

        # Test invalid temperature values
        for temp in [-0.1, 2.1, 10.0]:
            data = sample_agent_config_data.copy()
            data["temperature"] = temp

            with pytest.raises(ValidationError) as exc_info:
                AgentConfigBase(**data)

            errors = exc_info.value.errors()
            assert any(error["loc"][0] == "temperature" for error in errors)

    def test_agent_config_base_max_tokens_validation(self, sample_agent_config_data):
        """Test max_tokens field validation constraints."""
        # Test valid max_tokens
        for tokens in [1, 100, 4096, 10000]:
            data = sample_agent_config_data.copy()
            data["max_tokens"] = tokens
            config = AgentConfigBase(**data)
            assert config.max_tokens == tokens

        # Test invalid max_tokens
        for tokens in [0, -1, -100]:
            data = sample_agent_config_data.copy()
            data["max_tokens"] = tokens

            with pytest.raises(ValidationError) as exc_info:
                AgentConfigBase(**data)

            errors = exc_info.value.errors()
            assert any(error["loc"][0] == "max_tokens" for error in errors)

    def test_agent_config_base_top_p_validation(self, sample_agent_config_data):
        """Test top_p field validation constraints."""
        # Test valid top_p range
        for top_p in [0.0, 0.5, 0.9, 1.0]:
            data = sample_agent_config_data.copy()
            data["top_p"] = top_p
            config = AgentConfigBase(**data)
            assert config.top_p == top_p

        # Test invalid top_p values
        for top_p in [-0.1, 1.1, 2.0]:
            data = sample_agent_config_data.copy()
            data["top_p"] = top_p

            with pytest.raises(ValidationError) as exc_info:
                AgentConfigBase(**data)

            errors = exc_info.value.errors()
            assert any(error["loc"][0] == "top_p" for error in errors)

    def test_agent_config_base_json_fields_default(self, sample_agent_config_data):
        """Test JSON fields have correct default values."""
        data = sample_agent_config_data.copy()
        # Remove JSON fields to test defaults
        for field in ["capabilities", "tools", "autonomy_profiles"]:
            if field in data:
                del data[field]

        config = AgentConfigBase(**data)

        assert config.capabilities == "[]"
        assert config.tools == "[]"
        assert config.autonomy_profiles == "[]"


class TestAgentConfig:
    """Tests for AgentConfig entity model."""

    def test_agent_config_creation_valid_data(self, sample_agent_config_data):
        """Test AgentConfig creation with valid data."""
        config = AgentConfig(**sample_agent_config_data)

        assert config.role_name == "developer"
        assert config.display_name == "Software Developer"
        assert config.model_provider == "openai"
        assert config.model_name == "gpt-4o"
        assert config.is_active is True
        assert config.id is None  # Will be set by database

    def test_agent_config_automatic_timestamps(self, sample_agent_config_data):
        """Test AgentConfig automatic timestamp generation."""
        config = AgentConfig(**sample_agent_config_data)

        assert config.created_at is not None
        assert config.updated_at is not None
        assert isinstance(config.created_at, datetime)
        assert isinstance(config.updated_at, datetime)

    def test_agent_config_get_capabilities_list(self, sample_agent_config_data):
        """Test get_capabilities_list method."""
        config = AgentConfig(**sample_agent_config_data)

        capabilities = config.get_capabilities_list()
        assert isinstance(capabilities, list)
        assert "code_generation" in capabilities
        assert "debugging" in capabilities
        assert "testing" in capabilities

    def test_agent_config_set_capabilities_list(self, sample_agent_config_data):
        """Test set_capabilities_list method."""
        config = AgentConfig(**sample_agent_config_data)

        new_capabilities = ["analysis", "design", "deployment"]
        config.set_capabilities_list(new_capabilities)

        assert config.get_capabilities_list() == new_capabilities
        assert json.loads(config.capabilities) == new_capabilities

    def test_agent_config_get_tools_list(self, sample_agent_config_data):
        """Test get_tools_list method."""
        config = AgentConfig(**sample_agent_config_data)

        tools = config.get_tools_list()
        assert isinstance(tools, list)
        assert "git" in tools
        assert "python" in tools
        assert "docker" in tools

    def test_agent_config_set_tools_list(self, sample_agent_config_data):
        """Test set_tools_list method."""
        config = AgentConfig(**sample_agent_config_data)

        new_tools = ["kubernetes", "terraform", "helm"]
        config.set_tools_list(new_tools)

        assert config.get_tools_list() == new_tools
        assert json.loads(config.tools) == new_tools

    def test_agent_config_get_autonomy_profiles_list(self, sample_agent_config_data):
        """Test get_autonomy_profiles_list method."""
        config = AgentConfig(**sample_agent_config_data)

        profiles = config.get_autonomy_profiles_list()
        assert isinstance(profiles, list)
        assert "balanced" in profiles
        assert "conservative" in profiles

    def test_agent_config_set_autonomy_profiles_list(self, sample_agent_config_data):
        """Test set_autonomy_profiles_list method."""
        config = AgentConfig(**sample_agent_config_data)

        new_profiles = ["aggressive", "balanced"]
        config.set_autonomy_profiles_list(new_profiles)

        assert config.get_autonomy_profiles_list() == new_profiles
        assert json.loads(config.autonomy_profiles) == new_profiles

    def test_agent_config_empty_json_fields(self, sample_agent_config_data):
        """Test handling of empty JSON fields."""
        data = sample_agent_config_data.copy()
        data["capabilities"] = "[]"
        data["tools"] = "[]"
        data["autonomy_profiles"] = "[]"

        config = AgentConfig(**data)

        assert config.get_capabilities_list() == []
        assert config.get_tools_list() == []
        assert config.get_autonomy_profiles_list() == []

    def test_agent_config_invalid_json_fields(self, sample_agent_config_data):
        """Test handling of invalid JSON in string fields."""
        data = sample_agent_config_data.copy()
        data["capabilities"] = "invalid json"

        config = AgentConfig(**data)

        # Should return empty list for invalid JSON
        assert config.get_capabilities_list() == []

    def test_agent_config_repr(self, sample_agent_config_data):
        """Test AgentConfig string representation."""
        config = AgentConfig(**sample_agent_config_data)

        repr_str = repr(config)
        assert "AgentConfig" in repr_str
        assert config.role_name in repr_str
        assert config.model_provider in repr_str
        assert config.model_name in repr_str

    def test_agent_config_table_name(self):
        """Test AgentConfig table name configuration."""
        assert AgentConfig.__tablename__ == "agent_configs"

    def test_agent_config_inheritance(self):
        """Test AgentConfig inherits from AgentConfigBase."""
        assert issubclass(AgentConfig, AgentConfigBase)

    def test_agent_config_tenant_specific_override(self, sample_agent_config_data):
        """Test tenant-specific configuration override."""
        # Global config (no tenant)
        global_data = sample_agent_config_data.copy()
        global_data["tenant_id"] = None
        global_config = AgentConfig(**global_data)

        # Tenant-specific config
        tenant_data = sample_agent_config_data.copy()
        tenant_data["tenant_id"] = "tenant_123"
        tenant_config = AgentConfig(**tenant_data)

        assert global_config.tenant_id is None
        assert tenant_config.tenant_id == "tenant_123"

    def test_agent_config_deactivation(self, sample_agent_config_data):
        """Test deactivating agent configuration."""
        config = AgentConfig(**sample_agent_config_data)

        assert config.is_active is True

        config.is_active = False
        assert config.is_active is False
