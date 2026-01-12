"""Tests for PydanticAIAgent model_settings handling."""

from __future__ import annotations


from gearmeshing_ai.agent_core.abstraction import AIAgentConfig
from gearmeshing_ai.agent_core.abstraction.adapters.pydantic_ai import PydanticAIAgent


class TestPydanticAIAgentModelSettings:
    """Tests for model_settings property handling in PydanticAIAgent."""

    def test_build_init_kwargs_with_model_settings_property(self) -> None:
        """Test that model_settings from config are merged into kwargs."""
        config = AIAgentConfig(
            name="test-agent",
            framework="pydantic_ai",
            model="gpt-4o",
            system_prompt="Test prompt",
            temperature=0.8,
            max_tokens=2000,
            model_settings={
                "top_p": 0.95,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.1,
            },
            metadata={"output_type": dict},
        )

        agent = PydanticAIAgent(config)
        kwargs = agent.build_init_kwargs()

        # Verify model_settings is set and includes both config values and model_settings
        assert "model_settings" in kwargs
        model_settings = kwargs["model_settings"]
        assert model_settings["temperature"] == 0.8
        assert model_settings["max_tokens"] == 2000
        assert model_settings["top_p"] == 0.95
        assert model_settings["frequency_penalty"] == 0.5
        assert model_settings["presence_penalty"] == 0.1

    def test_build_init_kwargs_with_empty_model_settings(self) -> None:
        """Test that empty model_settings doesn't break kwargs building."""
        config = AIAgentConfig(
            name="test-agent",
            framework="pydantic_ai",
            model="gpt-4o",
            system_prompt="Test prompt",
            temperature=0.7,
            model_settings={},  # Empty model_settings
            metadata={"output_type": list},
        )

        agent = PydanticAIAgent(config)
        kwargs = agent.build_init_kwargs()

        # Should still have model_settings with temperature and max_tokens
        assert "model_settings" in kwargs
        assert kwargs["model_settings"]["temperature"] == 0.7

    def test_build_init_kwargs_without_model_settings_property(self) -> None:
        """Test that missing model_settings property doesn't cause errors."""
        config = AIAgentConfig(
            name="test-agent",
            framework="pydantic_ai",
            model="gpt-4o",
            system_prompt="Test prompt",
            temperature=0.6,
            max_tokens=1500,
            # No model_settings property set (defaults to empty dict)
            metadata={"output_type": str},
        )

        agent = PydanticAIAgent(config)
        kwargs = agent.build_init_kwargs()

        # Should still build kwargs correctly
        assert "model_settings" in kwargs
        assert kwargs["model_settings"]["temperature"] == 0.6
        assert kwargs["model_settings"]["max_tokens"] == 1500

    def test_build_init_kwargs_model_settings_override_temperature(self) -> None:
        """Test that model_settings can override temperature from config."""
        config = AIAgentConfig(
            name="test-agent",
            framework="pydantic_ai",
            model="gpt-4o",
            temperature=0.5,  # Base temperature
            model_settings={
                "temperature": 0.9,  # Override temperature
                "top_k": 40,
            },
            metadata={"output_type": dict},
        )

        agent = PydanticAIAgent(config)
        kwargs = agent.build_init_kwargs()

        # model_settings should override the base temperature
        assert kwargs["model_settings"]["temperature"] == 0.9
        assert kwargs["model_settings"]["top_k"] == 40

    def test_build_init_kwargs_model_settings_with_custom_parameters(self) -> None:
        """Test that custom model parameters in model_settings are preserved."""
        config = AIAgentConfig(
            name="test-agent",
            framework="pydantic_ai",
            model="claude-3-opus",
            temperature=0.7,
            model_settings={
                "custom_param_1": "value1",
                "custom_param_2": 42,
                "custom_param_3": {"nested": "value"},
            },
            metadata={"output_type": list},
        )

        agent = PydanticAIAgent(config)
        kwargs = agent.build_init_kwargs()

        # Custom parameters should be preserved
        assert kwargs["model_settings"]["custom_param_1"] == "value1"
        assert kwargs["model_settings"]["custom_param_2"] == 42
        assert kwargs["model_settings"]["custom_param_3"] == {"nested": "value"}

    def test_build_init_kwargs_metadata_separate_from_model_settings(self) -> None:
        """Test that metadata parameters are separate from model_settings."""
        config = AIAgentConfig(
            name="test-agent",
            framework="pydantic_ai",
            model="gpt-4o",
            temperature=0.7,
            model_settings={
                "top_p": 0.9,
                "frequency_penalty": 0.5,
            },
            metadata={
                "output_type": dict,
                "custom_agent_param": "agent_value",
            },
        )

        agent = PydanticAIAgent(config)
        kwargs = agent.build_init_kwargs()

        # model_settings should only have model parameters
        assert "top_p" in kwargs["model_settings"]
        assert "frequency_penalty" in kwargs["model_settings"]
        assert "output_type" not in kwargs["model_settings"]

        # metadata should be in kwargs directly
        assert kwargs["output_type"] == dict
        assert kwargs["custom_agent_param"] == "agent_value"

    def test_build_init_kwargs_with_max_tokens_none(self) -> None:
        """Test handling when max_tokens is None."""
        config = AIAgentConfig(
            name="test-agent",
            framework="pydantic_ai",
            model="gpt-4o",
            temperature=0.7,  # temperature has default, cannot be None
            max_tokens=None,  # max_tokens can be None
            model_settings={"top_p": 0.9},
            metadata={"output_type": str},
        )

        agent = PydanticAIAgent(config)
        kwargs = agent.build_init_kwargs()

        # Should still have model_settings with provided values
        assert "model_settings" in kwargs
        assert kwargs["model_settings"]["temperature"] == 0.7
        assert "max_tokens" not in kwargs["model_settings"]  # None values excluded
        assert kwargs["model_settings"]["top_p"] == 0.9
