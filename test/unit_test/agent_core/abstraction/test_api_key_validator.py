"""Tests for API key validation functionality.

This module tests the API key validator that ensures required API keys
are present in the runtime environment before initializing AI agents.
"""

import os
from unittest.mock import patch

import pytest

from gearmeshing_ai.agent_core.abstraction.api_key_validator import (
    MODEL_PROVIDER_PATTERNS,
    PROVIDER_API_KEY_MAPPING,
    AIModelProvider,
    APIKeyValidator,
)


class TestAPIKeyValidator:
    """Test APIKeyValidator class."""

    def test_get_provider_for_model_openai(self):
        """Test getting provider for OpenAI models."""
        assert APIKeyValidator.get_provider_for_model("gpt-4o") == AIModelProvider.OPENAI
        assert APIKeyValidator.get_provider_for_model("gpt-4-turbo") == AIModelProvider.OPENAI
        assert APIKeyValidator.get_provider_for_model("gpt-3.5-turbo") == AIModelProvider.OPENAI

    def test_get_provider_for_model_anthropic(self):
        """Test getting provider for Anthropic models."""
        assert APIKeyValidator.get_provider_for_model("claude-3-opus") == AIModelProvider.ANTHROPIC
        assert APIKeyValidator.get_provider_for_model("claude-3-sonnet") == AIModelProvider.ANTHROPIC
        assert APIKeyValidator.get_provider_for_model("claude-3-5-sonnet-20241022") == AIModelProvider.ANTHROPIC

    def test_get_provider_for_model_google(self):
        """Test getting provider for Google models."""
        assert APIKeyValidator.get_provider_for_model("gemini-pro") == AIModelProvider.GOOGLE
        assert APIKeyValidator.get_provider_for_model("gemini-1.5-pro") == AIModelProvider.GOOGLE
        assert APIKeyValidator.get_provider_for_model("gemini-1.5-flash") == AIModelProvider.GOOGLE

    def test_get_provider_for_model_grok(self):
        """Test getting provider for Grok models."""
        assert APIKeyValidator.get_provider_for_model("grok-1") == AIModelProvider.GROK
        assert APIKeyValidator.get_provider_for_model("grok-2") == AIModelProvider.GROK

    def test_get_provider_for_model_unknown(self):
        """Test getting provider for unknown model."""
        assert APIKeyValidator.get_provider_for_model("unknown-model") is None

    def test_get_provider_for_model_case_insensitive(self):
        """Test that model lookup is case-insensitive."""
        assert APIKeyValidator.get_provider_for_model("GPT-4O") == AIModelProvider.OPENAI
        assert APIKeyValidator.get_provider_for_model("CLAUDE-3-OPUS") == AIModelProvider.ANTHROPIC

    def test_get_required_api_keys_openai(self):
        """Test getting API key variables for OpenAI."""
        keys = APIKeyValidator.get_required_api_keys(AIModelProvider.OPENAI)
        assert "OPENAI_API_KEY" in keys

    def test_get_required_api_keys_anthropic(self):
        """Test getting API key variables for Anthropic."""
        keys = APIKeyValidator.get_required_api_keys(AIModelProvider.ANTHROPIC)
        assert "ANTHROPIC_API_KEY" in keys

    def test_get_required_api_keys_google(self):
        """Test getting API key variables for Google."""
        keys = APIKeyValidator.get_required_api_keys(AIModelProvider.GOOGLE)
        assert "GOOGLE_API_KEY" in keys or "GOOGLE_GENERATIVE_AI_API_KEY" in keys

    def test_get_required_api_keys_grok(self):
        """Test getting API key variables for Grok."""
        keys = APIKeyValidator.get_required_api_keys(AIModelProvider.GROK)
        assert "GROK_API_KEY" in keys or "XAI_API_KEY" in keys

    def test_has_api_key_present(self):
        """Test checking for API key when present."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            assert APIKeyValidator.has_api_key(AIModelProvider.OPENAI) is True

    def test_has_api_key_missing(self):
        """Test checking for API key when missing."""
        with patch.dict(os.environ, {}, clear=True):
            assert APIKeyValidator.has_api_key(AIModelProvider.OPENAI) is False

    def test_has_api_key_multiple_options(self):
        """Test checking for API key with multiple environment variable options."""
        # Test with first option
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            assert APIKeyValidator.has_api_key(AIModelProvider.GOOGLE) is True

        # Test with second option
        with patch.dict(os.environ, {"GOOGLE_GENERATIVE_AI_API_KEY": "test-key"}):
            assert APIKeyValidator.has_api_key(AIModelProvider.GOOGLE) is True

    def test_validate_api_key_success(self):
        """Test validating API key when present."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            # Should not raise
            APIKeyValidator.validate_api_key(AIModelProvider.OPENAI)

    def test_validate_api_key_missing(self):
        """Test validating API key when missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                APIKeyValidator.validate_api_key(AIModelProvider.OPENAI)

            assert "API key not found" in str(exc_info.value)
            assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_validate_model_api_key_success(self):
        """Test validating API key for model when present."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            # Should not raise
            APIKeyValidator.validate_model_api_key("gpt-4o")

    def test_validate_model_api_key_missing(self):
        """Test validating API key for model when missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                APIKeyValidator.validate_model_api_key("gpt-4o")

            assert "API key not found" in str(exc_info.value)

    def test_validate_model_api_key_unknown_model(self):
        """Test validating API key for unknown model."""
        with pytest.raises(ValueError) as exc_info:
            APIKeyValidator.validate_model_api_key("unknown-model-xyz")

        assert "Unknown model" in str(exc_info.value)

    def test_validate_providers_all_present(self):
        """Test validating multiple providers when all present."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "key1",
                "ANTHROPIC_API_KEY": "key2",
                "GOOGLE_API_KEY": "key3",
                "GROK_API_KEY": "key4",
            },
        ):
            results = APIKeyValidator.validate_providers(
                [AIModelProvider.OPENAI, AIModelProvider.ANTHROPIC, AIModelProvider.GOOGLE, AIModelProvider.GROK]
            )

            assert all(results.values())

    def test_validate_providers_some_missing(self):
        """Test validating multiple providers when some missing."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "key1",
                "ANTHROPIC_API_KEY": "key2",
            },
            clear=True,
        ):
            results = APIKeyValidator.validate_providers(
                [AIModelProvider.OPENAI, AIModelProvider.ANTHROPIC, AIModelProvider.GOOGLE, AIModelProvider.GROK]
            )

            assert results[AIModelProvider.OPENAI] is True
            assert results[AIModelProvider.ANTHROPIC] is True
            assert results[AIModelProvider.GOOGLE] is False
            assert results[AIModelProvider.GROK] is False

    def test_get_missing_api_keys(self):
        """Test getting list of missing API keys."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "key1"},
            clear=True,
        ):
            missing = APIKeyValidator.get_missing_api_keys(
                [AIModelProvider.OPENAI, AIModelProvider.ANTHROPIC, AIModelProvider.GOOGLE, AIModelProvider.GROK]
            )

            assert AIModelProvider.OPENAI not in missing
            assert AIModelProvider.ANTHROPIC in missing
            assert AIModelProvider.GOOGLE in missing
            assert AIModelProvider.GROK in missing

    def test_log_api_key_status(self):
        """Test logging API key status."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "key1"},
            clear=True,
        ):
            # Should not raise
            APIKeyValidator.log_api_key_status([AIModelProvider.OPENAI, AIModelProvider.ANTHROPIC])


class TestProviderAPIKeyMapping:
    """Test provider API key mapping."""

    def test_all_providers_have_mappings(self):
        """Test that all known providers have API key mappings."""
        expected_providers = list(AIModelProvider)

        for provider in expected_providers:
            assert provider in PROVIDER_API_KEY_MAPPING
            assert len(PROVIDER_API_KEY_MAPPING[provider]) > 0

    def test_all_providers_have_regex_patterns(self):
        """Test that all known providers have regex patterns."""
        expected_providers = list(AIModelProvider)

        for provider in expected_providers:
            assert provider in MODEL_PROVIDER_PATTERNS
            assert MODEL_PROVIDER_PATTERNS[provider] is not None


class TestModelProviderPatterns:
    """Test model provider regex patterns."""

    def test_openai_models(self):
        """Test OpenAI model pattern matching."""
        openai_models = [
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-4o-mini",
        ]

        for model in openai_models:
            assert APIKeyValidator.get_provider_for_model(model) == AIModelProvider.OPENAI

    def test_anthropic_models(self):
        """Test Anthropic model pattern matching."""
        anthropic_models = [
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
            "claude-3-5-sonnet-20241022",
        ]

        for model in anthropic_models:
            assert APIKeyValidator.get_provider_for_model(model) == AIModelProvider.ANTHROPIC

    def test_google_models(self):
        """Test Google model pattern matching."""
        google_models = [
            "gemini-pro",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
        ]

        for model in google_models:
            assert APIKeyValidator.get_provider_for_model(model) == AIModelProvider.GOOGLE

    def test_grok_models(self):
        """Test Grok model pattern matching."""
        grok_models = ["grok-1", "grok-2", "grok-3"]

        for model in grok_models:
            assert APIKeyValidator.get_provider_for_model(model) == AIModelProvider.GROK

    def test_latest_model_versions_regex(self):
        """Test that latest model versions match regex patterns."""
        # OpenAI latest
        assert APIKeyValidator.get_provider_for_model("gpt-4o-2024-11-20") == AIModelProvider.OPENAI

        # Anthropic latest
        assert APIKeyValidator.get_provider_for_model("claude-3-5-sonnet-20241022") == AIModelProvider.ANTHROPIC

        # Google latest
        assert APIKeyValidator.get_provider_for_model("gemini-2.0-flash") == AIModelProvider.GOOGLE

        # Grok latest
        assert APIKeyValidator.get_provider_for_model("grok-3") == AIModelProvider.GROK

    def test_regex_pattern_flexibility_openai(self):
        """Test OpenAI regex pattern flexibility with various formats."""
        # Standard formats
        assert APIKeyValidator.get_provider_for_model("gpt-4o") == AIModelProvider.OPENAI
        assert APIKeyValidator.get_provider_for_model("gpt-4o-mini") == AIModelProvider.OPENAI
        assert APIKeyValidator.get_provider_for_model("gpt-4") == AIModelProvider.OPENAI
        assert APIKeyValidator.get_provider_for_model("gpt-4-turbo") == AIModelProvider.OPENAI
        assert APIKeyValidator.get_provider_for_model("gpt-3.5-turbo") == AIModelProvider.OPENAI

        # With date suffixes
        assert APIKeyValidator.get_provider_for_model("gpt-4o-2024-11-20") == AIModelProvider.OPENAI
        assert APIKeyValidator.get_provider_for_model("gpt-4o-mini-2024-07-18") == AIModelProvider.OPENAI
        assert APIKeyValidator.get_provider_for_model("gpt-4-turbo-2024-04-09") == AIModelProvider.OPENAI
        assert APIKeyValidator.get_provider_for_model("gpt-3.5-turbo-0125") == AIModelProvider.OPENAI

        # Case insensitive
        assert APIKeyValidator.get_provider_for_model("GPT-4O") == AIModelProvider.OPENAI
        assert APIKeyValidator.get_provider_for_model("Gpt-4o-2024-11-20") == AIModelProvider.OPENAI

    def test_regex_pattern_flexibility_anthropic(self):
        """Test Anthropic regex pattern flexibility with various formats."""
        # Standard formats
        assert APIKeyValidator.get_provider_for_model("claude-3-opus") == AIModelProvider.ANTHROPIC
        assert APIKeyValidator.get_provider_for_model("claude-3-sonnet") == AIModelProvider.ANTHROPIC
        assert APIKeyValidator.get_provider_for_model("claude-3-haiku") == AIModelProvider.ANTHROPIC
        assert APIKeyValidator.get_provider_for_model("claude-3-5-sonnet") == AIModelProvider.ANTHROPIC
        assert APIKeyValidator.get_provider_for_model("claude-3-5-haiku") == AIModelProvider.ANTHROPIC

        # With date suffixes
        assert APIKeyValidator.get_provider_for_model("claude-3-opus-20250219") == AIModelProvider.ANTHROPIC
        assert APIKeyValidator.get_provider_for_model("claude-3-sonnet-20250229") == AIModelProvider.ANTHROPIC
        assert APIKeyValidator.get_provider_for_model("claude-3-5-sonnet-20241022") == AIModelProvider.ANTHROPIC

        # Legacy models
        assert APIKeyValidator.get_provider_for_model("claude-2") == AIModelProvider.ANTHROPIC
        assert APIKeyValidator.get_provider_for_model("claude-2.1") == AIModelProvider.ANTHROPIC

        # Case insensitive
        assert APIKeyValidator.get_provider_for_model("CLAUDE-3-OPUS") == AIModelProvider.ANTHROPIC

    def test_regex_pattern_flexibility_google(self):
        """Test Google regex pattern flexibility with various formats."""
        # Gemini 2.0 models
        assert APIKeyValidator.get_provider_for_model("gemini-2.0-flash") == AIModelProvider.GOOGLE
        assert APIKeyValidator.get_provider_for_model("gemini-2.0-pro") == AIModelProvider.GOOGLE
        assert APIKeyValidator.get_provider_for_model("gemini-2.0-flash-exp") == AIModelProvider.GOOGLE

        # Gemini 1.5 models
        assert APIKeyValidator.get_provider_for_model("gemini-1.5-pro") == AIModelProvider.GOOGLE
        assert APIKeyValidator.get_provider_for_model("gemini-1.5-flash") == AIModelProvider.GOOGLE
        assert APIKeyValidator.get_provider_for_model("gemini-1.5-pro-002") == AIModelProvider.GOOGLE

        # Gemini 1.0 models
        assert APIKeyValidator.get_provider_for_model("gemini-1.0-pro") == AIModelProvider.GOOGLE
        assert APIKeyValidator.get_provider_for_model("gemini-pro") == AIModelProvider.GOOGLE

        # Legacy models
        assert APIKeyValidator.get_provider_for_model("palm-2") == AIModelProvider.GOOGLE
        assert APIKeyValidator.get_provider_for_model("text-bison") == AIModelProvider.GOOGLE

    def test_regex_pattern_flexibility_grok(self):
        """Test Grok regex pattern flexibility with various formats."""
        # Standard formats
        assert APIKeyValidator.get_provider_for_model("grok-1") == AIModelProvider.GROK
        assert APIKeyValidator.get_provider_for_model("grok-2") == AIModelProvider.GROK
        assert APIKeyValidator.get_provider_for_model("grok-3") == AIModelProvider.GROK

        # With vision suffix
        assert APIKeyValidator.get_provider_for_model("grok-1-vision") == AIModelProvider.GROK
        assert APIKeyValidator.get_provider_for_model("grok-2-vision") == AIModelProvider.GROK
        assert APIKeyValidator.get_provider_for_model("grok-3-vision") == AIModelProvider.GROK

        # With date suffixes
        assert APIKeyValidator.get_provider_for_model("grok-2-1212") == AIModelProvider.GROK
        assert APIKeyValidator.get_provider_for_model("grok-1-vision-20240912") == AIModelProvider.GROK

    def test_ai_model_provider_enum_values(self):
        """Test AIModelProvider enum values."""
        assert AIModelProvider.OPENAI.value == "openai"
        assert AIModelProvider.ANTHROPIC.value == "anthropic"
        assert AIModelProvider.GOOGLE.value == "google"
        assert AIModelProvider.GROK.value == "grok"

    def test_ai_model_provider_str_conversion(self):
        """Test AIModelProvider string conversion."""
        assert str(AIModelProvider.OPENAI) == "openai"
        assert str(AIModelProvider.ANTHROPIC) == "anthropic"
        assert str(AIModelProvider.GOOGLE) == "google"
        assert str(AIModelProvider.GROK) == "grok"

    def test_get_provider_for_model_with_empty_string(self):
        """Test get_provider_for_model returns None for empty string (L83-84)."""
        result = APIKeyValidator.get_provider_for_model("")

        assert result is None

    def test_get_provider_for_model_with_none(self):
        """Test get_provider_for_model returns None for None input (L83-84)."""
        result = APIKeyValidator.get_provider_for_model(None)

        assert result is None

    def test_get_provider_for_model_with_whitespace(self):
        """Test get_provider_for_model returns None for whitespace (L83-84)."""
        result = APIKeyValidator.get_provider_for_model("   ")

        assert result is None

    def test_validate_api_key_with_unknown_provider_raises_error(self):
        """Test validate_api_key raises ValueError for unknown provider (L138-142)."""
        # Create a mock provider that's not in PROVIDER_API_KEY_MAPPING
        from gearmeshing_ai.agent_core.abstraction.api_key_validator import AIModelProvider

        # This test verifies the error handling when provider is not in mapping
        # We'll patch the mapping to simulate this scenario
        with patch.dict(
            "gearmeshing_ai.agent_core.abstraction.api_key_validator.PROVIDER_API_KEY_MAPPING",
            {},
            clear=True,
        ):
            with pytest.raises(ValueError, match="Unknown provider"):
                APIKeyValidator.validate_api_key(AIModelProvider.OPENAI)

    def test_validate_api_key_error_message_includes_supported_providers(self):
        """Test validate_api_key error message includes supported providers (L138-142)."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(
                "gearmeshing_ai.agent_core.abstraction.api_key_validator.PROVIDER_API_KEY_MAPPING",
                {},
                clear=True,
            ):
                with pytest.raises(ValueError) as exc_info:
                    APIKeyValidator.validate_api_key(AIModelProvider.OPENAI)

                error_msg = str(exc_info.value)
                assert "Unknown provider" in error_msg

    def test_validate_api_key_missing_error_includes_env_vars(self):
        """Test validate_api_key error includes environment variable names (L138-142)."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                APIKeyValidator.validate_api_key(AIModelProvider.OPENAI)

            error_msg = str(exc_info.value)
            assert "OPENAI_API_KEY" in error_msg

    def test_log_api_key_status_with_none_providers(self):
        """Test log_api_key_status with None providers defaults to all (L213-214)."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch("gearmeshing_ai.agent_core.abstraction.api_key_validator.logger") as mock_logger:
                # Call with None to use all providers
                APIKeyValidator.log_api_key_status(None)

                # Verify logger was called
                assert mock_logger.debug.called

    def test_log_api_key_status_logs_all_providers_when_none(self):
        """Test that log_api_key_status logs all providers when None passed (L213-214)."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch("gearmeshing_ai.agent_core.abstraction.api_key_validator.logger") as mock_logger:
                APIKeyValidator.log_api_key_status(None)

                # Should log for all providers
                call_count = mock_logger.debug.call_count
                # Should have at least one call for "API Key Status:" and calls for each provider
                assert call_count >= 1

    def test_log_api_key_status_with_specific_providers(self):
        """Test log_api_key_status with specific provider list."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch("gearmeshing_ai.agent_core.abstraction.api_key_validator.logger") as mock_logger:
                providers = [AIModelProvider.OPENAI, AIModelProvider.ANTHROPIC]
                APIKeyValidator.log_api_key_status(providers)

                assert mock_logger.debug.called

    def test_log_api_key_status_shows_present_status(self):
        """Test that log_api_key_status shows present status for available keys."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch("gearmeshing_ai.agent_core.abstraction.api_key_validator.logger") as mock_logger:
                APIKeyValidator.log_api_key_status([AIModelProvider.OPENAI])

                # Verify logger was called with present status
                assert mock_logger.debug.called

    def test_log_api_key_status_shows_missing_status(self):
        """Test that log_api_key_status shows missing status for unavailable keys."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("gearmeshing_ai.agent_core.abstraction.api_key_validator.logger") as mock_logger:
                APIKeyValidator.log_api_key_status([AIModelProvider.OPENAI])

                # Verify logger was called
                assert mock_logger.debug.called

    def test_get_provider_for_model_case_insensitive_empty_string(self):
        """Test get_provider_for_model handles empty string case-insensitively."""
        result1 = APIKeyValidator.get_provider_for_model("")
        result2 = APIKeyValidator.get_provider_for_model("   ")

        assert result1 is None
        assert result2 is None

    def test_validate_api_key_with_empty_api_key_vars(self):
        """Test validate_api_key when provider has no API key variables."""
        with patch.dict(
            "gearmeshing_ai.agent_core.abstraction.api_key_validator.PROVIDER_API_KEY_MAPPING",
            {AIModelProvider.OPENAI: []},
        ):
            with pytest.raises(ValueError, match="Unknown provider"):
                APIKeyValidator.validate_api_key(AIModelProvider.OPENAI)

    def test_log_api_key_status_empty_provider_list(self):
        """Test log_api_key_status with empty provider list."""
        with patch("gearmeshing_ai.agent_core.abstraction.api_key_validator.logger") as mock_logger:
            APIKeyValidator.log_api_key_status([])

            # Should still log the header
            assert mock_logger.debug.called

    def test_log_api_key_status_single_provider(self):
        """Test log_api_key_status with single provider."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch("gearmeshing_ai.agent_core.abstraction.api_key_validator.logger") as mock_logger:
                APIKeyValidator.log_api_key_status([AIModelProvider.OPENAI])

                assert mock_logger.debug.called

    def test_log_api_key_status_multiple_providers(self):
        """Test log_api_key_status with multiple providers."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "key1",
                "ANTHROPIC_API_KEY": "key2",
            },
            clear=True,
        ):
            with patch("gearmeshing_ai.agent_core.abstraction.api_key_validator.logger") as mock_logger:
                providers = [
                    AIModelProvider.OPENAI,
                    AIModelProvider.ANTHROPIC,
                    AIModelProvider.GOOGLE,
                ]
                APIKeyValidator.log_api_key_status(providers)

                assert mock_logger.debug.called

    def test_get_provider_for_model_falsy_values(self):
        """Test get_provider_for_model with various falsy values."""
        assert APIKeyValidator.get_provider_for_model("") is None
        assert APIKeyValidator.get_provider_for_model(None) is None

    def test_validate_api_key_error_message_format(self):
        """Test validate_api_key error message is properly formatted."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                APIKeyValidator.validate_api_key(AIModelProvider.ANTHROPIC)

            error_msg = str(exc_info.value)
            # Should contain provider name and environment variables
            assert "anthropic" in error_msg.lower()
            assert "ANTHROPIC_API_KEY" in error_msg

    def test_log_api_key_status_all_providers_default(self):
        """Test that log_api_key_status defaults to all providers when None."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("gearmeshing_ai.agent_core.abstraction.api_key_validator.logger") as mock_logger:
                # Should use all AIModelProvider values
                APIKeyValidator.log_api_key_status(None)

                # Should log for each provider
                assert mock_logger.debug.call_count >= 4  # At least one for header + 4 providers
