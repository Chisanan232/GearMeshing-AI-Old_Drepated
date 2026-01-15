"""Tests for model provider with database-driven configuration."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from gearmeshing_ai.agent_core.model_provider import (
    ModelProvider,
    create_model_for_role,
    get_model_provider,
)


class TestModelProvider:
    """Tests for ModelProvider class."""

    def test_model_provider_initialization_requires_db_session(self) -> None:
        """Test ModelProvider initialization requires database session."""
        with pytest.raises(ValueError, match="db_session is required"):
            ModelProvider(db_session=None)  # type: ignore[arg-type]

    def test_model_provider_initialization_with_db_session(self) -> None:
        """Test ModelProvider initialization with database session."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)
        assert provider.db_session is mock_session
        assert provider._db_provider is None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_create_openai_model_with_explicit_params(self) -> None:
        """Test creating OpenAI model with explicit parameters."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch("gearmeshing_ai.agent_core.model_provider.OpenAIResponsesModel") as mock_openai:
            mock_openai.return_value = MagicMock()
            provider._create_openai_model(
                "gpt-4o",
                temperature=0.5,
                max_tokens=2048,
                top_p=0.8,
            )
            mock_openai.assert_called_once()

    @patch.dict(os.environ, {})
    def test_create_openai_model_missing_api_key(self) -> None:
        """Test creating OpenAI model without API key."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            provider._create_openai_model("gpt-4o")

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_create_anthropic_model_with_explicit_params(self) -> None:
        """Test creating Anthropic model with explicit parameters."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch("gearmeshing_ai.agent_core.model_provider.AnthropicModel") as mock_anthropic:
            mock_anthropic.return_value = MagicMock()
            provider._create_anthropic_model(
                "claude-3-5-sonnet",
                temperature=0.6,
                max_tokens=3000,
                top_p=0.85,
            )
            mock_anthropic.assert_called_once()

    @patch.dict(os.environ, {})
    def test_create_anthropic_model_missing_api_key(self) -> None:
        """Test creating Anthropic model without API key."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            provider._create_anthropic_model("claude-3-5-sonnet")

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    def test_create_google_model_with_explicit_params(self) -> None:
        """Test creating Google model with explicit parameters."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch("gearmeshing_ai.agent_core.model_provider.GoogleModel") as mock_google:
            mock_google.return_value = MagicMock()
            provider._create_google_model(
                "gemini-2.0-flash",
                temperature=0.4,
                max_tokens=2500,
                top_p=0.75,
            )
            mock_google.assert_called_once()

    @patch.dict(os.environ, {})
    def test_create_google_model_missing_api_key(self) -> None:
        """Test creating Google model without API key."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
            provider._create_google_model("gemini-2.0-flash")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_create_model_with_provider_dispatch(self) -> None:
        """Test create_model dispatches to correct provider."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch.object(provider, "_create_openai_model") as mock_openai:
            mock_openai.return_value = MagicMock()
            provider.create_model("openai", "gpt-4o", temperature=0.7)
            mock_openai.assert_called_once()

    def test_create_model_with_unsupported_provider(self) -> None:
        """Test create_model with unsupported provider."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with pytest.raises(ValueError, match="Unsupported provider"):
            provider.create_model("unsupported", "model")

    def test_create_model_provider_case_insensitive(self) -> None:
        """Test that provider names are case-insensitive."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch.object(provider, "_create_openai_model") as mock_openai:
            mock_openai.return_value = MagicMock()
            provider.create_model("OPENAI", "gpt-4o")
            mock_openai.assert_called_once()

    def test_create_model_for_role_with_database(self) -> None:
        """Test create_model_for_role with database configuration."""
        from gearmeshing_ai.agent_core.schemas.config import ModelConfig

        mock_session = MagicMock()
        mock_db_provider = MagicMock()
        mock_db_provider.get_model_config.return_value = ModelConfig(
            provider="openai",
            model="gpt-4o",
            temperature=0.7,
            max_tokens=4096,
            top_p=0.9,
        )

        provider = ModelProvider(db_session=mock_session)
        provider._db_provider = mock_db_provider

        with patch.object(provider, "create_model") as mock_create:
            mock_create.return_value = MagicMock()
            provider.create_model_for_role("dev", tenant_id="test-tenant")
            mock_create.assert_called_once()

    def test_get_model_provider_returns_instance(self) -> None:
        """Test that get_model_provider returns ModelProvider instance."""
        mock_session = MagicMock()
        provider = get_model_provider(mock_session)

        assert isinstance(provider, ModelProvider)
        assert provider.db_session is mock_session


class TestModelProviderConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_model_for_role_convenience_function(self) -> None:
        """Test create_model_for_role convenience function."""
        mock_session = MagicMock()
        mock_provider = MagicMock()
        mock_provider.create_model_for_role.return_value = MagicMock()

        with patch("gearmeshing_ai.agent_core.model_provider.get_model_provider") as mock_get:
            mock_get.return_value = mock_provider

            model = create_model_for_role(mock_session, "dev", tenant_id="test-tenant")

            mock_provider.create_model_for_role.assert_called_once_with("dev", "test-tenant")


class TestModelProviderDefaults:
    """Tests for default parameter handling."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_openai_uses_defaults_when_params_none(self) -> None:
        """Test OpenAI model uses defaults when params are None."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch("gearmeshing_ai.agent_core.model_provider.OpenAIResponsesModel") as mock_openai:
            mock_openai.return_value = MagicMock()

            provider._create_openai_model("gpt-4o", temperature=None, max_tokens=None, top_p=None)

            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args.kwargs
            assert "settings" in call_kwargs
            assert mock_openai.called

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_anthropic_uses_defaults_when_params_none(self) -> None:
        """Test Anthropic model uses defaults when params are None."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch("gearmeshing_ai.agent_core.model_provider.AnthropicModel") as mock_anthropic:
            mock_anthropic.return_value = MagicMock()

            provider._create_anthropic_model("claude-3-5-sonnet", temperature=None, max_tokens=None, top_p=None)

            mock_anthropic.assert_called_once()
            call_kwargs = mock_anthropic.call_args.kwargs
            assert "settings" in call_kwargs
            assert mock_anthropic.called

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    def test_google_uses_defaults_when_params_none(self) -> None:
        """Test Google model uses defaults when params are None."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch("gearmeshing_ai.agent_core.model_provider.GoogleModel") as mock_google:
            mock_google.return_value = MagicMock()

            provider._create_google_model("gemini-2.0-flash", temperature=None, max_tokens=None, top_p=None)

            mock_google.assert_called_once()
            call_kwargs = mock_google.call_args.kwargs
            assert "settings" in call_kwargs
            assert mock_google.called


class TestModelProviderDispatch:
    """Tests for provider dispatch logic (lines 90-98)."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_create_model_dispatches_to_openai(self) -> None:
        """Test create_model dispatches to OpenAI provider."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch.object(provider, "_create_openai_model") as mock_openai:
            mock_openai.return_value = MagicMock()
            result = provider.create_model("openai", "gpt-4o", temperature=0.7, max_tokens=2048, top_p=0.9)

            mock_openai.assert_called_once_with("gpt-4o", 0.7, 2048, 0.9)
            assert result is not None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_create_model_dispatches_to_openai_case_insensitive(self) -> None:
        """Test create_model dispatches to OpenAI with case-insensitive provider name."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch.object(provider, "_create_openai_model") as mock_openai:
            mock_openai.return_value = MagicMock()
            provider.create_model("OpenAI", "gpt-4o")
            mock_openai.assert_called_once()

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_create_model_dispatches_to_anthropic(self) -> None:
        """Test create_model dispatches to Anthropic provider."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch.object(provider, "_create_anthropic_model") as mock_anthropic:
            mock_anthropic.return_value = MagicMock()
            result = provider.create_model(
                "anthropic", "claude-3-5-sonnet", temperature=0.6, max_tokens=3000, top_p=0.85
            )

            mock_anthropic.assert_called_once_with("claude-3-5-sonnet", 0.6, 3000, 0.85)
            assert result is not None

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_create_model_dispatches_to_anthropic_case_insensitive(self) -> None:
        """Test create_model dispatches to Anthropic with case-insensitive provider name."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch.object(provider, "_create_anthropic_model") as mock_anthropic:
            mock_anthropic.return_value = MagicMock()
            provider.create_model("ANTHROPIC", "claude-3-5-sonnet")
            mock_anthropic.assert_called_once()

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    def test_create_model_dispatches_to_google(self) -> None:
        """Test create_model dispatches to Google provider."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch.object(provider, "_create_google_model") as mock_google:
            mock_google.return_value = MagicMock()
            result = provider.create_model("google", "gemini-2.0-flash", temperature=0.4, max_tokens=2500, top_p=0.75)

            mock_google.assert_called_once_with("gemini-2.0-flash", 0.4, 2500, 0.75)
            assert result is not None

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    def test_create_model_dispatches_to_google_case_insensitive(self) -> None:
        """Test create_model dispatches to Google with case-insensitive provider name."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch.object(provider, "_create_google_model") as mock_google:
            mock_google.return_value = MagicMock()
            provider.create_model("Google", "gemini-2.0-flash")
            mock_google.assert_called_once()

    def test_create_model_raises_for_unsupported_provider(self) -> None:
        """Test create_model raises ValueError for unsupported provider."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with pytest.raises(ValueError, match="Unsupported provider: unsupported"):
            provider.create_model("unsupported", "some-model")

    def test_create_model_raises_for_empty_provider(self) -> None:
        """Test create_model raises ValueError for empty provider."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with pytest.raises(ValueError, match="Unsupported provider"):
            provider.create_model("", "some-model")

    def test_create_model_raises_for_invalid_provider_names(self) -> None:
        """Test create_model raises ValueError for various invalid provider names."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        invalid_providers = ["openai2", "claude", "gemini", "gpt", "mistral", "llama"]
        for invalid_provider in invalid_providers:
            with pytest.raises(ValueError, match="Unsupported provider"):
                provider.create_model(invalid_provider, "some-model")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "ANTHROPIC_API_KEY": "test-key"})
    def test_create_model_with_all_parameters(self) -> None:
        """Test create_model passes all parameters correctly to provider."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch.object(provider, "_create_openai_model") as mock_openai:
            mock_openai.return_value = MagicMock()
            provider.create_model(
                "openai",
                "gpt-4o",
                temperature=0.5,
                max_tokens=1024,
                top_p=0.95,
            )

            mock_openai.assert_called_once_with("gpt-4o", 0.5, 1024, 0.95)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_create_model_with_none_parameters(self) -> None:
        """Test create_model passes None parameters correctly to provider."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch.object(provider, "_create_openai_model") as mock_openai:
            mock_openai.return_value = MagicMock()
            provider.create_model("openai", "gpt-4o", temperature=None, max_tokens=None, top_p=None)

            mock_openai.assert_called_once_with("gpt-4o", None, None, None)


class TestModelProviderFallback:
    """Tests for fallback model creation logic (lines 255-258)."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "ANTHROPIC_API_KEY": "test-key"})
    def test_create_fallback_model_openai_to_anthropic(self) -> None:
        """Test creating fallback model with OpenAI primary and Anthropic fallback."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch.object(provider, "create_model") as mock_create:
            mock_primary = MagicMock()
            mock_fallback = MagicMock()
            mock_create.side_effect = [mock_primary, mock_fallback]

            with patch("gearmeshing_ai.agent_core.model_provider.FallbackModel") as mock_fallback_model:
                mock_fallback_model.return_value = MagicMock()
                result = provider.create_fallback_model(
                    "openai",
                    "gpt-4o",
                    "anthropic",
                    "claude-3-5-sonnet",
                    temperature=0.7,
                    max_tokens=2048,
                    top_p=0.9,
                )

                assert mock_create.call_count == 2
                mock_fallback_model.assert_called_once_with(mock_primary, mock_fallback)
                assert result is not None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "GOOGLE_API_KEY": "test-key"})
    def test_create_fallback_model_openai_to_google(self) -> None:
        """Test creating fallback model with OpenAI primary and Google fallback."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch.object(provider, "create_model") as mock_create:
            mock_primary = MagicMock()
            mock_fallback = MagicMock()
            mock_create.side_effect = [mock_primary, mock_fallback]

            with patch("gearmeshing_ai.agent_core.model_provider.FallbackModel") as mock_fallback_model:
                mock_fallback_model.return_value = MagicMock()
                result = provider.create_fallback_model(
                    "openai",
                    "gpt-4o",
                    "google",
                    "gemini-2.0-flash",
                )

                assert mock_create.call_count == 2
                mock_fallback_model.assert_called_once_with(mock_primary, mock_fallback)
                assert result is not None

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "GOOGLE_API_KEY": "test-key"})
    def test_create_fallback_model_anthropic_to_google(self) -> None:
        """Test creating fallback model with Anthropic primary and Google fallback."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch.object(provider, "create_model") as mock_create:
            mock_primary = MagicMock()
            mock_fallback = MagicMock()
            mock_create.side_effect = [mock_primary, mock_fallback]

            with patch("gearmeshing_ai.agent_core.model_provider.FallbackModel") as mock_fallback_model:
                mock_fallback_model.return_value = MagicMock()
                result = provider.create_fallback_model(
                    "anthropic",
                    "claude-3-5-sonnet",
                    "google",
                    "gemini-2.0-flash",
                    temperature=0.5,
                    max_tokens=3000,
                    top_p=0.8,
                )

                assert mock_create.call_count == 2
                mock_fallback_model.assert_called_once_with(mock_primary, mock_fallback)
                assert result is not None

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key", "OPENAI_API_KEY": "test-key"})
    def test_create_fallback_model_google_to_openai(self) -> None:
        """Test creating fallback model with Google primary and OpenAI fallback."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch.object(provider, "create_model") as mock_create:
            mock_primary = MagicMock()
            mock_fallback = MagicMock()
            mock_create.side_effect = [mock_primary, mock_fallback]

            with patch("gearmeshing_ai.agent_core.model_provider.FallbackModel") as mock_fallback_model:
                mock_fallback_model.return_value = MagicMock()
                result = provider.create_fallback_model(
                    "google",
                    "gemini-2.0-flash",
                    "openai",
                    "gpt-4o",
                )

                assert mock_create.call_count == 2
                mock_fallback_model.assert_called_once_with(mock_primary, mock_fallback)
                assert result is not None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "ANTHROPIC_API_KEY": "test-key"})
    def test_create_fallback_model_passes_parameters_to_both_models(self) -> None:
        """Test that fallback model creation passes parameters to both primary and fallback."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch.object(provider, "create_model") as mock_create:
            mock_primary = MagicMock()
            mock_fallback = MagicMock()
            mock_create.side_effect = [mock_primary, mock_fallback]

            with patch("gearmeshing_ai.agent_core.model_provider.FallbackModel") as mock_fallback_model:
                mock_fallback_model.return_value = MagicMock()
                provider.create_fallback_model(
                    "openai",
                    "gpt-4o",
                    "anthropic",
                    "claude-3-5-sonnet",
                    temperature=0.6,
                    max_tokens=2500,
                    top_p=0.85,
                )

                calls = mock_create.call_args_list
                assert len(calls) == 2
                # Check primary model call
                assert calls[0][0] == ("openai", "gpt-4o", 0.6, 2500, 0.85)
                # Check fallback model call
                assert calls[1][0] == ("anthropic", "claude-3-5-sonnet", 0.6, 2500, 0.85)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "ANTHROPIC_API_KEY": "test-key"})
    def test_create_fallback_model_with_none_parameters(self) -> None:
        """Test fallback model creation with None parameters."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch.object(provider, "create_model") as mock_create:
            mock_primary = MagicMock()
            mock_fallback = MagicMock()
            mock_create.side_effect = [mock_primary, mock_fallback]

            with patch("gearmeshing_ai.agent_core.model_provider.FallbackModel") as mock_fallback_model:
                mock_fallback_model.return_value = MagicMock()
                provider.create_fallback_model(
                    "openai",
                    "gpt-4o",
                    "anthropic",
                    "claude-3-5-sonnet",
                    temperature=None,
                    max_tokens=None,
                    top_p=None,
                )

                calls = mock_create.call_args_list
                assert len(calls) == 2
                # Both calls should have None parameters
                assert calls[0][0] == ("openai", "gpt-4o", None, None, None)
                assert calls[1][0] == ("anthropic", "claude-3-5-sonnet", None, None, None)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_create_fallback_model_primary_creation_fails(self) -> None:
        """Test fallback model creation when primary model creation fails."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch.object(provider, "create_model") as mock_create:
            mock_create.side_effect = RuntimeError("Primary model creation failed")

            with pytest.raises(RuntimeError, match="Primary model creation failed"):
                provider.create_fallback_model(
                    "openai",
                    "gpt-4o",
                    "anthropic",
                    "claude-3-5-sonnet",
                )

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "ANTHROPIC_API_KEY": "test-key"})
    def test_create_fallback_model_fallback_creation_fails(self) -> None:
        """Test fallback model creation when fallback model creation fails."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch.object(provider, "create_model") as mock_create:
            mock_primary = MagicMock()
            mock_create.side_effect = [mock_primary, RuntimeError("Fallback model creation failed")]

            with pytest.raises(RuntimeError, match="Fallback model creation failed"):
                provider.create_fallback_model(
                    "openai",
                    "gpt-4o",
                    "anthropic",
                    "claude-3-5-sonnet",
                )

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "ANTHROPIC_API_KEY": "test-key"})
    def test_create_fallback_model_returns_fallback_model_instance(self) -> None:
        """Test that create_fallback_model returns a FallbackModel instance."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch.object(provider, "create_model") as mock_create:
            mock_primary = MagicMock()
            mock_fallback = MagicMock()
            mock_create.side_effect = [mock_primary, mock_fallback]

            with patch("gearmeshing_ai.agent_core.model_provider.FallbackModel") as mock_fallback_model:
                expected_result = MagicMock()
                mock_fallback_model.return_value = expected_result

                result = provider.create_fallback_model(
                    "openai",
                    "gpt-4o",
                    "anthropic",
                    "claude-3-5-sonnet",
                )

                assert result is expected_result
                mock_fallback_model.assert_called_once_with(mock_primary, mock_fallback)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "ANTHROPIC_API_KEY": "test-key"})
    def test_create_fallback_model_same_provider_different_models(self) -> None:
        """Test fallback model with same provider but different models."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with patch.object(provider, "create_model") as mock_create:
            mock_primary = MagicMock()
            mock_fallback = MagicMock()
            mock_create.side_effect = [mock_primary, mock_fallback]

            with patch("gearmeshing_ai.agent_core.model_provider.FallbackModel") as mock_fallback_model:
                mock_fallback_model.return_value = MagicMock()
                result = provider.create_fallback_model(
                    "openai",
                    "gpt-4o",
                    "openai",
                    "gpt-4-turbo",
                )

                calls = mock_create.call_args_list
                assert calls[0][0] == ("openai", "gpt-4o", None, None, None)
                assert calls[1][0] == ("openai", "gpt-4-turbo", None, None, None)
                assert result is not None


class TestGetProviderFromModelName:
    """Tests for get_provider_from_model_name method."""

    def test_get_provider_from_model_name_gpt_models(self) -> None:
        """Test provider detection for GPT models."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        gpt_models = ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
        for model in gpt_models:
            result = provider.get_provider_from_model_name(model)
            assert result == "openai", f"Expected 'openai' for model {model}, got {result}"

    def test_get_provider_from_model_name_claude_models(self) -> None:
        """Test provider detection for Claude models."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        claude_models = ["claude-3-5-sonnet", "claude-3-opus", "claude-3-haiku", "claude-2.1"]
        for model in claude_models:
            result = provider.get_provider_from_model_name(model)
            assert result == "anthropic", f"Expected 'anthropic' for model {model}, got {result}"

    def test_get_provider_from_model_name_gemini_models(self) -> None:
        """Test provider detection for Gemini models."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        gemini_models = ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
        for model in gemini_models:
            result = provider.get_provider_from_model_name(model)
            assert result == "google", f"Expected 'google' for model {model}, got {result}"

    def test_get_provider_from_model_name_grok_models(self) -> None:
        """Test provider detection for Grok models."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        grok_models = ["grok-1", "grok-2", "grok-beta"]
        for model in grok_models:
            result = provider.get_provider_from_model_name(model)
            assert result == "grok", f"Expected 'grok' for model {model}, got {result}"

    def test_get_provider_from_model_name_case_insensitive(self) -> None:
        """Test that provider detection is case-insensitive."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        # Test uppercase
        result = provider.get_provider_from_model_name("GPT-4O")
        assert result == "openai"

        # Test mixed case
        result = provider.get_provider_from_model_name("Claude-3-Opus")
        assert result == "anthropic"

    def test_get_provider_from_model_name_invalid_model(self) -> None:
        """Test that invalid model names raise ValueError."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        invalid_models = ["unknown-model", "custom-model", "random-ai-model", "unsupported-llm"]
        for model in invalid_models:
            with pytest.raises(ValueError, match="Could not determine provider"):
                provider.get_provider_from_model_name(model)

    def test_get_provider_from_model_name_empty_string(self) -> None:
        """Test that empty model name raises ValueError."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with pytest.raises(ValueError, match="Could not determine provider"):
            provider.get_provider_from_model_name("")

    def test_get_provider_from_model_name_none_value(self) -> None:
        """Test that None model name raises appropriate error."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        with pytest.raises((ValueError, AttributeError)):
            provider.get_provider_from_model_name(None)  # type: ignore[arg-type]

    def test_get_provider_from_model_name_with_version_numbers(self) -> None:
        """Test provider detection with various version number formats."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        # GPT with version
        result = provider.get_provider_from_model_name("gpt-4o-2024-11-20")
        assert result == "openai"

        # Claude with version
        result = provider.get_provider_from_model_name("claude-3-5-sonnet-20241022")
        assert result == "anthropic"

        # Gemini with version
        result = provider.get_provider_from_model_name("gemini-2.0-flash-001")
        assert result == "google"

    def test_get_provider_from_model_name_gpt_4_variants(self) -> None:
        """Test provider detection for various GPT-4 variants."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        gpt4_variants = [
            "gpt-4",
            "gpt-4-32k",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-4o",
            "gpt-4o-mini",
        ]
        for model in gpt4_variants:
            result = provider.get_provider_from_model_name(model)
            assert result == "openai", f"Expected 'openai' for {model}"

    def test_get_provider_from_model_name_claude_variants(self) -> None:
        """Test provider detection for various Claude variants."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        claude_variants = [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022",
        ]
        for model in claude_variants:
            result = provider.get_provider_from_model_name(model)
            assert result == "anthropic", f"Expected 'anthropic' for {model}"

    def test_get_provider_from_model_name_gemini_variants(self) -> None:
        """Test provider detection for various Gemini variants."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        gemini_variants = [
            "gemini-pro",
            "gemini-pro-vision",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-2.0-flash",
        ]
        for model in gemini_variants:
            result = provider.get_provider_from_model_name(model)
            assert result == "google", f"Expected 'google' for {model}"

    def test_get_provider_from_model_name_returns_string(self) -> None:
        """Test that get_provider_from_model_name returns a string."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        result = provider.get_provider_from_model_name("gpt-4o")
        assert isinstance(result, str)
        assert result in ["openai", "anthropic", "google", "grok"]

    def test_get_provider_from_model_name_consistency(self) -> None:
        """Test that same model always returns same provider."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        model = "gpt-4o"
        result1 = provider.get_provider_from_model_name(model)
        result2 = provider.get_provider_from_model_name(model)
        assert result1 == result2

    def test_get_provider_from_model_name_special_characters(self) -> None:
        """Test provider detection with special characters in model name."""
        mock_session = MagicMock()
        provider = ModelProvider(db_session=mock_session)

        # Valid model with special characters
        result = provider.get_provider_from_model_name("gpt-4o-2024-11-20")
        assert result == "openai"

        # Invalid model with special characters
        with pytest.raises(ValueError):
            provider.get_provider_from_model_name("invalid@model#name")
