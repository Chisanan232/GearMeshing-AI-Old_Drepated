"""Integration tests for model provider with engine and planner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gearmeshing_ai.agent_core.model_provider import (
    ModelProvider,
    create_model_for_role,
    get_model_provider,
)
from gearmeshing_ai.agent_core.schemas.config import ModelConfig


class TestModelProviderIntegration:
    """Integration tests for ModelProvider with database configuration."""

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        return MagicMock()

    @pytest.fixture
    def mock_model_config(self):
        """Create a mock model configuration."""
        return ModelConfig(
            role="dev",
            provider="openai",
            model="gpt-4o",
            temperature=0.7,
            max_tokens=4096,
            top_p=0.9,
        )

    def test_get_model_provider_returns_provider(self, mock_db_session):
        """Test get_model_provider factory function."""
        from pydantic import SecretStr

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.ai_provider.openai.api_key = SecretStr("test-key")
            provider = get_model_provider(mock_db_session)
            assert isinstance(provider, ModelProvider)
            assert provider.db_session is mock_db_session

    def test_model_provider_lazy_loads_db_provider(self, mock_db_session):
        """Test that ModelProvider lazily loads DatabaseConfigProvider."""
        from pydantic import SecretStr

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.ai_provider.openai.api_key = SecretStr("test-key")
            provider = ModelProvider(db_session=mock_db_session)
            assert provider._db_provider is None

            # Access db_provider
            db_provider = provider._get_db_provider()
            assert db_provider is not None
            assert provider._db_provider is db_provider

            # Second access returns same instance
            db_provider2 = provider._get_db_provider()
            assert db_provider2 is db_provider

    def test_create_model_dispatches_to_correct_provider(self, mock_db_session):
        """Test that create_model dispatches to correct provider implementation."""
        from pydantic import SecretStr

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.ai_provider.openai.api_key = SecretStr("test-key")
            provider = ModelProvider(db_session=mock_db_session)

            with patch.object(provider, "_create_openai_model") as mock_openai:
                mock_openai.return_value = MagicMock()
                provider.create_model("openai", "gpt-4o", temperature=0.5)
                mock_openai.assert_called_once_with("gpt-4o", 0.5, None, None)

    def test_create_model_dispatches_anthropic(self, mock_db_session):
        """Test that create_model dispatches to Anthropic provider."""
        from pydantic import SecretStr

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.ai_provider.anthropic.api_key = SecretStr("test-key")
            provider = ModelProvider(db_session=mock_db_session)

            with patch.object(provider, "_create_anthropic_model") as mock_anthropic:
                mock_anthropic.return_value = MagicMock()
                provider.create_model("anthropic", "claude-3-5-sonnet")
                mock_anthropic.assert_called_once()

    def test_create_model_dispatches_google(self, mock_db_session):
        """Test that create_model dispatches to Google provider."""
        from pydantic import SecretStr

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.ai_provider.google.api_key = SecretStr("test-key")
            provider = ModelProvider(db_session=mock_db_session)

            with patch.object(provider, "_create_google_model") as mock_google:
                mock_google.return_value = MagicMock()
                provider.create_model("google", "gemini-2.0-flash")
                mock_google.assert_called_once()

    def test_create_model_raises_on_unsupported_provider(self, mock_db_session):
        """Test that create_model raises ValueError for unsupported provider."""
        provider = ModelProvider(db_session=mock_db_session)

        with pytest.raises(ValueError, match="Unsupported provider"):
            provider.create_model("unsupported", "some-model")

    def test_create_fallback_model_creates_both_models(self, mock_db_session):
        """Test that create_fallback_model creates both primary and fallback models."""
        from pydantic import SecretStr

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.ai_provider.openai.api_key = SecretStr("test-key")
            mock_settings.ai_provider.anthropic.api_key = SecretStr("test-key")
            provider = ModelProvider(db_session=mock_db_session)

            with patch.object(provider, "create_model") as mock_create:
                mock_primary = MagicMock()
                mock_fallback = MagicMock()
                mock_create.side_effect = [mock_primary, mock_fallback]

                with patch("gearmeshing_ai.agent_core.model_provider.FallbackModel") as mock_fallback_model:
                    mock_fallback_model.return_value = MagicMock()
                    result = provider.create_fallback_model(
                        "openai", "gpt-4o", "anthropic", "claude-3-5-sonnet", temperature=0.5
                    )

                    assert mock_create.call_count == 2
                    # Verify both models were created with correct parameters
                    calls = mock_create.call_args_list
                    # Check positional arguments
                    assert calls[0][0][0] == "openai"  # provider
                    assert calls[0][0][1] == "gpt-4o"  # model
                    assert calls[1][0][0] == "anthropic"  # provider
                    assert calls[1][0][1] == "claude-3-5-sonnet"  # model

    def test_create_model_for_role_uses_db_config(self, mock_db_session, mock_model_config):
        """Test that create_model_for_role loads config from database."""
        from pydantic import SecretStr

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.ai_provider.openai.api_key = SecretStr("test-key")
            provider = ModelProvider(db_session=mock_db_session)

            with patch.object(provider, "_get_db_provider") as mock_get_db:
                mock_db_provider = MagicMock()
                mock_db_provider.get.return_value = mock_model_config
                mock_get_db.return_value = mock_db_provider

                with patch.object(provider, "create_model") as mock_create:
                    mock_create.return_value = MagicMock()
                    provider.create_model_for_role("dev", tenant_id="acme-corp")

                    # Verify database config was fetched
                    mock_db_provider.get.assert_called_once_with("dev", "acme-corp")

                    # Verify create_model was called with config values
                    mock_create.assert_called_once_with(
                        provider="openai",
                        model="gpt-4o",
                        temperature=0.7,
                        max_tokens=4096,
                        top_p=0.9,
                    )

    def test_create_model_for_role_function(self, mock_db_session, mock_model_config):
        """Test the module-level create_model_for_role function."""
        from pydantic import SecretStr

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.ai_provider.openai.api_key = SecretStr("test-key")
            with patch("gearmeshing_ai.agent_core.model_provider.get_model_provider") as mock_get_provider:
                mock_provider = MagicMock()
                mock_provider.create_model_for_role.return_value = MagicMock()
                mock_get_provider.return_value = mock_provider

                result = create_model_for_role(mock_db_session, "dev", tenant_id="acme-corp")

                mock_get_provider.assert_called_once_with(mock_db_session)
                mock_provider.create_model_for_role.assert_called_once_with("dev", "acme-corp")

    def test_create_model_with_all_parameters(self, mock_db_session):
        """Test create_model with all parameters specified."""
        from pydantic import SecretStr

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.ai_provider.openai.api_key = SecretStr("test-key")
            provider = ModelProvider(db_session=mock_db_session)

            with patch("gearmeshing_ai.agent_core.model_provider.OpenAIResponsesModel") as mock_openai:
                mock_openai.return_value = MagicMock()
                provider.create_model(
                    "openai",
                    "gpt-4o",
                    temperature=0.3,
                    max_tokens=2048,
                    top_p=0.5,
                )

                # Verify OpenAI model was called with correct model name
                mock_openai.assert_called_once()
                call_args = mock_openai.call_args
                # First positional arg should be model name
                assert call_args[0][0] == "gpt-4o"
                # Settings should be in kwargs
                assert "settings" in call_args[1]

    def test_create_model_with_default_parameters(self, mock_db_session):
        """Test create_model uses defaults when parameters not provided."""
        from pydantic import SecretStr

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.ai_provider.openai.api_key = SecretStr("test-key")
            provider = ModelProvider(db_session=mock_db_session)

            with patch("gearmeshing_ai.agent_core.model_provider.OpenAIResponsesModel") as mock_openai:
                mock_openai.return_value = MagicMock()
                provider.create_model("openai", "gpt-4o")

                # Verify OpenAI model was called with correct model name
                mock_openai.assert_called_once()
                call_args = mock_openai.call_args
                # First positional arg should be model name
                assert call_args[0][0] == "gpt-4o"
                # Settings should be in kwargs
                assert "settings" in call_args[1]

    def test_create_model_for_role_missing_config(self, mock_db_session):
        """Test create_model_for_role raises when config not found."""
        provider = ModelProvider(db_session=mock_db_session)

        with patch.object(provider, "_get_db_provider") as mock_get_db:
            mock_db_provider = MagicMock()
            mock_db_provider.get.side_effect = ValueError("Role not found")
            mock_get_db.return_value = mock_db_provider

            with pytest.raises(ValueError, match="Role not found"):
                provider.create_model_for_role("nonexistent-role")
