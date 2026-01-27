"""Tests for unified model provider with abstraction layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gearmeshing_ai.agent_core.model_provider import (
    UnifiedModelProvider,
    create_model_for_role,
    get_model_provider,
)


class TestUnifiedModelProvider:
    """Tests for UnifiedModelProvider class."""

    def test_unified_model_provider_initialization_requires_db_session(self) -> None:
        """Test UnifiedModelProvider initialization requires database session."""
        with pytest.raises(ValueError, match="db_session is required"):
            UnifiedModelProvider(db_session=None)  # type: ignore[arg-type]

    def test_unified_model_provider_initialization_with_db_session(self) -> None:
        """Test UnifiedModelProvider initialization with database session."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)
        assert provider.db_session is mock_session
        assert provider._db_provider is None

    def test_create_model_calls_abstraction_layer(self) -> None:
        """Test that create_model calls the abstraction layer."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        with patch.object(provider._provider, "create_model") as mock_create:
            mock_create.return_value = MagicMock()
            
            result = provider.create_model("openai", "gpt-4o", temperature=0.5, max_tokens=2048, top_p=0.8)
            
            mock_create.assert_called_once()
            assert result is not None

    def test_create_model_with_unsupported_provider(self) -> None:
        """Test create_model with unsupported provider."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        with pytest.raises(ValueError, match="Unsupported provider"):
            provider.create_model("unsupported", "model")

    def test_get_supported_providers(self) -> None:
        """Test getting supported providers."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        with patch.object(provider._provider, "get_supported_providers") as mock_supported:
            mock_supported.return_value = ["openai", "anthropic", "google"]
            
            result = provider.get_supported_providers()
            
            assert result == ["openai", "anthropic", "google"]
            mock_supported.assert_called_once()

    def test_get_supported_models(self) -> None:
        """Test getting supported models for a provider."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        with patch.object(provider._provider, "get_supported_models") as mock_models:
            mock_models.return_value = ["gpt-4o", "gpt-4-turbo"]
            
            result = provider.get_supported_models("openai")
            
            assert result == ["gpt-4o", "gpt-4-turbo"]
            mock_models.assert_called_once_with("openai")

    def test_create_fallback_model(self) -> None:
        """Test creating fallback model."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        with patch.object(provider._provider, "create_fallback_model") as mock_fallback:
            mock_fallback.return_value = MagicMock()
            
            result = provider.create_fallback_model(
                "openai", "gpt-4o", "anthropic", "claude-3-5-sonnet"
            )
            
            mock_fallback.assert_called_once()
            assert result is not None

    def test_get_provider_from_model_name(self) -> None:
        """Test getting provider from model name."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        with patch.object(provider._provider, "get_provider_from_model_name") as mock_get_provider:
            mock_get_provider.return_value = "openai"
            
            result = provider.get_provider_from_model_name("gpt-4o")
            
            assert result == "openai"
            mock_get_provider.assert_called_once_with("gpt-4o")

    def test_create_model_for_role(self) -> None:
        """Test creating model for role."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        with patch.object(provider, "create_model_for_role") as mock_create:
            mock_create.return_value = MagicMock()
            
            result = provider.create_model_for_role("dev", tenant_id="acme-corp")
            
            mock_create.assert_called_once_with("dev", tenant_id="acme-corp")
            assert result is not None


class TestModelProviderFunctions:
    """Tests for model provider convenience functions."""

    def test_get_model_provider(self) -> None:
        """Test get_model_provider function."""
        mock_session = MagicMock()
        
        with patch("gearmeshing_ai.agent_core.model_provider.UnifiedModelProvider") as mock_provider_class:
            mock_provider_class.return_value = MagicMock()
            
            result = get_model_provider(mock_session)
            
            mock_provider_class.assert_called_once_with(mock_session, "pydantic_ai")
            assert result is not None

    def test_create_model_for_role_function(self) -> None:
        """Test create_model_for_role function."""
        mock_session = MagicMock()
        
        with patch("gearmeshing_ai.agent_core.model_provider.get_model_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_get_provider.return_value = mock_provider
            mock_provider.create_model_for_role.return_value = MagicMock()
            
            result = create_model_for_role(mock_session, "dev", tenant_id="acme-corp")
            
            mock_get_provider.assert_called_once_with(mock_session, "pydantic_ai")
            mock_provider.create_model_for_role.assert_called_once_with("dev", "acme-corp")
            assert result is not None


class TestModelProviderFrameworkSelection:
    """Tests for framework selection in UnifiedModelProvider."""

    def test_default_framework_is_pydantic_ai(self) -> None:
        """Test that default framework is pydantic_ai."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)
        
        assert provider.framework == "pydantic_ai"

    def test_explicit_framework_selection(self) -> None:
        """Test explicit framework selection."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session, framework="pydantic_ai")
        
        assert provider.framework == "pydantic_ai"

    def test_unsupported_framework_raises_error(self) -> None:
        """Test that unsupported framework raises error."""
        mock_session = MagicMock()
        
        with pytest.raises(ValueError, match="Unsupported framework"):
            UnifiedModelProvider(db_session=mock_session, framework="unsupported_framework")


class TestModelProviderIntegration:
    """Integration tests for model provider with abstraction layer."""

    def test_abstraction_layer_integration(self) -> None:
        """Test that UnifiedModelProvider properly integrates with abstraction layer."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)
        
        # Verify that the provider is initialized
        assert provider._provider is not None
        assert hasattr(provider._provider, "create_model")
        assert hasattr(provider._provider, "get_supported_providers")

    def test_model_config_creation(self) -> None:
        """Test that ModelConfig is created correctly."""
        from gearmeshing_ai.agent_core.abstraction import ModelConfig
        
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)
        
        with patch.object(provider._provider, "create_model") as mock_create:
            mock_create.return_value = MagicMock()
            
            # This should create a ModelConfig internally
            provider.create_model("openai", "gpt-4o", temperature=0.7, max_tokens=2048)
            
            # Verify the abstraction layer was called
            mock_create.assert_called_once()
            
            # Check the arguments passed to the abstraction layer
            call_args = mock_create.call_args[0][0]  # First positional argument
            assert isinstance(call_args, ModelConfig)
            assert call_args.provider == "openai"
            assert call_args.model == "gpt-4o"
            assert call_args.temperature == 0.7
            assert call_args.max_tokens == 2048
