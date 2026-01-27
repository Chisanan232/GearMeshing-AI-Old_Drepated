"""Integration tests for model provider with engine and planner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gearmeshing_ai.agent_core.model_provider import (
    UnifiedModelProvider,
    create_model_for_role,
    get_model_provider,
)
from gearmeshing_ai.agent_core.schemas.config import ModelConfig


class TestModelProviderIntegration:
    """Integration tests for UnifiedModelProvider with database configuration."""

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

    def test_get_model_provider_returns_unified_provider(self, mock_db_session):
        """Test get_model_provider factory function returns UnifiedModelProvider."""
        provider = get_model_provider(mock_db_session)
        assert isinstance(provider, UnifiedModelProvider)
        assert provider.db_session is mock_db_session

    def test_unified_model_provider_initialization(self, mock_db_session):
        """Test UnifiedModelProvider initialization with database session."""
        provider = UnifiedModelProvider(db_session=mock_db_session)
        assert provider.db_session is mock_db_session
        assert provider._db_provider is None
        assert provider.framework == "pydantic_ai"

    def test_create_model_integration_with_abstraction(self, mock_db_session):
        """Test that create_model properly integrates with abstraction layer."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

        with patch.object(provider._provider, "create_model") as mock_create:
            mock_create.return_value = MagicMock()
            
            result = provider.create_model("openai", "gpt-4o", temperature=0.5)
            
            mock_create.assert_called_once()
            assert result is not None

    def test_create_model_with_explicit_framework(self, mock_db_session):
        """Test UnifiedModelProvider with explicit framework selection."""
        provider = UnifiedModelProvider(db_session=mock_db_session, framework="pydantic_ai")
        
        with patch.object(provider._provider, "create_model") as mock_create:
            mock_create.return_value = MagicMock()
            
            result = provider.create_model("openai", "gpt-4o")
            
            mock_create.assert_called_once()
            assert result is not None

    def test_create_model_handles_provider_errors(self, mock_db_session):
        """Test that create_model properly handles abstraction layer errors."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

        with patch.object(provider._provider, "create_model") as mock_create:
            mock_create.side_effect = RuntimeError("Provider error")
            
            with pytest.raises(RuntimeError, match="Provider error"):
                provider.create_model("openai", "gpt-4o")

    def test_create_model_for_role_integration(self, mock_db_session, mock_model_config):
        """Test create_model_for_role with database configuration."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

        with patch.object(provider, "create_model_for_role") as mock_create_role:
            mock_create_role.return_value = MagicMock()
            
            result = provider.create_model_for_role("dev", tenant_id="acme-corp")
            
            mock_create_role.assert_called_once_with("dev", tenant_id="acme-corp")
            assert result is not None

    def test_get_supported_providers_integration(self, mock_db_session):
        """Test getting supported providers from abstraction layer."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

        with patch.object(provider._provider, "get_supported_providers") as mock_supported:
            mock_supported.return_value = ["openai", "anthropic", "google"]
            
            result = provider.get_supported_providers()
            
            assert result == ["openai", "anthropic", "google"]
            mock_supported.assert_called_once()

    def test_get_supported_models_integration(self, mock_db_session):
        """Test getting supported models from abstraction layer."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

        with patch.object(provider._provider, "get_supported_models") as mock_models:
            mock_models.return_value = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
            
            result = provider.get_supported_models("openai")
            
            assert result == ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
            mock_models.assert_called_once_with("openai")

    def test_create_fallback_model_integration(self, mock_db_session):
        """Test creating fallback model through abstraction layer."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

        with patch.object(provider._provider, "create_fallback_model") as mock_fallback:
            mock_fallback.return_value = MagicMock()
            
            result = provider.create_fallback_model(
                "openai", "gpt-4o", "anthropic", "claude-3-5-sonnet"
            )
            
            mock_fallback.assert_called_once()
            assert result is not None

    def test_get_provider_from_model_name_integration(self, mock_db_session):
        """Test getting provider from model name through abstraction layer."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

        with patch.object(provider._provider, "get_provider_from_model_name") as mock_get_provider:
            mock_get_provider.return_value = "openai"
            
            result = provider.get_provider_from_model_name("gpt-4o")
            
            assert result == "openai"
            mock_get_provider.assert_called_once_with("gpt-4o")

    def test_model_provider_factory_function_integration(self, mock_db_session):
        """Test get_model_provider factory function integration."""
        with patch("gearmeshing_ai.agent_core.model_provider.UnifiedModelProvider") as mock_provider_class:
            mock_provider_class.return_value = MagicMock()
            
            result = get_model_provider(mock_db_session)
            
            mock_provider_class.assert_called_once_with(mock_db_session, "pydantic_ai")
            assert result is not None

    def test_create_model_for_role_function_integration(self, mock_db_session):
        """Test create_model_for_role function integration."""
        with patch("gearmeshing_ai.agent_core.model_provider.get_model_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_get_provider.return_value = mock_provider
            mock_provider.create_model_for_role.return_value = MagicMock()
            
            result = create_model_for_role(mock_db_session, "dev", tenant_id="acme-corp")
            
            mock_get_provider.assert_called_once_with(mock_db_session, "pydantic_ai")
            mock_provider.create_model_for_role.assert_called_once_with("dev", "acme-corp")
            assert result is not None

    def test_abstraction_layer_error_propagation(self, mock_db_session):
        """Test that errors from abstraction layer are properly propagated."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

        # Test ValueError from abstraction layer
        with patch.object(provider._provider, "create_model") as mock_create:
            mock_create.side_effect = ValueError("Invalid provider")
            
            with pytest.raises(ValueError, match="Invalid provider"):
                provider.create_model("invalid", "model")

    def test_framework_selection_integration(self, mock_db_session):
        """Test framework selection in integration context."""
        # Test default framework
        provider_default = UnifiedModelProvider(db_session=mock_db_session)
        assert provider_default.framework == "pydantic_ai"
        
        # Test explicit framework
        provider_explicit = UnifiedModelProvider(db_session=mock_db_session, framework="pydantic_ai")
        assert provider_explicit.framework == "pydantic_ai"

    def test_unsupported_framework_integration(self, mock_db_session):
        """Test unsupported framework handling in integration context."""
        with pytest.raises(ValueError, match="Unsupported framework"):
            UnifiedModelProvider(db_session=mock_db_session, framework="unsupported_framework")

    def test_model_config_parameter_passing(self, mock_db_session):
        """Test that ModelConfig parameters are correctly passed to abstraction layer."""
        from gearmeshing_ai.agent_core.abstraction import ModelConfig
        
        provider = UnifiedModelProvider(db_session=mock_db_session)
        
        with patch.object(provider._provider, "create_model") as mock_create:
            mock_create.return_value = MagicMock()
            
            # Create model with all parameters
            provider.create_model(
                "openai", 
                "gpt-4o", 
                temperature=0.7, 
                max_tokens=2048, 
                top_p=0.9
            )
            
            # Verify ModelConfig was created correctly
            mock_create.assert_called_once()
            call_args = mock_create.call_args[0][0]
            assert isinstance(call_args, ModelConfig)
            assert call_args.provider == "openai"
            assert call_args.model == "gpt-4o"
            assert call_args.temperature == 0.7
            assert call_args.max_tokens == 2048
            assert call_args.top_p == 0.9

    def test_provider_lazy_initialization(self, mock_db_session):
        """Test that abstraction layer provider is properly initialized."""
        provider = UnifiedModelProvider(db_session=mock_db_session)
        
        # Provider should be initialized on construction
        assert provider._provider is not None
        assert hasattr(provider._provider, "create_model")
        assert hasattr(provider._provider, "get_supported_providers")
        assert hasattr(provider._provider, "get_supported_models")
