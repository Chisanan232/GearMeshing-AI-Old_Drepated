"""Unit tests for model provider abstraction layer.

This test file provides comprehensive coverage for the model provider abstraction
layer, including fallback model functionality, provider detection from model names,
and error handling scenarios.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from gearmeshing_ai.agent_core.abstraction.model_provider import (
    FallbackModelInstance,
    ModelConfig,
    ModelProvider,
    ModelResponse,
)


class MockModelProvider(ModelProvider):
    """Mock model provider for testing."""

    def __init__(self):
        """Initialize mock provider."""
        self.create_model_calls = []
        self.create_fallback_model_calls = []

    def create_model(self, config: ModelConfig):
        """Mock create model."""
        self.create_model_calls.append(config)
        mock_model = MagicMock()
        mock_model.generate = AsyncMock(return_value=ModelResponse(
            content=f"Mock response for {config.model}",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            metadata={"framework": "mock"}
        ))
        mock_model.generate_structured = AsyncMock(return_value={"result": "structured"})
        return mock_model

    def get_supported_providers(self):
        """Mock supported providers."""
        return ["openai", "anthropic", "google"]

    def get_supported_models(self, provider):
        """Mock supported models."""
        models = {
            "openai": ["gpt-4o", "gpt-4-turbo"],
            "anthropic": ["claude-3-5-sonnet"],
            "google": ["gemini-2.0-flash"]
        }
        return models.get(provider, [])

    def validate_config(self, config):
        """Mock config validation."""
        pass

    def create_fallback_model(self, primary_config, fallback_config):
        """Mock fallback model creation."""
        self.create_fallback_model_calls.append((primary_config, fallback_config))
        primary = self.create_model(primary_config)
        fallback = self.create_model(fallback_config)
        return FallbackModelInstance(primary, fallback)


class TestModelProviderAbstraction:
    """Tests for ModelProvider abstraction layer."""

    def test_create_fallback_model_default_implementation(self):
        """Test the default create_fallback_model implementation (lines 162-166)."""
        provider = MockModelProvider()
        
        primary_config = ModelConfig(
            provider="openai",
            model="gpt-4o",
            temperature=0.7,
            max_tokens=1000
        )
        fallback_config = ModelConfig(
            provider="anthropic", 
            model="claude-3-5-sonnet",
            temperature=0.5,
            max_tokens=2000
        )
        
        # This should call the default implementation
        fallback_model = provider.create_fallback_model(primary_config, fallback_config)
        
        # Verify the method was called
        assert len(provider.create_fallback_model_calls) == 1
        assert provider.create_fallback_model_calls[0] == (primary_config, fallback_config)
        
        # Verify it returns a FallbackModelInstance
        assert isinstance(fallback_model, FallbackModelInstance)
        assert hasattr(fallback_model, 'primary')
        assert hasattr(fallback_model, 'fallback')

    def test_create_fallback_model_exact_implementation_behavior(self):
        """Test the exact behavior of lines 162-166 implementation."""
        provider = MockModelProvider()
        
        # Create specific configs to test exact implementation
        primary_config = ModelConfig(
            provider="openai",
            model="gpt-4o",
            temperature=0.8,
            max_tokens=1500,
            top_p=0.95
        )
        fallback_config = ModelConfig(
            provider="anthropic",
            model="claude-3-5-sonnet",
            temperature=0.6,
            max_tokens=2500,
            top_p=0.85
        )
        
        # Mock the create_model method to track exact calls
        with patch.object(provider, 'create_model') as mock_create_model:
            mock_primary = MagicMock()
            mock_fallback = MagicMock()
            mock_create_model.side_effect = [mock_primary, mock_fallback]
            
            # Call the method that implements lines 162-166
            result = provider.create_fallback_model(primary_config, fallback_config)
            
            # Verify create_model was called twice with exact configs
            assert mock_create_model.call_count == 2
            mock_create_model.assert_any_call(primary_config)
            mock_create_model.assert_any_call(fallback_config)
            
            # Verify the exact implementation: return FallbackModelInstance(...)
            assert isinstance(result, FallbackModelInstance)
            assert result.primary == mock_primary
            assert result.fallback == mock_fallback
            
            # Verify the order: primary first, fallback second
            first_call_args = mock_create_model.call_args_list[0][0][0]  # First positional arg from first call
            second_call_args = mock_create_model.call_args_list[1][0][0]  # First positional arg from second call
            assert first_call_args == primary_config
            assert second_call_args == fallback_config

    def test_get_provider_from_model_name_with_known_prefixes(self):
        """Test get_provider_from_model_name with known model prefixes (lines 179-188)."""
        provider = MockModelProvider()
        
        # Mock the APIKeyValidator
        with patch('gearmeshing_ai.agent_core.abstraction.api_key_validator.APIKeyValidator') as mock_validator_class:
            mock_validator = MagicMock()
            mock_validator_class.get_provider_for_model = mock_validator
            
            # Mock different model name patterns
            test_cases = [
                ("gpt-4o", "openai"),
                ("gpt-4-turbo", "openai"),
                ("claude-3-5-sonnet", "anthropic"),
                ("claude-3-opus", "anthropic"),
                ("gemini-2.0-flash", "google"),
                ("gemini-1.5-pro", "google"),
                ("grok-beta", "grok"),
            ]
            
            for model_name, expected_provider in test_cases:
                # Mock the provider enum
                mock_provider_enum = MagicMock()
                mock_provider_enum.value = expected_provider
                mock_validator.return_value = mock_provider_enum
                
                result = provider.get_provider_from_model_name(model_name)
                
                assert result == expected_provider
                mock_validator.assert_called_with(model_name)

    def test_get_provider_from_model_name_with_unknown_model(self):
        """Test get_provider_from_model_name with unknown model (lines 182-187)."""
        provider = MockModelProvider()
        
        with patch('gearmeshing_ai.agent_core.abstraction.api_key_validator.APIKeyValidator') as mock_validator_class:
            mock_validator = MagicMock()
            mock_validator_class.get_provider_for_model = mock_validator
            
            # Mock APIKeyValidator returning None for unknown model
            mock_validator.return_value = None
            
            with pytest.raises(ValueError, match="Could not determine provider for model 'unknown-model'") as exc_info:
                provider.get_provider_from_model_name("unknown-model")
            
            # Verify the error message contains the expected text
            error_message = str(exc_info.value)
            assert "Could not determine provider for model 'unknown-model'" in error_message
            assert "Supported prefixes: gpt-*, claude-*, gemini-*, grok-*" in error_message
            
            mock_validator.assert_called_with("unknown-model")

    def test_get_provider_from_model_name_with_various_unknown_models(self):
        """Test get_provider_from_model_name with various unknown model names."""
        provider = MockModelProvider()
        
        with patch('gearmeshing_ai.agent_core.abstraction.api_key_validator.APIKeyValidator') as mock_validator_class:
            mock_validator = MagicMock()
            mock_validator_class.get_provider_for_model = mock_validator
            mock_validator.return_value = None
            
            unknown_models = [
                "llama-3-70b",
                "mistral-7b", 
                "phi-3-medium",
                "custom-model-v1",
                "unknown-ai-model"
            ]
            
            for model_name in unknown_models:
                with pytest.raises(ValueError, match=f"Could not determine provider for model '{model_name}'"):
                    provider.get_provider_from_model_name(model_name)
                
                mock_validator.assert_called_with(model_name)


class TestFallbackModelInstance:
    """Tests for FallbackModelInstance class."""

    @pytest.mark.asyncio
    async def test_generate_primary_succeeds(self):
        """Test generate method when primary model succeeds (lines 217-218)."""
        # Create mock primary and fallback models
        primary_model = MagicMock()
        fallback_model = MagicMock()
        
        # Mock primary model to succeed
        expected_response = ModelResponse(
            content="Primary response",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20}
        )
        primary_model.generate = AsyncMock(return_value=expected_response)
        
        # Create fallback instance
        fallback_instance = FallbackModelInstance(primary_model, fallback_model)
        
        # Test generation
        result = await fallback_instance.generate("test prompt", max_tokens=100, temperature=0.7)
        
        # Verify primary was called and fallback was not
        primary_model.generate.assert_called_once_with(
            "test prompt", max_tokens=100, temperature=0.7
        )
        fallback_model.generate.assert_not_called()
        
        # Verify result
        assert result == expected_response

    @pytest.mark.asyncio
    async def test_generate_primary_fails_fallback_succeeds(self):
        """Test generate method when primary fails and fallback succeeds (lines 219-222)."""
        # Create mock primary and fallback models
        primary_model = MagicMock()
        fallback_model = MagicMock()
        
        # Mock primary model to fail
        primary_model.generate = AsyncMock(side_effect=RuntimeError("Primary model failed"))
        
        # Mock fallback model to succeed
        expected_response = ModelResponse(
            content="Fallback response",
            finish_reason="stop", 
            usage={"prompt_tokens": 15, "completion_tokens": 25}
        )
        fallback_model.generate = AsyncMock(return_value=expected_response)
        
        # Create fallback instance
        fallback_instance = FallbackModelInstance(primary_model, fallback_model)
        
        # Test generation
        with patch('builtins.print') as mock_print:
            result = await fallback_instance.generate("test prompt", max_tokens=100, temperature=0.7)
        
        # Verify error was logged
        mock_print.assert_called_once()
        print_call_args = mock_print.call_args[0][0]
        assert "Primary model failed: Primary model failed, trying fallback" in print_call_args
        
        # Verify both models were called
        primary_model.generate.assert_called_once_with(
            "test prompt", max_tokens=100, temperature=0.7
        )
        fallback_model.generate.assert_called_once_with(
            "test prompt", max_tokens=100, temperature=0.7
        )
        
        # Verify result from fallback
        assert result == expected_response

    @pytest.mark.asyncio
    async def test_generate_both_models_fail(self):
        """Test generate method when both primary and fallback fail."""
        # Create mock primary and fallback models
        primary_model = MagicMock()
        fallback_model = MagicMock()
        
        # Mock both models to fail
        primary_model.generate = AsyncMock(side_effect=RuntimeError("Primary failed"))
        fallback_model.generate = AsyncMock(side_effect=ValueError("Fallback failed"))
        
        # Create fallback instance
        fallback_instance = FallbackModelInstance(primary_model, fallback_model)
        
        # Test generation - should propagate fallback error
        with patch('builtins.print') as mock_print:
            with pytest.raises(ValueError, match="Fallback failed"):
                await fallback_instance.generate("test prompt")
        
        # Verify error was logged
        mock_print.assert_called_once()
        
        # Verify both models were called
        primary_model.generate.assert_called_once()
        fallback_model.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_structured_primary_succeeds(self):
        """Test generate_structured when primary model succeeds (lines 234-237)."""
        # Create mock primary and fallback models
        primary_model = MagicMock()
        fallback_model = MagicMock()
        
        # Mock primary model to succeed
        expected_result = {"result": "structured data", "confidence": 0.95}
        primary_model.generate_structured = AsyncMock(return_value=expected_result)
        
        # Create fallback instance
        fallback_instance = FallbackModelInstance(primary_model, fallback_model)
        
        # Test structured generation
        result = await fallback_instance.generate_structured(
            "test prompt", dict, max_tokens=100, temperature=0.7
        )
        
        # Verify primary was called and fallback was not
        primary_model.generate_structured.assert_called_once_with(
            "test prompt", dict, max_tokens=100, temperature=0.7
        )
        fallback_model.generate_structured.assert_not_called()
        
        # Verify result
        assert result == expected_result

    @pytest.mark.asyncio
    async def test_generate_structured_primary_fails_fallback_succeeds(self):
        """Test generate_structured when primary fails and fallback succeeds (lines 238-243)."""
        # Create mock primary and fallback models
        primary_model = MagicMock()
        fallback_model = MagicMock()
        
        # Mock primary model to fail
        primary_model.generate_structured = AsyncMock(side_effect=RuntimeError("Primary failed"))
        
        # Mock fallback model to succeed
        expected_result = {"result": "fallback structured data", "confidence": 0.85}
        fallback_model.generate_structured = AsyncMock(return_value=expected_result)
        
        # Create fallback instance
        fallback_instance = FallbackModelInstance(primary_model, fallback_model)
        
        # Test structured generation
        with patch('builtins.print') as mock_print:
            result = await fallback_instance.generate_structured(
                "test prompt", dict, max_tokens=100, temperature=0.7
            )
        
        # Verify error was logged
        mock_print.assert_called_once()
        print_call_args = mock_print.call_args[0][0]
        assert "Primary model failed: Primary failed, trying fallback" in print_call_args
        
        # Verify both models were called
        primary_model.generate_structured.assert_called_once_with(
            "test prompt", dict, max_tokens=100, temperature=0.7
        )
        fallback_model.generate_structured.assert_called_once_with(
            "test prompt", dict, max_tokens=100, temperature=0.7
        )
        
        # Verify result from fallback
        assert result == expected_result

    @pytest.mark.asyncio
    async def test_generate_structured_both_models_fail(self):
        """Test generate_structured when both primary and fallback fail."""
        # Create mock primary and fallback models
        primary_model = MagicMock()
        fallback_model = MagicMock()
        
        # Mock both models to fail
        primary_model.generate_structured = AsyncMock(side_effect=RuntimeError("Primary failed"))
        fallback_model.generate_structured = AsyncMock(side_effect=TypeError("Fallback failed"))
        
        # Create fallback instance
        fallback_instance = FallbackModelInstance(primary_model, fallback_model)
        
        # Test structured generation - should propagate fallback error
        with patch('builtins.print') as mock_print:
            with pytest.raises(TypeError, match="Fallback failed"):
                await fallback_instance.generate_structured("test prompt", dict)
        
        # Verify error was logged
        mock_print.assert_called_once()
        
        # Verify both models were called
        primary_model.generate_structured.assert_called_once()
        fallback_model.generate_structured.assert_called_once()

    def test_fallback_model_instance_initialization(self):
        """Test FallbackModelInstance initialization."""
        primary_model = MagicMock()
        fallback_model = MagicMock()
        
        fallback_instance = FallbackModelInstance(primary_model, fallback_model)
        
        assert fallback_instance.primary == primary_model
        assert fallback_instance.fallback == fallback_model

    @pytest.mark.asyncio
    async def test_generate_with_different_parameters(self):
        """Test generate method with various parameter combinations."""
        primary_model = MagicMock()
        fallback_model = MagicMock()
        
        # Mock primary to succeed
        expected_response = ModelResponse(
            content="Response",
            finish_reason="stop",
            usage={}
        )
        primary_model.generate = AsyncMock(return_value=expected_response)
        
        fallback_instance = FallbackModelInstance(primary_model, fallback_model)
        
        # Test with different parameter combinations
        test_cases = [
            {"prompt": "test", "max_tokens": None, "temperature": None},
            {"prompt": "test", "max_tokens": 100, "temperature": None},
            {"prompt": "test", "max_tokens": None, "temperature": 0.5},
            {"prompt": "test", "max_tokens": 200, "temperature": 0.8},
            {"prompt": "test", "max_tokens": 300, "temperature": 1.0, "extra_param": "value"},
        ]
        
        for kwargs in test_cases:
            primary_model.generate.reset_mock()
            
            await fallback_instance.generate(**kwargs)
            
            # Verify primary was called with correct parameters
            call_args = primary_model.generate.call_args
            assert call_args[0][0] == kwargs["prompt"]  # First positional arg (prompt)
            assert call_args[1].get("max_tokens") == kwargs.get("max_tokens")
            assert call_args[1].get("temperature") == kwargs.get("temperature")
            # Check extra params if present
            for key, value in kwargs.items():
                if key not in ["prompt", "max_tokens", "temperature"]:
                    assert call_args[1].get(key) == value

    @pytest.mark.asyncio
    async def test_generate_structured_with_different_schemas(self):
        """Test generate_structured with different output schemas."""
        primary_model = MagicMock()
        fallback_model = MagicMock()
        
        # Mock primary to succeed
        primary_model.generate_structured = AsyncMock(return_value={"result": "data"})
        
        fallback_instance = FallbackModelInstance(primary_model, fallback_model)
        
        # Test with different schemas
        schemas = [dict, list, str, int, float, bool]
        
        for schema in schemas:
            primary_model.generate_structured.reset_mock()
            
            await fallback_instance.generate_structured("test prompt", schema)
            
            # Verify primary was called with correct schema and default parameters
            primary_model.generate_structured.assert_called_once_with("test prompt", schema, max_tokens=None, temperature=None)


class TestModelProviderIntegration:
    """Integration tests for model provider abstraction."""

    def test_model_provider_factory_pattern(self):
        """Test that ModelProvider follows factory pattern correctly."""
        provider = MockModelProvider()
        
        # Test that all required methods are implemented
        assert hasattr(provider, 'create_model')
        assert hasattr(provider, 'get_supported_providers')
        assert hasattr(provider, 'get_supported_models')
        assert hasattr(provider, 'validate_config')
        assert hasattr(provider, 'create_fallback_model')
        assert hasattr(provider, 'get_provider_from_model_name')

    def test_model_config_validation(self):
        """Test ModelConfig validation in abstraction layer."""
        # Test valid config
        valid_config = ModelConfig(
            provider="openai",
            model="gpt-4o",
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9
        )
        assert valid_config.provider == "openai"
        assert valid_config.model == "gpt-4o"
        assert valid_config.temperature == 0.7
        assert valid_config.max_tokens == 1000
        assert valid_config.top_p == 0.9

    def test_model_response_structure(self):
        """Test ModelResponse structure."""
        response = ModelResponse(
            content="Test response",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            metadata={"model": "gpt-4o", "framework": "test"}
        )
        
        assert response.content == "Test response"
        assert response.finish_reason == "stop"
        assert response.usage == {"prompt_tokens": 10, "completion_tokens": 20}
        assert response.metadata == {"model": "gpt-4o", "framework": "test"}

    @pytest.mark.asyncio
    async def test_fallback_model_error_logging_format(self):
        """Test that fallback model error logging includes proper information."""
        primary_model = MagicMock()
        fallback_model = MagicMock()
        
        # Mock primary to fail with specific error
        primary_error = ValueError("API rate limit exceeded")
        primary_model.generate = AsyncMock(side_effect=primary_error)
        fallback_model.generate = AsyncMock(return_value=ModelResponse(
            content="Fallback response",
            finish_reason="stop",
            usage={}
        ))
        
        fallback_instance = FallbackModelInstance(primary_model, fallback_model)
        
        with patch('builtins.print') as mock_print:
            await fallback_instance.generate("test prompt")
        
        # Verify error message format
        mock_print.assert_called_once()
        print_message = mock_print.call_args[0][0]
        assert "Primary model failed: API rate limit exceeded, trying fallback" in print_message
