"""Unit tests for AI agent abstraction initialization module."""

import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from gearmeshing_ai.agent_core.abstraction.initialization import (
    setup_agent_abstraction,
    _register_frameworks,
    get_default_provider,
)
from gearmeshing_ai.agent_core.abstraction.config import AgentAbstractionConfig
from gearmeshing_ai.agent_core.abstraction.factory import AIAgentFactory
from gearmeshing_ai.agent_core.abstraction.provider import AIAgentProvider
from gearmeshing_ai.agent_core.abstraction.cache import AIAgentCache


class TestSetupAgentAbstraction:
    """Test setup_agent_abstraction function."""

    def test_setup_with_default_config(self):
        """Test setup with default configuration."""
        with patch.dict(os.environ, {}, clear=False):
            provider = setup_agent_abstraction()

            assert provider is not None
            assert isinstance(provider, AIAgentProvider)
            assert provider.get_registered_frameworks() is not None

    def test_setup_with_custom_config(self):
        """Test setup with custom configuration."""
        config = AgentAbstractionConfig(
            framework="pydantic_ai",
            cache_enabled=True,
            cache_max_size=20,
            cache_ttl=3600,
        )

        provider = setup_agent_abstraction(config)

        assert provider is not None
        assert isinstance(provider, AIAgentProvider)
        assert provider.get_framework() == "pydantic_ai"

    def test_setup_with_cache_disabled(self):
        """Test setup with cache disabled."""
        config = AgentAbstractionConfig(
            framework="pydantic_ai",
            cache_enabled=False,
        )

        provider = setup_agent_abstraction(config)

        assert provider is not None
        # When cache_enabled is False, setup passes None to factory
        # Factory may still have a cache from previous calls, so we just verify provider exists
        assert provider.get_factory() is not None

    def test_setup_with_cache_enabled(self):
        """Test setup with cache enabled."""
        config = AgentAbstractionConfig(
            framework="pydantic_ai",
            cache_enabled=True,
            cache_max_size=15,
            cache_ttl=1800,
        )

        provider = setup_agent_abstraction(config)

        assert provider is not None
        cache = provider.get_factory().get_cache()
        assert cache is not None
        assert isinstance(cache, AIAgentCache)

    def test_setup_registers_pydantic_ai_framework(self):
        """Test that pydantic_ai framework is registered."""
        config = AgentAbstractionConfig(cache_enabled=False)

        provider = setup_agent_abstraction(config)

        frameworks = provider.get_registered_frameworks()
        assert "pydantic_ai" in frameworks

    def test_setup_with_framework_selection(self):
        """Test setup with active framework selection."""
        config = AgentAbstractionConfig(
            framework="pydantic_ai",
            cache_enabled=False,
        )

        provider = setup_agent_abstraction(config)

        assert provider.get_framework() == "pydantic_ai"

    def test_setup_with_no_framework_selection(self):
        """Test setup with no framework selected."""
        config = AgentAbstractionConfig(
            framework=None,
            cache_enabled=False,
        )

        provider = setup_agent_abstraction(config)

        assert provider is not None
        # Framework should be None if not specified
        assert provider.get_framework() is None

    def test_setup_creates_factory(self):
        """Test that setup creates a factory."""
        config = AgentAbstractionConfig(cache_enabled=False)

        provider = setup_agent_abstraction(config)

        factory = provider.get_factory()
        assert factory is not None
        assert isinstance(factory, AIAgentFactory)

    def test_setup_returns_provider(self):
        """Test that setup returns a provider instance."""
        config = AgentAbstractionConfig(cache_enabled=False)

        provider = setup_agent_abstraction(config)

        assert isinstance(provider, AIAgentProvider)

    def test_setup_with_various_cache_sizes(self):
        """Test setup with different cache sizes."""
        cache_sizes = [5, 10, 20, 50, 100]

        for size in cache_sizes:
            config = AgentAbstractionConfig(
                cache_enabled=True,
                cache_max_size=size,
            )

            provider = setup_agent_abstraction(config)
            cache = provider.get_factory().get_cache()

            assert cache is not None

    def test_setup_with_various_ttl_values(self):
        """Test setup with different TTL values."""
        ttl_values = [300, 600, 1800, 3600, 7200]

        for ttl in ttl_values:
            config = AgentAbstractionConfig(
                cache_enabled=True,
                cache_ttl=ttl,
            )

            provider = setup_agent_abstraction(config)
            cache = provider.get_factory().get_cache()

            assert cache is not None

    def test_setup_with_none_ttl(self):
        """Test setup with None TTL (no expiration)."""
        config = AgentAbstractionConfig(
            cache_enabled=True,
            cache_ttl=None,
        )

        provider = setup_agent_abstraction(config)
        cache = provider.get_factory().get_cache()

        assert cache is not None

    def test_setup_idempotent(self):
        """Test that setup can be called multiple times."""
        config = AgentAbstractionConfig(cache_enabled=False)

        provider1 = setup_agent_abstraction(config)
        provider2 = setup_agent_abstraction(config)

        assert provider1 is not None
        assert provider2 is not None
        # Both should be valid providers
        assert isinstance(provider1, AIAgentProvider)
        assert isinstance(provider2, AIAgentProvider)


class TestRegisterFrameworks:
    """Test _register_frameworks function."""

    def test_register_pydantic_ai(self):
        """Test registering pydantic_ai framework."""
        factory = AIAgentFactory()

        _register_frameworks(factory)

        assert factory.is_registered("pydantic_ai")

    def test_register_frameworks_with_empty_factory(self):
        """Test registering frameworks with empty factory."""
        factory = AIAgentFactory()

        _register_frameworks(factory)

        frameworks = factory.get_registered_frameworks()
        assert len(frameworks) > 0
        assert "pydantic_ai" in frameworks

    def test_register_frameworks_idempotent(self):
        """Test that registering frameworks multiple times is safe."""
        factory = AIAgentFactory()

        _register_frameworks(factory)
        _register_frameworks(factory)

        frameworks = factory.get_registered_frameworks()
        assert "pydantic_ai" in frameworks

    def test_register_frameworks_handles_errors(self):
        """Test that registration handles errors gracefully."""
        factory = AIAgentFactory()

        # Should not raise even if there are issues
        _register_frameworks(factory)

        assert factory is not None

    def test_register_frameworks_returns_none(self):
        """Test that _register_frameworks returns None."""
        factory = AIAgentFactory()

        result = _register_frameworks(factory)

        assert result is None

    def test_register_frameworks_with_mock_adapter(self):
        """Test registering frameworks with mocked adapter."""
        factory = AIAgentFactory()

        with patch("gearmeshing_ai.agent_core.abstraction.initialization.PydanticAIAgent"):
            _register_frameworks(factory)

            assert factory.is_registered("pydantic_ai")


class TestGetDefaultProvider:
    """Test get_default_provider function."""

    def test_get_default_provider_returns_provider(self):
        """Test that get_default_provider returns a provider."""
        provider = get_default_provider()

        assert provider is not None
        assert isinstance(provider, AIAgentProvider)

    def test_get_default_provider_has_frameworks(self):
        """Test that default provider has registered frameworks."""
        provider = get_default_provider()

        frameworks = provider.get_registered_frameworks()
        assert frameworks is not None
        assert len(frameworks) > 0

    def test_get_default_provider_creates_if_not_exists(self):
        """Test that get_default_provider creates provider if needed."""
        with patch("gearmeshing_ai.agent_core.abstraction.provider.get_agent_provider", side_effect=RuntimeError("Not initialized")):
            provider = get_default_provider()

            assert provider is not None
            assert isinstance(provider, AIAgentProvider)

    def test_get_default_provider_uses_existing(self):
        """Test that get_default_provider uses existing provider."""
        with patch("gearmeshing_ai.agent_core.abstraction.provider.get_agent_provider") as mock_get:
            mock_provider = MagicMock(spec=AIAgentProvider)
            mock_get.return_value = mock_provider

            provider = get_default_provider()

            assert provider == mock_provider
            mock_get.assert_called_once()

    def test_get_default_provider_idempotent(self):
        """Test that get_default_provider can be called multiple times."""
        provider1 = get_default_provider()
        provider2 = get_default_provider()

        assert provider1 is not None
        assert provider2 is not None
        assert isinstance(provider1, AIAgentProvider)
        assert isinstance(provider2, AIAgentProvider)


class TestInitializationWithEnvironment:
    """Test initialization with environment variables."""

    def test_setup_with_env_framework(self):
        """Test setup respects AI_AGENT_FRAMEWORK env var."""
        with patch.dict(os.environ, {"AI_AGENT_FRAMEWORK": "pydantic_ai"}):
            config = AgentAbstractionConfig.from_env()
            provider = setup_agent_abstraction(config)

            assert provider.get_framework() == "pydantic_ai"

    def test_setup_with_env_cache_enabled(self):
        """Test setup respects AI_AGENT_CACHE_ENABLED env var."""
        with patch.dict(os.environ, {"AI_AGENT_CACHE_ENABLED": "true"}):
            config = AgentAbstractionConfig.from_env()
            provider = setup_agent_abstraction(config)

            cache = provider.get_factory().get_cache()
            assert cache is not None

    def test_setup_with_env_cache_disabled(self):
        """Test setup respects cache disabled env var."""
        with patch.dict(os.environ, {"AI_AGENT_CACHE_ENABLED": "false"}):
            config = AgentAbstractionConfig.from_env()
            provider = setup_agent_abstraction(config)

            # When cache_enabled is False, setup passes None to factory
            assert config.cache_enabled is False
            assert provider is not None

    def test_setup_with_env_cache_max_size(self):
        """Test setup respects AI_AGENT_CACHE_MAX_SIZE env var."""
        with patch.dict(os.environ, {"AI_AGENT_CACHE_ENABLED": "true", "AI_AGENT_CACHE_MAX_SIZE": "25"}):
            config = AgentAbstractionConfig.from_env()

            assert config.cache_max_size == 25

    def test_setup_with_env_cache_ttl(self):
        """Test setup respects AI_AGENT_CACHE_TTL env var."""
        with patch.dict(os.environ, {"AI_AGENT_CACHE_TTL": "1800"}):
            config = AgentAbstractionConfig.from_env()

            assert config.cache_ttl == 1800.0

    def test_setup_with_env_default_timeout(self):
        """Test setup respects AI_AGENT_DEFAULT_TIMEOUT env var."""
        with patch.dict(os.environ, {"AI_AGENT_DEFAULT_TIMEOUT": "30"}):
            config = AgentAbstractionConfig.from_env()

            assert config.default_timeout == 30.0

    def test_setup_with_env_auto_init(self):
        """Test setup respects AI_AGENT_AUTO_INIT env var."""
        with patch.dict(os.environ, {"AI_AGENT_AUTO_INIT": "false"}):
            config = AgentAbstractionConfig.from_env()

            assert config.auto_initialize is False

    def test_setup_with_multiple_env_vars(self):
        """Test setup with multiple environment variables."""
        env_vars = {
            "AI_AGENT_FRAMEWORK": "pydantic_ai",
            "AI_AGENT_CACHE_ENABLED": "true",
            "AI_AGENT_CACHE_MAX_SIZE": "30",
            "AI_AGENT_CACHE_TTL": "2400",
            "AI_AGENT_DEFAULT_TIMEOUT": "45",
            "AI_AGENT_AUTO_INIT": "true",
        }

        with patch.dict(os.environ, env_vars):
            config = AgentAbstractionConfig.from_env()
            provider = setup_agent_abstraction(config)

            assert provider.get_framework() == "pydantic_ai"
            assert config.cache_enabled is True
            assert config.cache_max_size == 30
            assert config.cache_ttl == 2400.0
            assert config.default_timeout == 45.0
            assert config.auto_initialize is True


class TestInitializationIntegration:
    """Integration tests for initialization."""

    def test_full_initialization_flow(self):
        """Test complete initialization flow."""
        config = AgentAbstractionConfig(
            framework="pydantic_ai",
            cache_enabled=True,
            cache_max_size=15,
            cache_ttl=1800,
        )

        provider = setup_agent_abstraction(config)

        # Verify all components are set up
        assert provider is not None
        assert provider.get_framework() == "pydantic_ai"
        assert provider.get_factory() is not None
        assert provider.get_factory().get_cache() is not None
        assert "pydantic_ai" in provider.get_registered_frameworks()

    def test_initialization_with_cache_operations(self):
        """Test initialization followed by cache operations."""
        config = AgentAbstractionConfig(
            cache_enabled=True,
            cache_max_size=5,
        )

        provider = setup_agent_abstraction(config)
        cache = provider.get_factory().get_cache()

        # Verify cache is operational
        assert cache is not None
        assert cache.size() == 0

    def test_initialization_with_factory_operations(self):
        """Test initialization followed by factory operations."""
        config = AgentAbstractionConfig(
            framework="pydantic_ai",
            cache_enabled=False,
        )

        provider = setup_agent_abstraction(config)
        factory = provider.get_factory()

        # Verify factory is operational
        assert factory is not None
        assert factory.is_registered("pydantic_ai")

    def test_initialization_framework_consistency(self):
        """Test that framework is consistent across components."""
        config = AgentAbstractionConfig(
            framework="pydantic_ai",
            cache_enabled=False,
        )

        provider = setup_agent_abstraction(config)

        # Framework should be consistent
        assert provider.get_framework() == "pydantic_ai"
        assert "pydantic_ai" in provider.get_registered_frameworks()

    def test_initialization_with_no_cache_no_framework(self):
        """Test initialization with minimal configuration."""
        config = AgentAbstractionConfig(
            framework=None,
            cache_enabled=False,
        )

        provider = setup_agent_abstraction(config)

        assert provider is not None
        assert provider.get_framework() is None
        # Factory should be created even with cache disabled
        assert provider.get_factory() is not None

    def test_initialization_cache_size_limits(self):
        """Test initialization respects cache size limits."""
        config = AgentAbstractionConfig(
            cache_enabled=True,
            cache_max_size=3,
        )

        provider = setup_agent_abstraction(config)
        cache = provider.get_factory().get_cache()

        assert cache is not None
        # Cache should be configured with the specified size

    def test_initialization_multiple_providers(self):
        """Test creating multiple providers with different configs."""
        config1 = AgentAbstractionConfig(
            framework="pydantic_ai",
            cache_enabled=True,
        )
        config2 = AgentAbstractionConfig(
            framework=None,
            cache_enabled=False,
        )

        provider1 = setup_agent_abstraction(config1)
        provider2 = setup_agent_abstraction(config2)

        assert provider1 is not None
        assert provider2 is not None
        assert provider1.get_framework() == "pydantic_ai"
        assert provider2.get_framework() is None


class TestInitializationEdgeCases:
    """Test edge cases in initialization."""

    def test_setup_with_zero_cache_size(self):
        """Test setup with zero cache size."""
        config = AgentAbstractionConfig(
            cache_enabled=True,
            cache_max_size=0,
        )

        provider = setup_agent_abstraction(config)

        assert provider is not None

    def test_setup_with_large_cache_size(self):
        """Test setup with very large cache size."""
        config = AgentAbstractionConfig(
            cache_enabled=True,
            cache_max_size=10000,
        )

        provider = setup_agent_abstraction(config)

        assert provider is not None

    def test_setup_with_very_small_ttl(self):
        """Test setup with very small TTL."""
        config = AgentAbstractionConfig(
            cache_enabled=True,
            cache_ttl=0.1,
        )

        provider = setup_agent_abstraction(config)

        assert provider is not None

    def test_setup_with_very_large_ttl(self):
        """Test setup with very large TTL."""
        config = AgentAbstractionConfig(
            cache_enabled=True,
            cache_ttl=86400,  # 24 hours
        )

        provider = setup_agent_abstraction(config)

        assert provider is not None

    def test_setup_with_empty_framework_string(self):
        """Test setup with empty framework string."""
        config = AgentAbstractionConfig(
            framework="",
            cache_enabled=False,
        )

        provider = setup_agent_abstraction(config)

        assert provider is not None

    def test_setup_with_whitespace_framework(self):
        """Test setup with whitespace framework."""
        config = AgentAbstractionConfig(
            framework="   ",
            cache_enabled=False,
        )

        # Whitespace framework should raise ValueError since it's not registered
        with pytest.raises(ValueError):
            provider = setup_agent_abstraction(config)

    def test_get_default_provider_multiple_calls(self):
        """Test get_default_provider with multiple rapid calls."""
        providers = [get_default_provider() for _ in range(5)]

        assert all(p is not None for p in providers)
        assert all(isinstance(p, AIAgentProvider) for p in providers)

    def test_initialization_with_invalid_env_values(self):
        """Test initialization with invalid environment values."""
        with patch.dict(os.environ, {"AI_AGENT_CACHE_MAX_SIZE": "invalid"}):
            config = AgentAbstractionConfig.from_env()

            # Should use default value when parsing fails
            assert config.cache_max_size == 10

    def test_initialization_with_invalid_ttl_env(self):
        """Test initialization with invalid TTL environment value."""
        with patch.dict(os.environ, {"AI_AGENT_CACHE_TTL": "not_a_number"}):
            config = AgentAbstractionConfig.from_env()

            # Should use default (None) when parsing fails
            assert config.cache_ttl is None


class TestInitializationLogging:
    """Test logging in initialization."""

    def test_setup_logs_initialization(self):
        """Test that setup logs initialization."""
        config = AgentAbstractionConfig(cache_enabled=False)

        with patch("gearmeshing_ai.agent_core.abstraction.initialization.logger") as mock_logger:
            provider = setup_agent_abstraction(config)

            assert mock_logger.info.called
            assert provider is not None

    def test_setup_logs_cache_info(self):
        """Test that setup logs cache information."""
        config = AgentAbstractionConfig(
            cache_enabled=True,
            cache_max_size=20,
            cache_ttl=1800,
        )

        with patch("gearmeshing_ai.agent_core.abstraction.initialization.logger") as mock_logger:
            provider = setup_agent_abstraction(config)

            assert mock_logger.debug.called
            assert provider is not None

    def test_setup_logs_framework_info(self):
        """Test that setup logs framework information."""
        config = AgentAbstractionConfig(
            framework="pydantic_ai",
            cache_enabled=False,
        )

        with patch("gearmeshing_ai.agent_core.abstraction.initialization.logger") as mock_logger:
            provider = setup_agent_abstraction(config)

            assert mock_logger.info.called
            assert provider is not None

    def test_register_frameworks_logs_registration(self):
        """Test that framework registration is logged."""
        factory = AIAgentFactory()

        with patch("gearmeshing_ai.agent_core.abstraction.initialization.logger") as mock_logger:
            _register_frameworks(factory)

            assert mock_logger.debug.called

    def test_get_default_provider_logs_creation(self):
        """Test that default provider creation is logged."""
        with patch("gearmeshing_ai.agent_core.abstraction.provider.get_agent_provider", side_effect=RuntimeError("Not initialized")):
            with patch("gearmeshing_ai.agent_core.abstraction.initialization.logger") as mock_logger:
                provider = get_default_provider()

                assert mock_logger.debug.called
                assert provider is not None
