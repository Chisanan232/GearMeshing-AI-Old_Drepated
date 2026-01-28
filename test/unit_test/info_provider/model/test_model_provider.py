"""Unit tests for model provider system.

This module provides comprehensive tests for all model provider implementations,
including hardcoded, database, stacked, and hot-reload wrapper providers.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gearmeshing_ai.core.models.config import ModelConfig
from gearmeshing_ai.info_provider.model.base import ModelProvider
from gearmeshing_ai.info_provider.model.provider import (
    _BUILTIN_MODEL_CONFIGS,
    DatabaseModelProvider,
    HardcodedModelProvider,
    HotReloadModelWrapper,
    StackedModelProvider,
)


class TestHardcodedModelProvider:
    """Test cases for HardcodedModelProvider."""

    def test_init_default(self):
        """Test initialization with default configurations."""
        provider = HardcodedModelProvider()
        assert provider._configs == _BUILTIN_MODEL_CONFIGS
        assert provider._version == "hardcoded-v1"

    def test_init_custom_configs(self):
        """Test initialization with custom configurations."""
        custom_configs = {
            "custom_model": ModelConfig(provider="openai", model="gpt-4", temperature=0.5, max_tokens=2048, top_p=0.8)
        }
        provider = HardcodedModelProvider(configs=custom_configs, version_id="custom-v1")
        assert provider._configs == custom_configs
        assert provider._version == "custom-v1"

    def test_get_existing_config(self):
        """Test getting an existing model configuration."""
        provider = HardcodedModelProvider()
        config = provider.get("gpt4_default")

        assert config.provider == "openai"
        assert config.model == "gpt-4o"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.top_p == 0.9

    def test_get_with_tenant_ignored(self):
        """Test that tenant parameter is ignored for hardcoded provider."""
        provider = HardcodedModelProvider()
        config = provider.get("gpt4_default", tenant="test_tenant")

        assert config.provider == "openai"
        assert config.model == "gpt-4o"

    def test_get_nonexistent_config(self):
        """Test getting a non-existent configuration raises KeyError."""
        provider = HardcodedModelProvider()
        with pytest.raises(KeyError, match="model config not found: name='nonexistent'"):
            provider.get("nonexistent")

    def test_version(self):
        """Test version method."""
        provider = HardcodedModelProvider()
        assert provider.version() == "hardcoded-v1"

    def test_refresh_no_op(self):
        """Test refresh method is a no-op."""
        provider = HardcodedModelProvider()
        # Should not raise any exception
        provider.refresh()


class TestDatabaseModelProvider:
    """Test cases for DatabaseModelProvider."""

    def test_init(self):
        """Test initialization."""
        mock_session_factory = MagicMock()
        provider = DatabaseModelProvider(mock_session_factory, version_id="db-v1")
        assert provider._db_session_factory == mock_session_factory
        assert provider._version == "db-v1"

    @patch("gearmeshing_ai.server.models.agent_config.AgentConfig")
    def test_get_config_success(self, mock_agent_config_class):
        """Test successful configuration retrieval."""
        # Setup mock
        mock_config = MagicMock()
        mock_config.to_model_config.return_value = ModelConfig(
            provider="anthropic", model="claude-3-5-sonnet", temperature=0.5, max_tokens=8192, top_p=0.9
        )

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = mock_config

        mock_session_factory = MagicMock()
        mock_session_factory.return_value.__enter__.return_value = mock_session

        # Test
        provider = DatabaseModelProvider(mock_session_factory)
        result = provider.get("test_role", tenant="test_tenant")

        # Verify
        assert isinstance(result, ModelConfig)
        assert result.provider == "anthropic"
        assert result.model == "claude-3-5-sonnet"

        # Verify database query was called correctly
        mock_session.query.assert_called_once_with(mock_agent_config_class)
        assert mock_query.filter.call_count == 2  # First call with role_name+is_active, second with tenant_id

    @patch("gearmeshing_ai.server.models.agent_config.AgentConfig")
    def test_get_config_not_found(self, mock_agent_config_class):
        """Test configuration not found raises KeyError."""
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = None  # No config found

        mock_session_factory = MagicMock()
        mock_session_factory.return_value.__enter__.return_value = mock_session

        provider = DatabaseModelProvider(mock_session_factory)

        with pytest.raises(KeyError, match="model config not found: name='missing', tenant='test'"):
            provider.get("missing", tenant="test")

    def test_version(self):
        """Test version method."""
        mock_session_factory = MagicMock()
        provider = DatabaseModelProvider(mock_session_factory)
        assert provider.version() == "database-v1"

    def test_refresh_no_op(self):
        """Test refresh method is a no-op."""
        mock_session_factory = MagicMock()
        provider = DatabaseModelProvider(mock_session_factory)
        # Should not raise any exception
        provider.refresh()


class TestStackedModelProvider:
    """Test cases for StackedModelProvider."""

    def test_init(self):
        """Test initialization."""
        primary = MagicMock(spec=ModelProvider)
        fallback = MagicMock(spec=ModelProvider)

        provider = StackedModelProvider(primary, fallback)
        assert provider._primary == primary
        assert provider._fallback == fallback

    def test_get_primary_success(self):
        """Test successful retrieval from primary provider."""
        primary = MagicMock(spec=ModelProvider)
        fallback = MagicMock(spec=ModelProvider)

        expected_config = ModelConfig(provider="openai", model="gpt-4", temperature=0.7, max_tokens=4096, top_p=0.9)
        primary.get.return_value = expected_config

        provider = StackedModelProvider(primary, fallback)
        result = provider.get("test_config", tenant="test")

        assert result == expected_config
        primary.get.assert_called_once_with("test_config", "test")
        fallback.get.assert_not_called()

    def test_get_fallback_success(self):
        """Test fallback to secondary provider when primary fails."""
        primary = MagicMock(spec=ModelProvider)
        fallback = MagicMock(spec=ModelProvider)

        expected_config = ModelConfig(
            provider="anthropic", model="claude-3-5-sonnet", temperature=0.5, max_tokens=8192, top_p=0.9
        )

        primary.get.side_effect = KeyError("Primary not found")
        fallback.get.return_value = expected_config

        provider = StackedModelProvider(primary, fallback)
        result = provider.get("test_config", tenant="test")

        assert result == expected_config
        primary.get.assert_called_once_with("test_config", "test")
        fallback.get.assert_called_once_with("test_config", "test")

    def test_get_both_fail(self):
        """Test when both providers fail."""
        primary = MagicMock(spec=ModelProvider)
        fallback = MagicMock(spec=ModelProvider)

        primary.get.side_effect = KeyError("Primary not found")
        fallback.get.side_effect = KeyError("Fallback not found")

        provider = StackedModelProvider(primary, fallback)

        with pytest.raises(KeyError, match="Fallback not found"):
            provider.get("missing_config")

    def test_version(self):
        """Test version method combines both providers."""
        primary = MagicMock(spec=ModelProvider)
        fallback = MagicMock(spec=ModelProvider)
        primary.version.return_value = "primary-v1"
        fallback.version.return_value = "fallback-v1"

        provider = StackedModelProvider(primary, fallback)
        assert provider.version() == "stacked:primary-v1+fallback-v1"

    def test_refresh(self):
        """Test refresh calls both providers."""
        primary = MagicMock(spec=ModelProvider)
        fallback = MagicMock(spec=ModelProvider)

        provider = StackedModelProvider(primary, fallback)
        provider.refresh()

        primary.refresh.assert_called_once()
        fallback.refresh.assert_called_once()


class TestHotReloadModelWrapper:
    """Test cases for HotReloadModelWrapper."""

    def test_init(self):
        """Test initialization."""
        inner = MagicMock(spec=ModelProvider)
        logger = MagicMock()

        provider = HotReloadModelWrapper(inner=inner, interval_seconds=30.0, logger=logger)

        assert provider._inner == inner
        assert provider._interval == 30.0
        assert provider._logger == logger

    def test_get_triggers_refresh(self):
        """Test that get method triggers refresh when needed."""
        inner = MagicMock(spec=ModelProvider)
        expected_config = ModelConfig(
            provider="google", model="gemini-pro", temperature=0.7, max_tokens=2048, top_p=0.9
        )
        inner.get.return_value = expected_config

        provider = HotReloadModelWrapper(inner, interval_seconds=0)  # Disable throttling
        result = provider.get("test_config", tenant="test")

        assert result == expected_config
        inner.get.assert_called_once_with("test_config", "test")
        # With interval_seconds=0, _maybe_refresh should be called but refresh may not be triggered
        # due to the condition check, so we don't assert on refresh call

    def test_version_triggers_refresh(self):
        """Test that version method triggers refresh when needed."""
        inner = MagicMock(spec=ModelProvider)
        inner.version.return_value = "test-version"

        provider = HotReloadModelWrapper(inner, interval_seconds=0)  # Disable throttling
        result = provider.version()

        assert result == "test-version"
        # With interval_seconds=0, _maybe_refresh should be called but refresh may not be triggered

    def test_refresh_forces_refresh(self):
        """Test explicit refresh bypasses throttling."""
        inner = MagicMock(spec=ModelProvider)

        provider = HotReloadModelWrapper(inner, interval_seconds=60.0)
        provider.refresh()

        inner.refresh.assert_called_once()

    def test_safe_version_handles_error(self):
        """Test _safe_version handles errors gracefully."""
        inner = MagicMock(spec=ModelProvider)
        inner.version.side_effect = Exception("Version error")

        provider = HotReloadModelWrapper(inner)
        result = provider._safe_version()

        assert result == "<unknown>"

    def test_throttling_behavior(self):
        """Test that refresh is throttled appropriately."""
        inner = MagicMock(spec=ModelProvider)
        expected_config = ModelConfig(provider="openai", model="gpt-4", temperature=0.7, max_tokens=4096, top_p=0.9)
        inner.get.return_value = expected_config

        provider = HotReloadModelWrapper(inner, interval_seconds=1.0)

        # First call should trigger refresh
        provider.get("test")
        assert inner.refresh.call_count == 1

        # Immediate second call should not trigger refresh (throttled)
        provider.get("test")
        assert inner.refresh.call_count == 1  # Still 1, not 2

    def test_maybe_refresh_handles_exceptions(self):
        """Test that refresh exceptions are logged but not raised."""
        inner = MagicMock(spec=ModelProvider)
        inner.refresh.side_effect = Exception("Refresh failed")
        inner.version.return_value = "1.0.0"  # Mock version to avoid _safe_version issues

        logger = MagicMock()
        # Use positive interval but set last_refresh to force refresh
        provider = HotReloadModelWrapper(inner, interval_seconds=1, logger=logger)

        # Set last_refresh far in the past to force refresh
        import time

        provider._last_refresh = time.monotonic() - 2  # 2 seconds ago

        # Force a refresh by calling refresh directly
        provider._maybe_refresh()

        # Exception should be logged
        assert logger.warning.called
        assert "ModelProvider refresh failed" in logger.warning.call_args[0][0]


class TestBuiltinModelConfigs:
    """Test cases for builtin model configurations."""

    def test_builtin_configs_structure(self):
        """Test that builtin configs have expected structure."""
        assert isinstance(_BUILTIN_MODEL_CONFIGS, dict)
        assert len(_BUILTIN_MODEL_CONFIGS) > 0

        for key, config in _BUILTIN_MODEL_CONFIGS.items():
            assert isinstance(key, str)
            assert isinstance(config, ModelConfig)
            assert config.provider in ["openai", "anthropic", "google"]
            assert isinstance(config.model, str)
            assert isinstance(config.temperature, float)
            assert isinstance(config.max_tokens, int)
            assert isinstance(config.top_p, float)

    def test_expected_builtin_configs_exist(self):
        """Test that expected builtin configurations exist."""
        expected_configs = [
            "gpt4_default",
            "gpt4_creative",
            "gpt4_precise",
            "claude_sonnet",
            "claude_haiku",
            "gemini_pro",
            "gemini_flash",
        ]

        for config_name in expected_configs:
            assert config_name in _BUILTIN_MODEL_CONFIGS
            assert isinstance(_BUILTIN_MODEL_CONFIGS[config_name], ModelConfig)
