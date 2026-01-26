"""Unit tests for model provider loader.

This module provides comprehensive tests for the model provider loader system,
including entry point resolution and fallback behavior.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from gearmeshing_ai.info_provider.model.base import ModelProvider
from gearmeshing_ai.info_provider.model.loader import load_model_provider
from gearmeshing_ai.info_provider.model.provider import HardcodedModelProvider


class TestModelProviderLoader:
    """Test cases for ModelProviderLoader."""

    @patch('gearmeshing_ai.server.core.config.settings')
    def test_load_hardcoded_provider_default(self, mock_settings):
        """Test loading hardcoded provider when settings is default."""
        mock_settings.gearmeshing_ai_model_provider = "hardcoded"
        
        provider = load_model_provider()
        
        assert isinstance(provider, HardcodedModelProvider)

    @patch('gearmeshing_ai.server.core.config.settings')
    def test_load_hardcoded_provider_empty_string(self, mock_settings):
        """Test loading hardcoded provider when settings is empty string."""
        mock_settings.gearmeshing_ai_model_provider = ""
        
        provider = load_model_provider()
        
        assert isinstance(provider, HardcodedModelProvider)

    @patch('gearmeshing_ai.server.core.config.settings')
    def test_load_hardcoded_provider_none(self, mock_settings):
        """Test loading hardcoded provider when settings is None."""
        mock_settings.gearmeshing_ai_model_provider = None
        
        provider = load_model_provider()
        
        assert isinstance(provider, HardcodedModelProvider)

    @patch('gearmeshing_ai.server.core.config.settings')
    @patch('gearmeshing_ai.info_provider.model.loader._iter_entry_points')
    def test_load_custom_provider_success(self, mock_iter_eps, mock_settings):
        """Test successful loading of custom provider via entry point."""
        mock_settings.gearmeshing_ai_model_provider = "custom_provider"
        
        # Setup mock entry point
        mock_ep = MagicMock()
        mock_ep.name = "custom_provider"
        
        mock_provider = MagicMock(spec=ModelProvider)
        mock_factory = MagicMock(return_value=mock_provider)
        mock_ep.load.return_value = mock_factory
        
        mock_iter_eps.return_value = [mock_ep]
        
        provider = load_model_provider()
        
        assert provider == mock_provider
        mock_ep.load.assert_called_once()
        mock_factory.assert_called_once()

    @patch('gearmeshing_ai.server.core.config.settings')
    @patch('gearmeshing_ai.info_provider.model.loader._iter_entry_points')
    def test_load_custom_provider_wrong_type(self, mock_iter_eps, mock_settings):
        """Test fallback when custom provider returns wrong type."""
        mock_settings.gearmeshing_ai_model_provider = "wrong_provider"
        
        # Setup mock entry point that returns wrong type
        mock_ep = MagicMock()
        mock_ep.name = "wrong_provider"
        mock_ep.load.return_value = MagicMock()  # Not a ModelProvider
        
        mock_iter_eps.return_value = [mock_ep]
        
        provider = load_model_provider()
        
        assert isinstance(provider, HardcodedModelProvider)  # Should fallback

    @patch('gearmeshing_ai.server.core.config.settings')
    @patch('gearmeshing_ai.info_provider.model.loader._iter_entry_points')
    @patch('gearmeshing_ai.info_provider.model.loader._LOGGER')
    def test_load_custom_provider_import_error(self, mock_logger, mock_iter_eps, mock_settings):
        """Test fallback when custom provider raises import error."""
        mock_settings.gearmeshing_ai_model_provider = "broken_provider"
        
        # Setup mock entry point that raises error
        mock_ep = MagicMock()
        mock_ep.name = "broken_provider"
        mock_ep.load.side_effect = ImportError("Module not found")
        
        mock_iter_eps.return_value = [mock_ep]
        
        provider = load_model_provider()
        
        assert isinstance(provider, HardcodedModelProvider)  # Should fallback
        mock_logger.warning.assert_called()

    @patch('gearmeshing_ai.server.core.config.settings')
    @patch('gearmeshing_ai.info_provider.model.loader._iter_entry_points')
    @patch('gearmeshing_ai.info_provider.model.loader._LOGGER')
    def test_load_custom_provider_not_found(self, mock_logger, mock_iter_eps, mock_settings):
        """Test fallback when custom provider name not found in entry points."""
        mock_settings.gearmeshing_ai_model_provider = "nonexistent_provider"
        
        # Setup empty entry points
        mock_iter_eps.return_value = []
        
        provider = load_model_provider()
        
        assert isinstance(provider, HardcodedModelProvider)  # Should fallback
        mock_logger.warning.assert_called()

    @patch('gearmeshing_ai.server.core.config.settings')
    @patch('gearmeshing_ai.info_provider.model.loader._iter_entry_points')
    def test_load_custom_provider_multiple_entry_points(self, mock_iter_eps, mock_settings):
        """Test loading when multiple entry points exist."""
        mock_settings.gearmeshing_ai_model_provider = "target_provider"
        
        # Setup multiple entry points
        mock_ep1 = MagicMock()
        mock_ep1.name = "other_provider"
        
        mock_ep2 = MagicMock()
        mock_ep2.name = "target_provider"
        
        mock_provider = MagicMock(spec=ModelProvider)
        mock_factory = MagicMock(return_value=mock_provider)
        mock_ep2.load.return_value = mock_factory
        
        mock_iter_eps.return_value = [mock_ep1, mock_ep2]
        
        provider = load_model_provider()
        
        assert provider == mock_provider
        # Should only call load on the matching entry point
        mock_ep1.load.assert_not_called()
        mock_ep2.load.assert_called_once()

    def test_load_builtin_override(self):
        """Test loading with builtin override."""
        custom_builtin = HardcodedModelProvider(version_id="custom-builtin")
        
        provider = load_model_provider(builtin=custom_builtin)
        
        assert provider == custom_builtin
        assert provider.version() == "custom-builtin"

    @patch('gearmeshing_ai.server.core.config.settings')
    def test_load_with_settings_attribute_missing(self, mock_settings):
        """Test loading when settings attribute is missing."""
        # Simulate missing attribute
        del mock_settings.gearmeshing_ai_model_provider
        
        provider = load_model_provider()
        
        assert isinstance(provider, HardcodedModelProvider)

    @patch('gearmeshing_ai.info_provider.model.loader.metadata.entry_points')
    def test_iter_entry_points_exception_handling(self, mock_entry_points):
        """Test entry point iteration handles exceptions gracefully."""
        mock_entry_points.side_effect = Exception("Entry points error")
        
        # Import the actual function and call it
        from gearmeshing_ai.info_provider.model.loader import _iter_entry_points
        
        # Should not raise exception
        result = list(_iter_entry_points("test_group"))
        assert result == []


class TestIterEntryPoints:
    """Test cases for _iter_entry_points function."""

    @patch('gearmeshing_ai.info_provider.model.loader.metadata.entry_points')
    def test_iter_entry_points_success(self, mock_entry_points):
        """Test successful entry point iteration."""
        mock_eps = MagicMock()
        mock_eps.select.return_value = ["ep1", "ep2"]
        mock_entry_points.return_value = mock_eps
        
        from gearmeshing_ai.info_provider.model.loader import _iter_entry_points
        result = list(_iter_entry_points("test_group"))
        
        assert result == ["ep1", "ep2"]
        mock_eps.select.assert_called_once_with(group="test_group")

    @patch('gearmeshing_ai.info_provider.model.loader.metadata.entry_points')
    def test_iter_entry_points_metadata_exception(self, mock_entry_points):
        """Test handling of metadata.entry_points exception."""
        mock_entry_points.side_effect = Exception("Metadata error")
        
        from gearmeshing_ai.info_provider.model.loader import _iter_entry_points
        result = list(_iter_entry_points("test_group"))
        
        assert result == []

    @patch('gearmeshing_ai.info_provider.model.loader.metadata.entry_points')
    def test_iter_entry_points_no_select_method(self, mock_entry_points):
        """Test handling when entry_points doesn't have select method (older Python)."""
        # Create a mock that doesn't have select method
        class MockEntryPointsWithoutSelect:
            def __call__(self, *args, **kwargs):
                return ["ep1", "ep2"]
            # Explicitly ensure no select method
            def __getattr__(self, name):
                if name == 'select':
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
                return super().__getattribute__(name)
        
        mock_eps_without_select = MockEntryPointsWithoutSelect()
        mock_entry_points.return_value = mock_eps_without_select
        
        from gearmeshing_ai.info_provider.model.loader import _iter_entry_points
        # Should handle gracefully and return empty list
        result = list(_iter_entry_points("test_group"))
        
        assert result == []
