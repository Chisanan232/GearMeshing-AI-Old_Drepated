"""Tests for role provider loader module."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from gearmeshing_ai.info_provider.role.base import RoleProvider
from gearmeshing_ai.info_provider.role.loader import (
    _iter_entry_points,
    load_role_provider,
    load_role_provider_with_session,
)
from gearmeshing_ai.info_provider.role.provider import HardcodedRoleProvider


class TestIterEntryPoints:
    """Test the _iter_entry_points helper function."""

    def test_iter_entry_points_success_with_select(self):
        """Test successful entry point iteration with select method."""
        mock_ep1 = Mock()
        mock_ep1.name = "provider1"
        mock_ep2 = Mock()
        mock_ep2.name = "provider2"

        mock_eps = Mock()
        mock_eps.select.return_value = [mock_ep1, mock_ep2]

        with patch("gearmeshing_ai.info_provider.role.loader.metadata.entry_points", return_value=mock_eps):
            result = list(_iter_entry_points("test.group"))
            assert result == [mock_ep1, mock_ep2]
            mock_eps.select.assert_called_once_with(group="test.group")

    def test_iter_entry_points_fallback_no_select(self):
        """Test fallback when select method is not available."""
        mock_eps = Mock()
        mock_eps.select.side_effect = AttributeError("No select method")

        with patch("gearmeshing_ai.info_provider.role.loader.metadata.entry_points", return_value=mock_eps):
            result = list(_iter_entry_points("test.group"))
            assert result == []

    def test_iter_entry_points_exception_handling(self):
        """Test exception handling in entry points iteration."""
        with patch(
            "gearmeshing_ai.info_provider.role.loader.metadata.entry_points", side_effect=Exception("Metadata error")
        ):
            result = list(_iter_entry_points("test.group"))
            assert result == []

    def test_iter_entry_points_type_error_fallback(self):
        """Test fallback when TypeError is raised."""
        mock_eps = Mock()
        mock_eps.select.side_effect = TypeError("Type error")

        with patch("gearmeshing_ai.info_provider.role.loader.metadata.entry_points", return_value=mock_eps):
            result = list(_iter_entry_points("test.group"))
            assert result == []


class TestLoadRoleProvider:
    """Test the load_role_provider function."""

    @patch("gearmeshing_ai.server.core.config.settings")
    def test_load_hardcoded_provider_default(self, mock_settings):
        """Test loading hardcoded provider with default settings."""
        mock_settings.gearmeshing_ai_role_provider = "hardcoded"

        provider = load_role_provider()
        assert isinstance(provider, HardcodedRoleProvider)

    @patch("gearmeshing_ai.server.core.config.settings")
    def test_load_hardcoded_provider_empty_string(self, mock_settings):
        """Test loading hardcoded provider with empty string."""
        mock_settings.gearmeshing_ai_role_provider = ""

        provider = load_role_provider()
        assert isinstance(provider, HardcodedRoleProvider)

    @patch("gearmeshing_ai.server.core.config.settings")
    def test_load_hardcoded_provider_none(self, mock_settings):
        """Test loading hardcoded provider when setting is None."""
        del mock_settings.gearmeshing_ai_role_provider  # Simulate missing attribute

        provider = load_role_provider()
        assert isinstance(provider, HardcodedRoleProvider)

    @patch("gearmeshing_ai.server.core.config.settings")
    def test_load_database_provider_without_session(self, mock_settings):
        """Test loading database provider when session is not available."""
        mock_settings.gearmeshing_ai_role_provider = "database"

        with patch("gearmeshing_ai.info_provider.role.loader._LOGGER") as mock_logger:
            provider = load_role_provider()
            assert isinstance(provider, HardcodedRoleProvider)
            mock_logger.info.assert_called()
            mock_logger.warning.assert_called()

    @patch("gearmeshing_ai.server.core.config.settings")
    def test_load_database_provider_with_exception(self, mock_settings):
        """Test database provider loading with exception."""
        mock_settings.gearmeshing_ai_role_provider = "database"

        with patch("gearmeshing_ai.info_provider.role.loader._LOGGER") as mock_logger:
            # Simulate an exception during database provider init by mocking the import
            with patch("gearmeshing_ai.server.core.config.settings", side_effect=Exception("DB error")):
                provider = load_role_provider()
                assert isinstance(provider, HardcodedRoleProvider)
                mock_logger.warning.assert_called()

    @patch("gearmeshing_ai.server.core.config.settings")
    def test_load_custom_provider_success(self, mock_settings):
        """Test successful loading of custom provider via entry point."""
        mock_settings.gearmeshing_ai_role_provider = "custom_provider"

        # Create mock entry point and provider
        mock_provider = Mock(spec=RoleProvider)
        mock_factory = Mock(return_value=mock_provider)
        mock_ep = Mock()
        mock_ep.name = "custom_provider"
        mock_ep.load.return_value = mock_factory

        with patch("gearmeshing_ai.info_provider.role.loader._iter_entry_points", return_value=[mock_ep]):
            provider = load_role_provider()
            assert provider == mock_provider
            mock_ep.load.assert_called_once()
            mock_factory.assert_called_once()

    @patch("gearmeshing_ai.server.core.config.settings")
    def test_load_custom_provider_wrong_type(self, mock_settings):
        """Test custom provider that returns wrong type."""
        mock_settings.gearmeshing_ai_role_provider = "bad_provider"

        # Create mock entry point that returns non-RoleProvider
        mock_factory = Mock(return_value="not a provider")
        mock_ep = Mock()
        mock_ep.name = "bad_provider"
        mock_ep.load.return_value = mock_factory

        with patch("gearmeshing_ai.info_provider.role.loader._iter_entry_points", return_value=[mock_ep]):
            with patch("gearmeshing_ai.info_provider.role.loader._LOGGER") as mock_logger:
                provider = load_role_provider()
                assert isinstance(provider, HardcodedRoleProvider)
                mock_logger.warning.assert_called()

    @patch("gearmeshing_ai.server.core.config.settings")
    def test_load_custom_provider_load_exception(self, mock_settings):
        """Test custom provider that raises exception during load."""
        mock_settings.gearmeshing_ai_role_provider = "error_provider"

        # Create mock entry point that raises exception
        mock_ep = Mock()
        mock_ep.name = "error_provider"
        mock_ep.load.side_effect = Exception("Load failed")

        with patch("gearmeshing_ai.info_provider.role.loader._iter_entry_points", return_value=[mock_ep]):
            with patch("gearmeshing_ai.info_provider.role.loader._LOGGER") as mock_logger:
                provider = load_role_provider()
                assert isinstance(provider, HardcodedRoleProvider)
                mock_logger.warning.assert_called()

    @patch("gearmeshing_ai.server.core.config.settings")
    def test_load_custom_provider_not_found(self, mock_settings):
        """Test when custom provider is not found in entry points."""
        mock_settings.gearmeshing_ai_role_provider = "missing_provider"

        # Create mock entry point with different name
        mock_ep = Mock()
        mock_ep.name = "other_provider"

        with patch("gearmeshing_ai.info_provider.role.loader._iter_entry_points", return_value=[mock_ep]):
            with patch("gearmeshing_ai.info_provider.role.loader._LOGGER") as mock_logger:
                provider = load_role_provider()
                assert isinstance(provider, HardcodedRoleProvider)
                mock_logger.warning.assert_called()

    @patch("gearmeshing_ai.server.core.config.settings")
    def test_load_with_builtin_override(self, mock_settings):
        """Test loading with builtin provider override."""
        mock_settings.gearmeshing_ai_role_provider = "hardcoded"

        builtin_provider = Mock(spec=RoleProvider)
        provider = load_role_provider(builtin=builtin_provider)
        assert provider == builtin_provider

    @patch("gearmeshing_ai.server.core.config.settings")
    def test_load_with_builtin_override_fallback(self, mock_settings):
        """Test loading with builtin override for missing provider."""
        mock_settings.gearmeshing_ai_role_provider = "missing_provider"

        builtin_provider = Mock(spec=RoleProvider)
        with patch("gearmeshing_ai.info_provider.role.loader._iter_entry_points", return_value=[]):
            with patch("gearmeshing_ai.info_provider.role.loader._LOGGER") as mock_logger:
                provider = load_role_provider(builtin=builtin_provider)
                assert provider == builtin_provider
                mock_logger.warning.assert_called()


class TestLoadRoleProviderWithSession:
    """Test the load_role_provider_with_session function."""

    @patch("gearmeshing_ai.server.core.config.settings")
    def test_load_with_session_hardcoded(self, mock_settings):
        """Test loading hardcoded provider with session."""
        mock_settings.gearmeshing_ai_role_provider = "hardcoded"

        mock_session = Mock()
        provider = load_role_provider_with_session(mock_session)
        assert isinstance(provider, HardcodedRoleProvider)

    @patch("gearmeshing_ai.server.core.config.settings")
    @pytest.mark.skip("Database module not implemented")
    def test_load_with_session_database_success(self, mock_settings):
        """Test successful loading of database provider with session."""
        mock_settings.gearmeshing_ai_role_provider = "database"

        mock_session = Mock()
        mock_db_provider = Mock(spec=RoleProvider)

        # Mock the database import at the module level
        with patch.object(
            load_role_provider_with_session,
            "__globals__",
            {"get_database_role_provider": lambda session: mock_db_provider},
        ):
            provider = load_role_provider_with_session(mock_session)
            assert provider == mock_db_provider

    @patch("gearmeshing_ai.server.core.config.settings")
    @pytest.mark.skip("Database module not implemented")
    def test_load_with_session_database_exception(self, mock_settings):
        """Test database provider loading with exception."""
        mock_settings.gearmeshing_ai_role_provider = "database"

        mock_session = Mock()

        with patch(
            "gearmeshing_ai.info_provider.role.database.get_database_role_provider", side_effect=Exception("DB error")
        ):
            with patch("gearmeshing_ai.info_provider.role.loader._LOGGER") as mock_logger:
                provider = load_role_provider_with_session(mock_session)
                assert isinstance(provider, HardcodedRoleProvider)
                mock_logger.warning.assert_called()

    @patch("gearmeshing_ai.server.core.config.settings")
    def test_load_with_session_custom_provider(self, mock_settings):
        """Test loading custom provider with session (falls back to regular loader)."""
        mock_settings.gearmeshing_ai_role_provider = "custom_provider"

        mock_session = Mock()
        mock_provider = Mock(spec=RoleProvider)
        mock_factory = Mock(return_value=mock_provider)
        mock_ep = Mock()
        mock_ep.name = "custom_provider"
        mock_ep.load.return_value = mock_factory

        with patch("gearmeshing_ai.info_provider.role.loader._iter_entry_points", return_value=[mock_ep]):
            provider = load_role_provider_with_session(mock_session)
            assert provider == mock_provider

    @patch("gearmeshing_ai.server.core.config.settings")
    def test_load_with_session_builtin_override(self, mock_settings):
        """Test loading with builtin provider override and session."""
        mock_settings.gearmeshing_ai_role_provider = "hardcoded"

        mock_session = Mock()
        builtin_provider = Mock(spec=RoleProvider)
        provider = load_role_provider_with_session(mock_session, builtin=builtin_provider)
        assert provider == builtin_provider
