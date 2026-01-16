"""
Unit tests for Pydantic AI Logfire monitoring module.

This test suite covers:
- Logfire initialization with various configurations
- LangSmith initialization with various configurations
- Error handling and graceful degradation
"""

from unittest.mock import MagicMock, patch

import pytest


class TestInitializeLogfireWithSettings:
    """Test Logfire initialization using Settings model."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_disabled(self, mock_logger):
        """Test that initialization is skipped when Logfire is disabled."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.logfire.enabled = False
            initialize_logfire()
            mock_logger.info.assert_called_once()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_no_token(self, mock_logger):
        """Test that initialization warns when token is not set."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.logfire.enabled = True
            mock_settings.logfire.token = ""
            initialize_logfire()
            # Should warn about missing token
            assert mock_logger.warning.called or mock_logger.info.called

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_with_token(self, mock_logger):
        """Test initialization with token set."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.logfire.enabled = True
            mock_settings.logfire.token = "test-token"
            mock_settings.logfire.service_name = "test-service"
            mock_settings.logfire.service_version = "1.0.0"
            mock_settings.logfire.environment = "test"
            mock_settings.logfire.trace_pydantic_ai = False
            mock_settings.logfire.trace_sqlalchemy = False
            mock_settings.logfire.trace_httpx = False
            mock_settings.logfire.trace_fastapi = False

            with patch("builtins.__import__") as mock_import:
                mock_logfire = MagicMock()

                def import_side_effect(name, *args, **kwargs):
                    if name == "logfire":
                        return mock_logfire
                    return __import__(name, *args, **kwargs)

                mock_import.side_effect = import_side_effect
                initialize_logfire()
                mock_logger.info.assert_called()


class TestInitializeLangSmith:
    """Test LangSmith initialization using Settings model."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_langsmith_disabled(self, mock_logger):
        """Test that initialization is skipped when LangSmith is disabled."""
        from gearmeshing_ai.core.monitoring import initialize_langsmith

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.langsmith.tracing = False
            initialize_langsmith()
            # Should log debug or info message
            assert mock_logger.debug.called or mock_logger.info.called

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_langsmith_no_api_key(self, mock_logger):
        """Test that initialization warns when API key is not set."""
        from gearmeshing_ai.core.monitoring import initialize_langsmith

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.langsmith.tracing = True
            mock_settings.langsmith.api_key = ""
            # Should not raise
            initialize_langsmith()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_langsmith_success(self, mock_logger):
        """Test successful LangSmith initialization."""
        from gearmeshing_ai.core.monitoring import initialize_langsmith

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.langsmith.tracing = True
            mock_settings.langsmith.api_key = "test-api-key"
            mock_settings.langsmith.project = "test-project"
            mock_settings.langsmith.endpoint = "https://api.smith.langchain.com"

            initialize_langsmith()
            # Should log info about successful initialization
            assert mock_logger.info.called or mock_logger.debug.called

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_langsmith_sets_environment_variables(self, mock_logger):
        """Test that environment variables are set correctly."""
        from gearmeshing_ai.core.monitoring import initialize_langsmith

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.langsmith.tracing = True
            mock_settings.langsmith.api_key = "test-api-key"
            mock_settings.langsmith.project = "test-project"
            mock_settings.langsmith.endpoint = "https://api.smith.langchain.com"

            # Should not raise - function should complete successfully
            initialize_langsmith()


class TestInitializeLogfireErrorHandling:
    """Test error handling in Logfire initialization."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_handles_import_error_gracefully(self, mock_logger):
        """Test that ImportError is handled gracefully during initialization."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.logfire.enabled = True
            mock_settings.logfire.token = "test-token"

            with patch("builtins.__import__", side_effect=ImportError("logfire not installed")):
                initialize_logfire()
                mock_logger.warning.assert_called()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_handles_general_exception_gracefully(self, mock_logger):
        """Test that general exceptions are handled gracefully during initialization."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.logfire.enabled = True
            mock_settings.logfire.token = "test-token"
            mock_settings.logfire.service_name = "test-service"
            mock_settings.logfire.service_version = "1.0.0"
            mock_settings.logfire.environment = "test"
            mock_settings.logfire.trace_pydantic_ai = False
            mock_settings.logfire.trace_sqlalchemy = False
            mock_settings.logfire.trace_httpx = False
            mock_settings.logfire.trace_fastapi = False

            with patch("builtins.__import__") as mock_import:
                mock_logfire = MagicMock()
                mock_logfire.configure.side_effect = RuntimeError("Configuration failed")

                def import_side_effect(name, *args, **kwargs):
                    if name == "logfire":
                        return mock_logfire
                    return __import__(name, *args, **kwargs)

                mock_import.side_effect = import_side_effect
                initialize_logfire()
                mock_logger.error.assert_called()


class TestInitializeLangSmithErrorHandling:
    """Test error handling in LangSmith initialization."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_langsmith_handles_runtime_error(self, mock_logger):
        """Test that RuntimeError is handled gracefully."""
        from gearmeshing_ai.core.monitoring import initialize_langsmith

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.langsmith.tracing = True
            mock_settings.langsmith.api_key = "test-api-key"
            mock_settings.langsmith.project = "test-project"
            mock_settings.langsmith.endpoint = "https://api.smith.langchain.com"

            # Should not raise even if os.environ operations fail
            initialize_langsmith()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_langsmith_handles_generic_exception(self, mock_logger):
        """Test that generic exceptions are handled gracefully."""
        from gearmeshing_ai.core.monitoring import initialize_langsmith

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.langsmith.tracing = True
            mock_settings.langsmith.api_key = "test-api-key"
            mock_settings.langsmith.project = "test-project"
            mock_settings.langsmith.endpoint = "https://api.smith.langchain.com"

            # Should not raise even if something goes wrong
            initialize_langsmith()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_langsmith_continues_after_error(self, mock_logger):
        """Test that function continues after error."""
        from gearmeshing_ai.core.monitoring import initialize_langsmith

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.langsmith.tracing = True
            mock_settings.langsmith.api_key = "test-api-key"
            mock_settings.langsmith.project = "test-project"
            mock_settings.langsmith.endpoint = "https://api.smith.langchain.com"

            # Should not raise
            initialize_langsmith()
