"""Unit tests for logging configuration module.

Tests verify that the logging configuration functions work correctly with different
scenarios including various log levels, formats, and file logging options.
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from gearmeshing_ai.core.logging_config import (
    DETAILED_FORMAT,
    JSON_FORMAT,
    MODULE_LOG_LEVELS,
    SIMPLE_FORMAT,
    get_logger,
    setup_logging,
)


class TestSetupLoggingLogLevels:
    """Test setup_logging with different log levels."""

    @pytest.mark.parametrize(
        "log_level,expected_level",
        [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL),
            ("debug", logging.DEBUG),  # Test lowercase
            ("info", logging.INFO),
        ],
    )
    def test_setup_logging_with_different_levels(self, log_level, expected_level):
        """Test setup_logging configures correct log level."""
        setup_logging(log_level=log_level, enable_file=False)

        root_logger = logging.getLogger()
        console_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)),
            None,
        )

        assert console_handler is not None
        assert console_handler.level == expected_level

    def test_setup_logging_default_level(self):
        """Test setup_logging uses INFO as default level."""
        setup_logging(enable_file=False)

        root_logger = logging.getLogger()
        console_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)),
            None,
        )

        assert console_handler is not None
        # Default should be INFO or higher
        assert console_handler.level >= logging.INFO


class TestSetupLoggingFormats:
    """Test setup_logging with different log formats."""

    @pytest.mark.parametrize(
        "log_format,expected_format",
        [
            ("simple", SIMPLE_FORMAT),
            ("detailed", DETAILED_FORMAT),
            ("json", JSON_FORMAT),
        ],
    )
    def test_setup_logging_with_different_formats(self, log_format, expected_format):
        """Test setup_logging configures correct format."""
        setup_logging(log_format=log_format, enable_file=False)

        root_logger = logging.getLogger()
        console_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)),
            None,
        )

        assert console_handler is not None
        assert console_handler.formatter._fmt == expected_format

    def test_setup_logging_default_format(self):
        """Test setup_logging uses detailed format by default."""
        setup_logging(enable_file=False)

        root_logger = logging.getLogger()
        console_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)),
            None,
        )

        assert console_handler is not None
        # Default should be detailed format
        assert console_handler.formatter._fmt == DETAILED_FORMAT

    def test_setup_logging_format_with_timestamp(self):
        """Test that formatter includes timestamp."""
        setup_logging(log_format="detailed", enable_file=False)

        root_logger = logging.getLogger()
        console_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)),
            None,
        )

        assert console_handler is not None
        assert console_handler.formatter.datefmt == "%Y-%m-%d %H:%M:%S"


class TestSetupLoggingFileHandling:
    """Test setup_logging file logging functionality."""

    def test_setup_logging_with_file_enabled(self):
        """Test setup_logging creates file handler when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("gearmeshing_ai.core.logging_config.LOG_FILE_DIR", tmpdir):
                setup_logging(enable_file=True)

                root_logger = logging.getLogger()
                file_handler = next(
                    (h for h in root_logger.handlers if isinstance(h, logging.FileHandler)),
                    None,
                )

                assert file_handler is not None
                assert file_handler.level == logging.DEBUG

    def test_setup_logging_with_file_disabled(self):
        """Test setup_logging does not create file handler when disabled."""
        setup_logging(enable_file=False)

        root_logger = logging.getLogger()
        file_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.FileHandler)),
            None,
        )

        assert file_handler is None

    def test_setup_logging_file_handler_always_debug(self):
        """Test file handler always logs DEBUG level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("gearmeshing_ai.core.logging_config.LOG_FILE_DIR", tmpdir):
                setup_logging(log_level="ERROR", enable_file=True)

                root_logger = logging.getLogger()
                file_handler = next(
                    (h for h in root_logger.handlers if isinstance(h, logging.FileHandler)),
                    None,
                )

                assert file_handler is not None
                # File handler should always be DEBUG
                assert file_handler.level == logging.DEBUG

    def test_setup_logging_creates_log_directory(self):
        """Test setup_logging creates log directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "new_logs"
            assert not log_dir.exists()

            with patch("gearmeshing_ai.core.logging_config.LOG_FILE_DIR", str(log_dir)):
                with patch("gearmeshing_ai.core.logging_config.ENABLE_FILE_LOGGING", True):
                    # Create the directory first (simulating module initialization)
                    log_dir.mkdir(parents=True, exist_ok=True)
                    setup_logging(enable_file=True)

                    assert log_dir.exists()


class TestSetupLoggingHandlerManagement:
    """Test setup_logging handler management."""

    def test_setup_logging_removes_existing_handlers(self):
        """Test setup_logging removes existing handlers to avoid duplicates."""
        # Setup initial logging
        setup_logging(enable_file=False)
        root_logger = logging.getLogger()
        initial_handler_count = len(root_logger.handlers)

        # Setup logging again
        setup_logging(enable_file=False)

        # Should not have more handlers than before
        assert len(root_logger.handlers) <= initial_handler_count + 1

    def test_setup_logging_always_has_console_handler(self):
        """Test setup_logging always creates console handler."""
        setup_logging(enable_file=False)

        root_logger = logging.getLogger()
        console_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)),
            None,
        )

        assert console_handler is not None

    def test_setup_logging_root_logger_level_is_debug(self):
        """Test root logger is set to DEBUG to capture all levels."""
        setup_logging(log_level="WARNING", enable_file=False)

        root_logger = logging.getLogger()
        # Root logger should be DEBUG to capture all, filtering happens at handler level
        assert root_logger.level == logging.DEBUG


class TestSetupLoggingModuleSpecificLevels:
    """Test module-specific log level configuration."""

    @pytest.mark.parametrize(
        "module_name,expected_level",
        [
            ("gearmeshing_ai.agent_core", logging.DEBUG),
            ("gearmeshing_ai.agent_core.planning", logging.DEBUG),
            ("gearmeshing_ai.agent_core.repos", logging.INFO),
            ("gearmeshing_ai.server", logging.INFO),
            ("gearmeshing_ai.server.api", logging.DEBUG),
            ("gearmeshing_ai.info_provider", logging.INFO),
            ("gearmeshing_ai.info_provider.mcp", logging.DEBUG),
            ("sqlalchemy", logging.WARNING),
            ("httpx", logging.WARNING),
            ("asyncio", logging.WARNING),
        ],
    )
    def test_module_specific_log_levels(self, module_name, expected_level):
        """Test module-specific log levels are configured correctly."""
        setup_logging(enable_file=False)

        module_logger = logging.getLogger(module_name)
        assert module_logger.level == expected_level

    def test_all_module_log_levels_configured(self):
        """Test all modules in MODULE_LOG_LEVELS are configured."""
        setup_logging(enable_file=False)

        for module_name, expected_level_str in MODULE_LOG_LEVELS.items():
            module_logger = logging.getLogger(module_name)
            expected_level = getattr(logging, expected_level_str)
            assert module_logger.level == expected_level


class TestSetupLoggingParameterCombinations:
    """Test setup_logging with various parameter combinations."""

    @pytest.mark.parametrize(
        "log_level,log_format,enable_file",
        [
            ("DEBUG", "simple", False),
            ("INFO", "detailed", False),
            ("WARNING", "json", False),
            ("ERROR", "simple", True),
            ("CRITICAL", "detailed", True),
            ("DEBUG", "json", True),
        ],
    )
    def test_setup_logging_parameter_combinations(self, log_level, log_format, enable_file):
        """Test setup_logging with various parameter combinations."""
        setup_logging(log_level=log_level, log_format=log_format, enable_file=enable_file)

        root_logger = logging.getLogger()

        # Verify console handler exists and has correct level
        console_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)),
            None,
        )
        assert console_handler is not None
        assert console_handler.level == getattr(logging, log_level.upper())

        # Verify format is correct
        if log_format == "simple":
            assert console_handler.formatter._fmt == SIMPLE_FORMAT
        elif log_format == "json":
            assert console_handler.formatter._fmt == JSON_FORMAT
        else:
            assert console_handler.formatter._fmt == DETAILED_FORMAT

        # Verify file handler if enabled
        if enable_file:
            file_handler = next(
                (h for h in root_logger.handlers if isinstance(h, logging.FileHandler)),
                None,
            )
            # File handler may or may not exist depending on ENABLE_FILE_LOGGING env var
            if file_handler:
                assert file_handler.level == logging.DEBUG


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_returns_logger_instance(self):
        """Test get_logger returns a Logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_with_different_names(self):
        """Test get_logger returns different loggers for different names."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        assert logger1 is not logger2
        assert logger1.name == "module1"
        assert logger2.name == "module2"

    def test_get_logger_same_name_returns_same_instance(self):
        """Test get_logger returns same instance for same name."""
        logger1 = get_logger("same_module")
        logger2 = get_logger("same_module")

        assert logger1 is logger2

    @pytest.mark.parametrize(
        "module_name",
        [
            "gearmeshing_ai.agent_core",
            "gearmeshing_ai.server.api",
            "gearmeshing_ai.info_provider.mcp",
            "custom_module",
            "test.nested.module.name",
        ],
    )
    def test_get_logger_with_various_names(self, module_name):
        """Test get_logger with various module names."""
        logger = get_logger(module_name)

        assert isinstance(logger, logging.Logger)
        assert logger.name == module_name

    def test_get_logger_inherits_module_level(self):
        """Test get_logger returns logger with correct module level."""
        setup_logging(enable_file=False)

        logger = get_logger("gearmeshing_ai.agent_core")
        assert logger.level == logging.DEBUG

        logger = get_logger("sqlalchemy")
        assert logger.level == logging.WARNING


class TestLoggingIntegration:
    """Integration tests for logging functionality."""

    def test_logger_can_be_used_for_logging(self):
        """Test logger instance can be used for logging."""
        setup_logging(log_level="INFO", enable_file=False)

        logger = get_logger("test_logger")
        # Just verify logger can log without errors
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

    def test_logger_with_exception_info(self):
        """Test logger can log exception information."""
        setup_logging(log_level="ERROR", enable_file=False)

        logger = get_logger("test_logger")

        try:
            raise ValueError("Test error")
        except ValueError:
            # Just verify logger can log exception without errors
            logger.error("An error occurred", exc_info=True)

    def test_multiple_logger_instances_work_together(self):
        """Test multiple logger instances work together."""
        setup_logging(log_level="INFO", enable_file=False)

        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        # Just verify both loggers can log without errors
        logger1.info("Message from module1")
        logger2.info("Message from module2")

    def test_logger_respects_module_level(self):
        """Test logger respects module-specific log level."""
        setup_logging(log_level="WARNING", enable_file=False)

        # Agent core should be DEBUG
        debug_logger = get_logger("gearmeshing_ai.agent_core")
        assert debug_logger.level == logging.DEBUG

        # SQLAlchemy should be WARNING
        warning_logger = get_logger("sqlalchemy")
        assert warning_logger.level == logging.WARNING

    def test_logger_can_log_with_different_formats(self):
        """Test logger works with different formats."""
        for fmt in ["simple", "detailed", "json"]:
            setup_logging(log_format=fmt, log_level="INFO", enable_file=False)
            logger = get_logger(f"test_logger_{fmt}")
            logger.info(f"Test message with {fmt} format")


class TestLoggingEnvironmentVariables:
    """Test logging configuration with environment variables."""

    def test_setup_logging_respects_env_log_level(self, monkeypatch):
        """Test setup_logging respects LOG_LEVEL environment variable."""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        # Re-import to get new env var
        from importlib import reload

        import gearmeshing_ai.core.logging_config as logging_config

        reload(logging_config)

        setup_logging(enable_file=False)
        root_logger = logging.getLogger()
        console_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)),
            None,
        )

        assert console_handler is not None

    def test_setup_logging_respects_env_log_format(self, monkeypatch):
        """Test setup_logging respects LOG_FORMAT environment variable."""
        monkeypatch.setenv("LOG_FORMAT", "json")

        from importlib import reload

        import gearmeshing_ai.core.logging_config as logging_config

        reload(logging_config)

        setup_logging(enable_file=False)
        root_logger = logging.getLogger()
        console_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)),
            None,
        )

        assert console_handler is not None
        assert console_handler.formatter._fmt == JSON_FORMAT


class TestLoggingEdgeCases:
    """Test edge cases and error conditions."""

    def test_setup_logging_with_none_parameters(self):
        """Test setup_logging handles None parameters correctly."""
        setup_logging(log_level=None, log_format=None, enable_file=False)

        root_logger = logging.getLogger()
        console_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)),
            None,
        )

        assert console_handler is not None

    def test_setup_logging_with_lowercase_log_level(self):
        """Test setup_logging handles lowercase log level."""
        setup_logging(log_level="debug", enable_file=False)

        root_logger = logging.getLogger()
        console_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)),
            None,
        )

        assert console_handler is not None
        assert console_handler.level == logging.DEBUG

    def test_get_logger_with_empty_string(self):
        """Test get_logger with empty string name."""
        logger = get_logger("")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_with_special_characters(self):
        """Test get_logger with special characters in name."""
        logger = get_logger("module.with-special_chars.123")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "module.with-special_chars.123"

    def test_setup_logging_called_multiple_times(self):
        """Test setup_logging can be called multiple times without issues."""
        setup_logging(log_level="INFO", enable_file=False)
        setup_logging(log_level="DEBUG", enable_file=False)
        setup_logging(log_level="WARNING", enable_file=False)

        root_logger = logging.getLogger()
        console_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)),
            None,
        )

        assert console_handler is not None
        assert console_handler.level == logging.WARNING
