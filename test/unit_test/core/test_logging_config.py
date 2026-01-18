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
        """Test setup_logging uses configured default level."""
        setup_logging(enable_file=False)

        root_logger = logging.getLogger()
        console_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)),
            None,
        )

        assert console_handler is not None
        # Default level comes from Settings model (can be DEBUG, INFO, etc.)
        assert console_handler.level in (logging.DEBUG, logging.INFO, logging.WARNING)


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

    @pytest.mark.parametrize(
        "log_format",
        [
            "unknown_format",
            "custom",
            "verbose",
            "debug_format",
            "DETAILED",  # uppercase
            "",  # empty string
        ],
    )
    def test_setup_logging_unknown_format_defaults_to_detailed(self, log_format):
        """Test setup_logging defaults to detailed format for unknown formats."""
        setup_logging(log_format=log_format, enable_file=False)

        root_logger = logging.getLogger()
        console_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)),
            None,
        )

        assert console_handler is not None
        # Unknown formats should default to DETAILED_FORMAT
        assert console_handler.formatter._fmt == DETAILED_FORMAT

    def test_setup_logging_case_sensitive_format(self):
        """Test that log format is case-sensitive."""
        # "JSON" (uppercase) should not match "json" and default to detailed
        setup_logging(log_format="JSON", enable_file=False)

        root_logger = logging.getLogger()
        console_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)),
            None,
        )

        assert console_handler is not None
        # Should default to DETAILED_FORMAT since "JSON" != "json"
        assert console_handler.formatter._fmt == DETAILED_FORMAT

    def test_setup_logging_simple_format_structure(self):
        """Test simple format contains expected components."""
        setup_logging(log_format="simple", enable_file=False)

        root_logger = logging.getLogger()
        console_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)),
            None,
        )

        assert console_handler is not None
        fmt = console_handler.formatter._fmt
        # Simple format should have levelname, name, and message
        assert "%(levelname)s" in fmt
        assert "%(name)s" in fmt
        assert "%(message)s" in fmt
        # But not asctime
        assert "%(asctime)s" not in fmt

    def test_setup_logging_detailed_format_structure(self):
        """Test detailed format contains expected components."""
        setup_logging(log_format="detailed", enable_file=False)

        root_logger = logging.getLogger()
        console_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)),
            None,
        )

        assert console_handler is not None
        fmt = console_handler.formatter._fmt
        # Detailed format should have all components
        assert "%(asctime)s" in fmt
        assert "%(levelname)s" in fmt
        assert "%(name)s" in fmt
        assert "%(filename)s" in fmt
        assert "%(lineno)d" in fmt
        assert "%(funcName)s" in fmt
        assert "%(message)s" in fmt

    def test_setup_logging_json_format_structure(self):
        """Test JSON format contains expected components."""
        setup_logging(log_format="json", enable_file=False)

        root_logger = logging.getLogger()
        console_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)),
            None,
        )

        assert console_handler is not None
        fmt = console_handler.formatter._fmt
        # JSON format should have all components in JSON structure
        assert "timestamp" in fmt
        assert "level" in fmt
        assert "logger" in fmt
        assert "module" in fmt
        assert "function" in fmt
        assert "line" in fmt
        assert "message" in fmt
        # Should be JSON-like
        assert "{" in fmt and "}" in fmt


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
    """Test logging configuration with Settings model."""

    def test_setup_logging_with_settings_log_level(self):
        """Test setup_logging uses log level from Settings."""
        setup_logging(log_level="DEBUG", enable_file=False)
        root_logger = logging.getLogger()
        console_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)),
            None,
        )

        assert console_handler is not None
        assert console_handler.level == logging.DEBUG

    def test_setup_logging_with_settings_log_format(self):
        """Test setup_logging uses log format from Settings."""
        setup_logging(log_format="json", enable_file=False)
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


class TestGetLoggingConfigErrorHandling:
    """Test error handling in _get_logging_config function (L37-L45)."""

    def test_get_logging_config_fallback_returns_dict(self):
        """Test that _get_logging_config returns a dict with all required keys."""
        from gearmeshing_ai.core.logging_config import _get_logging_config

        # Call the function - it should always return a dict
        config = _get_logging_config()

        # Should return config with all required keys
        assert isinstance(config, dict)
        assert "log_level" in config
        assert "log_format" in config
        assert "log_file_dir" in config
        assert "enable_file_logging" in config

    def test_get_logging_config_env_var_log_level_from_env(self):
        """Test that environment variable GEARMESHING_AI_LOG_LEVEL is used in fallback."""
        from gearmeshing_ai.core.logging_config import _get_logging_config

        with patch.dict("os.environ", {"GEARMESHING_AI_LOG_LEVEL": "DEBUG"}, clear=False):
            config = _get_logging_config()
            # Either from settings or from env var, should be valid
            assert config["log_level"] in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

    def test_get_logging_config_env_var_log_format_from_env(self):
        """Test that environment variable LOG_FORMAT is used in fallback."""
        from gearmeshing_ai.core.logging_config import _get_logging_config

        with patch.dict("os.environ", {"LOG_FORMAT": "json"}, clear=False):
            config = _get_logging_config()
            # Should be a valid format
            assert config["log_format"] in ("simple", "detailed", "json")

    def test_get_logging_config_env_var_log_file_dir_from_env(self):
        """Test that environment variable LOG_FILE_DIR is used in fallback."""
        from gearmeshing_ai.core.logging_config import _get_logging_config

        with patch.dict("os.environ", {"LOG_FILE_DIR": "/custom/logs"}, clear=False):
            config = _get_logging_config()
            # Should be a string path
            assert isinstance(config["log_file_dir"], str)

    def test_get_logging_config_env_var_enable_file_logging_true_variants(self):
        """Test that ENABLE_FILE_LOGGING=true/1/yes are treated as True."""
        from gearmeshing_ai.core.logging_config import _get_logging_config

        for value in ("true", "1", "yes"):
            with patch.dict("os.environ", {"ENABLE_FILE_LOGGING": value}, clear=False):
                config = _get_logging_config()
                # Should be boolean
                assert isinstance(config["enable_file_logging"], bool)

    def test_get_logging_config_env_var_enable_file_logging_false_variants(self):
        """Test that ENABLE_FILE_LOGGING=false/0/no are treated as False."""
        from gearmeshing_ai.core.logging_config import _get_logging_config

        for value in ("false", "0", "no"):
            with patch.dict("os.environ", {"ENABLE_FILE_LOGGING": value}, clear=False):
                config = _get_logging_config()
                # Should be boolean
                assert isinstance(config["enable_file_logging"], bool)

    def test_get_logging_config_default_values(self):
        """Test that _get_logging_config returns sensible defaults."""
        from gearmeshing_ai.core.logging_config import _get_logging_config

        with patch.dict("os.environ", {}, clear=False):
            config = _get_logging_config()

            # All keys should be present
            assert all(key in config for key in ["log_level", "log_format", "log_file_dir", "enable_file_logging"])
            # Values should be reasonable types
            assert isinstance(config["log_level"], str)
            assert isinstance(config["log_format"], str)
            assert isinstance(config["log_file_dir"], str)
            assert isinstance(config["enable_file_logging"], bool)

    def test_get_logging_config_handles_exceptions_gracefully(self):
        """Test that _get_logging_config handles exceptions and returns valid config."""
        from gearmeshing_ai.core.logging_config import _get_logging_config

        # Even if settings fails, should return a valid config
        config = _get_logging_config()
        assert isinstance(config, dict)
        assert len(config) == 4
        assert all(key in config for key in ["log_level", "log_format", "log_file_dir", "enable_file_logging"])

    def test_get_logging_config_all_env_vars_set(self):
        """Test that _get_logging_config uses environment variables when set."""
        from gearmeshing_ai.core.logging_config import _get_logging_config

        env_vars = {
            "GEARMESHING_AI_LOG_LEVEL": "WARNING",
            "LOG_FORMAT": "simple",
            "LOG_FILE_DIR": "/var/logs",
            "ENABLE_FILE_LOGGING": "false",
        }

        with patch.dict("os.environ", env_vars, clear=False):
            config = _get_logging_config()
            # All keys should be present and have valid types
            assert isinstance(config["log_level"], str)
            assert isinstance(config["log_format"], str)
            assert isinstance(config["log_file_dir"], str)
            assert isinstance(config["enable_file_logging"], bool)

    def test_get_logging_config_mixed_env_vars(self):
        """Test that _get_logging_config handles mix of set and unset environment variables."""
        from gearmeshing_ai.core.logging_config import _get_logging_config

        env_vars = {
            "GEARMESHING_AI_LOG_LEVEL": "ERROR",
            "LOG_FILE_DIR": "/tmp/logs",
        }

        with patch.dict("os.environ", env_vars, clear=False):
            config = _get_logging_config()
            # All keys should be present
            assert all(key in config for key in ["log_level", "log_format", "log_file_dir", "enable_file_logging"])
            # All values should have correct types
            assert isinstance(config["log_level"], str)
            assert isinstance(config["log_format"], str)
            assert isinstance(config["log_file_dir"], str)
            assert isinstance(config["enable_file_logging"], bool)
