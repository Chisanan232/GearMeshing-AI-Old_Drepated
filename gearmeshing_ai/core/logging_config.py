"""
Logging Configuration Module.

This module provides centralized logging configuration for the GearMeshing-AI project.
It sets up structured logging with different levels for different modules and environments.

Features:
- Configurable log levels per module
- Console and file logging
- Structured logging with JSON format support
- Request/response logging for API calls
- Performance metrics logging
"""

import logging
import logging.config
import os
from pathlib import Path
from typing import Optional


def _get_logging_config():
    """Get logging configuration from settings model.

    This function is used to defer settings import until needed,
    avoiding circular imports during module initialization.
    """
    try:
        from gearmeshing_ai.server.core.config import settings

        return {
            "log_level": settings.gearmeshing_ai_log_level.upper(),
            "log_format": "detailed",  # Default format
            "log_file_dir": "logs",  # Default directory
            "enable_file_logging": True,  # Default enabled
        }
    except Exception:
        # Fallback to environment variables if settings not available
        return {
            "log_level": os.getenv("GEARMESHING_AI_LOG_LEVEL", "INFO").upper(),
            "log_format": os.getenv("LOG_FORMAT", "detailed"),
            "log_file_dir": os.getenv("LOG_FILE_DIR", "logs"),
            "enable_file_logging": os.getenv("ENABLE_FILE_LOGGING", "true").lower() in ("true", "1", "yes"),
        }


# Get initial logging config
_config = _get_logging_config()
LOG_LEVEL = _config["log_level"]
LOG_FORMAT = _config["log_format"]
LOG_FILE_DIR = _config["log_file_dir"]
ENABLE_FILE_LOGGING = _config["enable_file_logging"]

# Create logs directory if it doesn't exist
if ENABLE_FILE_LOGGING:
    Path(LOG_FILE_DIR).mkdir(parents=True, exist_ok=True)


# Define log formats
SIMPLE_FORMAT = "%(levelname)s - %(name)s - %(message)s"

DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s"

JSON_FORMAT = (
    '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", '
    '"module": "%(filename)s", "function": "%(funcName)s", "line": %(lineno)d, '
    '"message": "%(message)s"}'
)

# Select format based on environment
if LOG_FORMAT == "json":
    FORMAT = JSON_FORMAT
elif LOG_FORMAT == "simple":
    FORMAT = SIMPLE_FORMAT
else:
    FORMAT = DETAILED_FORMAT


# Module-specific log levels
MODULE_LOG_LEVELS = {
    # Core modules
    "gearmeshing_ai.agent_core": "DEBUG",
    "gearmeshing_ai.agent_core.planning": "DEBUG",
    "gearmeshing_ai.agent_core.runtime": "DEBUG",
    "gearmeshing_ai.agent_core.repos": "INFO",
    "gearmeshing_ai.agent_core.policy": "DEBUG",
    "gearmeshing_ai.agent_core.service": "DEBUG",
    "gearmeshing_ai.agent_core.model_provider": "DEBUG",
    # Server modules
    "gearmeshing_ai.server": "INFO",
    "gearmeshing_ai.server.api": "DEBUG",
    "gearmeshing_ai.server.services": "DEBUG",
    "gearmeshing_ai.server.core": "INFO",
    # Info provider modules
    "gearmeshing_ai.info_provider": "INFO",
    "gearmeshing_ai.info_provider.mcp": "DEBUG",
    "gearmeshing_ai.info_provider.prompt": "INFO",
    # Third-party libraries (reduce noise)
    "sqlalchemy": "WARNING",
    "sqlalchemy.engine": "WARNING",
    "sqlalchemy.pool": "WARNING",
    "httpx": "WARNING",
    "asyncio": "WARNING",
    "uvicorn": "INFO",
    "uvicorn.access": "INFO",
}


def setup_logging(
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    enable_file: bool = True,
) -> None:
    """
    Configure logging for the application.

    Args:
        log_level: Override default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Override default format (simple, detailed, json)
        enable_file: Whether to enable file logging
    """
    level = (log_level or LOG_LEVEL).upper()
    fmt = log_format or LOG_FORMAT

    # Determine format string
    if fmt == "json":
        format_str = JSON_FORMAT
    elif fmt == "simple":
        format_str = SIMPLE_FORMAT
    else:
        format_str = DETAILED_FORMAT

    # Create formatter
    formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels, filter at handler level

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if enabled)
    if enable_file and ENABLE_FILE_LOGGING:
        log_file = Path(LOG_FILE_DIR) / "gearmeshing_ai.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Configure module-specific log levels
    for module_name, module_level in MODULE_LOG_LEVELS.items():
        module_logger = logging.getLogger(module_name)
        module_logger.setLevel(module_level)

    # Log startup message
    root_logger.info(
        f"Logging configured: level={level}, format={fmt}, file_logging={enable_file and ENABLE_FILE_LOGGING}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: The module name (typically __name__)

    Returns:
        A configured logger instance
    """
    return logging.getLogger(name)


# Configure logging on module import
setup_logging()
