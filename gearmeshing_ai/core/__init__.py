"""
Core utilities and configuration for GearMeshing-AI.

This package provides core functionality including logging configuration,
database setup, and other shared utilities.
"""

from gearmeshing_ai.core.logging_config import get_logger, setup_logging

__all__ = ["get_logger", "setup_logging"]
