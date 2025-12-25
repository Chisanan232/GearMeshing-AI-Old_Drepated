"""
Exception handlers for the GearMeshing-AI server.

This package contains custom exception handlers for different error types
and a setup function to register them with the FastAPI application.
"""

from .global_handler import setup_exception_handlers

__all__ = ["setup_exception_handlers"]
