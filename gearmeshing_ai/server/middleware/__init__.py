"""
Middleware modules for the GearMeshing-AI server.

This package contains custom middleware for request/response logging,
error handling, and other cross-cutting concerns.
"""

from .logfire_middleware import LogfireMiddleware

__all__ = ["LogfireMiddleware"]
