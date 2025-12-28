"""
Logfire Middleware for FastAPI.

This middleware provides automatic tracing and monitoring of FastAPI requests
using Pydantic AI Logfire, including:
- Request/response logging
- Performance metrics
- Error tracking
- Request context propagation
"""

import time
from typing import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from gearmeshing_ai.core.logging_config import get_logger
from gearmeshing_ai.core.monitoring import log_api_request

logger = get_logger(__name__)


class LogfireMiddleware(BaseHTTPMiddleware):
    """Middleware for tracing API requests with Logfire."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and log metrics.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware/handler

        Returns:
            The HTTP response
        """
        # Record start time
        start_time = time.time()

        # Extract request information
        method = request.method
        path = request.url.path
        query_string = request.url.query

        # Add request context to logs
        request.state.start_time = start_time
        request.state.method = method
        request.state.path = path

        try:
            # Process the request
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Log the request
            log_api_request(
                method=method,
                path=path,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )

            # Add performance header
            response.headers["X-Process-Time"] = str(duration_ms)

            # Log slow requests
            if duration_ms > 1000:  # More than 1 second
                logger.warning(
                    f"Slow API request: {method} {path} took {duration_ms:.2f}ms",
                    extra={
                        "method": method,
                        "path": path,
                        "duration_ms": duration_ms,
                        "status_code": response.status_code,
                    },
                )

            return response

        except Exception as e:
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Log the error
            logger.error(
                f"API request failed: {method} {path}",
                exc_info=True,
                extra={
                    "method": method,
                    "path": path,
                    "duration_ms": duration_ms,
                    "error": str(e),
                },
            )

            # Log to Logfire
            log_api_request(
                method=method,
                path=path,
                status_code=500,
                duration_ms=duration_ms,
            )

            raise
