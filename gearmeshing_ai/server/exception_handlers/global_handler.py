"""
Global Exception Handler for FastAPI Application.

This module provides a global exception handler that catches all unhandled
exceptions and logs detailed information including error ID, request context,
and full traceback for debugging purposes.
"""

import traceback
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from gearmeshing_ai.core.logging_config import get_logger

logger = get_logger(__name__)


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler to log detailed error information.
    
    This handler is called for any unhandled exception in the application.
    It logs the full error context and returns a JSON response with an error ID
    that clients can use to reference the error when reporting issues.
    
    Args:
        request: The HTTP request that caused the exception
        exc: The exception that was raised
        
    Returns:
        JSONResponse with error details and error ID
    """
    # Generate unique error ID for tracking
    error_id = id(exc)
    
    # Log the error with full context
    logger.error(
        f'Unhandled exception [{error_id}] in {request.method} {request.url.path}: {str(exc)}',
        exc_info=True,
        extra={
            'error_id': error_id,
            'method': request.method,
            'path': request.url.path,
            'query_params': dict(request.query_params),
            'client': request.client.host if request.client else 'unknown',
            'error_type': type(exc).__name__,
            'traceback': traceback.format_exc(),
        }
    )
    
    # Return error response with error ID
    return JSONResponse(
        status_code=500,
        content={
            'detail': 'Internal server error',
            'error_id': error_id,
            'error_type': type(exc).__name__,
        }
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """
    Register exception handlers with the FastAPI application.
    
    This function should be called during application initialization to set up
    all custom exception handlers.
    
    Args:
        app: The FastAPI application instance
    """
    app.add_exception_handler(Exception, global_exception_handler)
    logger.debug("Exception handlers registered successfully")
