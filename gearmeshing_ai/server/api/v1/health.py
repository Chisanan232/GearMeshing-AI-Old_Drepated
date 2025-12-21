"""
Health Check Endpoints.

This module provides basic system status endpoints (health, version)
used for monitoring and deployment verification.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get(
    "/health",
    summary="Health Check",
    description="Check the operational status of the API server.",
    response_description="Status object.",
)
async def health_check():
    """
    Health check endpoint.

    Returns a simple status indicator to confirm the server is running and reachable.
    """
    return {"status": "ok"}


@router.get(
    "/version",
    summary="Get Version",
    description="Retrieve version information for the API server.",
    response_description="Version object.",
)
async def version():
    """
    Get API version.

    Returns the current semantic version of the API and supported schema version.
    """
    return {"version": "0.0.1", "schema_version": "v1"}
