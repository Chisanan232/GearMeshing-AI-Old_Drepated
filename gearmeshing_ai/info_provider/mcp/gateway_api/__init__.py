"""Gateway API client and models.

Exposes a thin HTTP client for the MCP Gateway service and re-exports
domain/DTO models from the `gateway_api.models` subpackage for convenience.
"""

from .client import GatewayApiClient
from .errors import GatewayApiError
from .models import (
    GatewayServer,
    GatewayServerCreate,
    GatewayTransport,
)

__all__ = [
    "GatewayApiClient",
    "GatewayServer",
    "GatewayServerCreate",
    "GatewayTransport",
    "GatewayApiError",
]
