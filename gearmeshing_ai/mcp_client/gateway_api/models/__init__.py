"""Gateway API models.

Re-exports domain models (e.g., `GatewayServer`, `GatewayTransport`) and
DTOs (e.g., `GatewayServerCreate`, `ListServersQuery`, `ServerReadDTO`,
`ServersListPayloadDTO`) consumed by `GatewayApiClient` and strategy layers.
"""

from gearmeshing_ai.mcp_client.gateway_api.models.domain import (
    GatewayServer,
    GatewayTransport,
)
from gearmeshing_ai.mcp_client.gateway_api.models.dto import (
    GatewayServerCreate,
    ListServersQuery,
    ServerReadDTO,
    ServersListPayloadDTO,
)

__all__ = [
    # DTO models
    "ListServersQuery",
    "ServerReadDTO",
    "ServersListPayloadDTO",
    # Domain models
    "GatewayServer",
    "GatewayServerCreate",
    "GatewayTransport",
]
