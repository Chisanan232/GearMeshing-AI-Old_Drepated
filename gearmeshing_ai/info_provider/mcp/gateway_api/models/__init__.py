"""Gateway API models.

Re-exports domain models (e.g., `GatewayServer`, `GatewayTransport`) and
DTOs (e.g., `GatewayServerCreate`, `ListServersQuery`, `ServerReadDTO`,
`ServersListPayloadDTO`) consumed by `GatewayApiClient` and strategy layers.
"""

from .domain import (
    GatewayServer,
    GatewayTransport,
)
from .dto import (
    ListServersQuery,
    ServerReadDTO,
    ServersListPayloadDTO,
    CatalogServerDTO,
    CatalogListResponseDTO,
    CatalogServerRegisterResponseDTO,
    CatalogServerStatusResponseDTO,
    CatalogBulkRegisterResponseDTO,
    CatalogRegisterFailureDTO,
    GatewayCapabilitiesDTO,
    HeaderMapDTO,
    OAuthConfigDTO,
    GatewayReadDTO,
    ToolMetricsDTO,
    AuthenticationValuesDTO,
    JSONSchemaDTO,
    FreeformObjectDTO,
    HeadersDTO,
    ToolReadDTO,
    PaginationDTO,
    LinksDTO,
    AdminToolsListResponseDTO,
)

__all__ = [
    # DTO models
    "ListServersQuery",
    "ServerReadDTO",
    "ServersListPayloadDTO",
    # Domain models
    "CatalogServerDTO",
    "CatalogListResponseDTO",
    "CatalogServerRegisterResponseDTO",
    "CatalogServerStatusResponseDTO",
    "CatalogBulkRegisterResponseDTO",
    "CatalogRegisterFailureDTO",
    "GatewayCapabilitiesDTO",
    "HeaderMapDTO",
    "OAuthConfigDTO",
    "GatewayReadDTO",
    "ToolMetricsDTO",
    "AuthenticationValuesDTO",
    "JSONSchemaDTO",
    "FreeformObjectDTO",
    "HeadersDTO",
    "ToolReadDTO",
    "PaginationDTO",
    "LinksDTO",
    "AdminToolsListResponseDTO",
]
