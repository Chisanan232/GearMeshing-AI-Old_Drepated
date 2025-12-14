"""Gateway API DTO models

Overview
--------
Pydantic DTOs for the MCP Gateway management API, closely aligned with the
OpenAPI spec in ``docs/openapi_spec/mcp_gateway.json`` and the interactive API
docs at ``http://127.0.0.1:4444/docs``. These models centralize the contracts
for request/response payloads so that higher layers (clients/strategies) can be
simple, robust, and type-safe.

Design guidelines
-----------------
- Keep field names and aliases consistent with wire schema (e.g., ``teamId``, ``isActive``).
- Normalize flexible list/object shapes with validators at the DTO layer.
- Use permissive models (``extra=allow``) for provider-specific or evolving areas.
- Prefer DTO-driven parsing over ad-hoc dict manipulation in strategies.

Endpoint mapping (selected)
---------------------------
- ``GET /admin/mcp-registry/servers`` → ``CatalogListResponseDTO``
- ``POST /admin/mcp-registry/{server_id}/register`` → ``CatalogServerRegisterResponseDTO``
- ``GET /admin/gateways`` → ``List[GatewayReadDTO]`` (or normalized from object-with-items)
- ``GET /admin/gateways/{gateway_id}`` → ``GatewayReadDTO``
- ``GET /admin/tools`` → ``AdminToolsListResponseDTO``
- ``GET /admin/tools/{tool_id}`` → ``ToolReadDTO``
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, AnyHttpUrl, ConfigDict, Field, model_validator

from ...schemas.base import BaseSchema
from .domain import GatewayServer, GatewayTransport


# -----------------------------
# Admin: Servers (Gateway-managed)
# -----------------------------


class ListServersQuery(BaseSchema):
    """Query params for listing servers from the Gateway.

    Use `to_params()` to serialize to HTTP query parameters (booleans are lowercased strings).

    Examples:
        Build and serialize query params for the Gateway list endpoint.

        >>> q = ListServersQuery(include_inactive=True, tags="prod,search", team_id="team-123", visibility="team")
        >>> q.to_params()
        {'include_inactive': 'true', 'tags': 'prod,search', 'team_id': 'team-123', 'visibility': 'team'}

    References:
        - Used by clients to build query strings for catalog list endpoints.
        - OpenAPI: GET /admin/mcp-registry/servers (see docs/openapi_spec/mcp_gateway.json)
    """

    include_inactive: Optional[bool] = Field(
        default=None,
        description="If true, include inactive servers in results. If false, only active servers. If omitted, server default applies.",
        examples=[True, False],
    )
    tags: Optional[str] = Field(
        default=None,
        description="Comma-separated list of tags to filter servers by (logical OR). e.g., 'prod,search'.",
        examples=["prod,search"],
    )
    team_id: Optional[str] = Field(
        default=None,
        description="Team ID scope for the results.",
        examples=["team-123"],
    )
    visibility: Optional[str] = Field(
        default=None,
        description="Visibility filter (e.g., public/team/private).",
        examples=["team"],
    )

    def to_params(self) -> Dict[str, str]:
        params: Dict[str, str] = {}
        if self.include_inactive is not None:
            params["include_inactive"] = str(self.include_inactive).lower()
        if self.tags is not None:
            params["tags"] = self.tags
        if self.team_id is not None:
            params["team_id"] = self.team_id
        if self.visibility is not None:
            params["visibility"] = self.visibility
        return params


class ServerReadDTO(BaseSchema):
    """Single Gateway-managed server entry as returned by admin APIs.

    Fields align with the Gateway schema and include alias support for `teamId` and `isActive`.
    """

    id: str = Field(..., description="Gateway server identifier.")
    name: str = Field(..., description="Human-readable name of the server.")
    url: AnyHttpUrl = Field(..., description="Underlying MCP server base URL configured in the Gateway.")
    transport: GatewayTransport = Field(..., description="Transport used to reach the underlying MCP server.")
    description: Optional[str] = Field(None, description="Optional description of the server entry.")
    tags: Optional[List[str]] = Field(None, description="Tags associated with the server entry.")
    visibility: Optional[str] = Field(None, description="Visibility (public/team/private).")
    team_id: Optional[str] = Field(
        default=None,
        alias="teamId",
        description="Owning team identifier, if applicable.",
        examples=["team-123"],
    )
    is_active: Optional[bool] = Field(
        default=None,
        alias="isActive",
        description="Whether the server is currently active in the Gateway.",
        examples=[True],
    )
    metrics: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metrics object reported by the Gateway for this server."
    )

    def to_gateway_server(self) -> GatewayServer:
        """Map this DTO to the `GatewayServer` domain model."""
        return GatewayServer(
            id=self.id,
            name=self.name,
            url=self.url,
            transport=self.transport,
            description=self.description,
            tags=self.tags,
            visibility=self.visibility,
            team_id=self.team_id,
            is_active=self.is_active,
            metrics=self.metrics,
        )


class ServersListPayloadDTO(BaseSchema):
    """Normalized list payload for Gateway servers.

    Accepts multiple wire shapes and normalizes to `{items: [...]}`:
    - list → `{items: [...]}`
    - `{items: [...]}` preserved
    - `{servers: [...]}` → `{items: [...]}`
    """

    items: List[ServerReadDTO] = Field(
        ..., description="Normalized list of servers returned by the Gateway list endpoints."
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, v):
        """Coerce supported wire shapes into the normalized `{items: [...]}` form."""
        if isinstance(v, list):
            return {"items": v}
        if isinstance(v, dict):
            if isinstance(v.get("items"), list):
                return v
            if isinstance(v.get("servers"), list):
                nv = dict(v)
                nv["items"] = nv.pop("servers")
                return nv
        return v


class GatewayServerCreate(BaseSchema):
    """DTO for creating/registering a server in the Gateway."""

    name: str = Field(
        ...,
        description="Desired human-readable name for the server inside the Gateway.",
        min_length=1,
        max_length=128,
        examples=["clickup-mcp"],
    )
    url: AnyHttpUrl = Field(
        ...,
        description="Base URL of the MCP server to be registered in the Gateway.",
        examples=["http://clickup-mcp:8000/mcp/"],
    )
    transport: GatewayTransport = Field(
        ...,
        description="Transport used to connect the Gateway to the underlying MCP server.",
        examples=[GatewayTransport.STREAMABLE_HTTP],
    )
    auth_token: Optional[str] = Field(
        None,
        description="Optional token the Gateway should use when calling the underlying server.",
        min_length=1,
        max_length=512,
        examples=["Bearer ghp_exampletoken"],
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Optional tags to associate with the server upon creation.",
    )
    visibility: Optional[str] = Field(
        default=None,
        description="Desired visibility (e.g., team/private).",
    )
    team_id: Optional[str] = Field(
        default=None,
        description="Team ID to associate with this server.",
        max_length=128,
    )


# -----------------------------
# Admin: Catalog (Registry)
# -----------------------------


class CatalogServerDTO(BaseSchema):
    """Catalog server entry in the MCP registry.

    API
    ---
    Produced by Catalog list endpoints and used to render discovery UIs.
    Typical fields include category, provider, transport, and whether the
    server is already registered/available in the current Gateway.

    References:
      - OpenAPI: components.schemas.CatalogServer
      - Endpoint: ``GET /admin/mcp-registry/servers``
    """

    id: str = Field(..., description="Catalog identifier (unique within registry).", examples=["clickup"])
    name: str = Field(..., description="Human-readable server name.", examples=["clickup"])
    category: str = Field(..., description="Server category for faceting/filtering.", examples=["Utilities"])
    url: str = Field(
        ...,
        description="Gateway endpoint or connection URL for this catalog entry.",
        examples=["http://clickup-mcp:8082/sse/sse"],
    )
    auth_type: str = Field(
        ..., description="Authentication type required by this server (e.g., Open, Bearer, Basic).", examples=["Open"]
    )
    provider: str = Field(..., description="Provider name for catalog grouping.", examples=["E2E"])
    description: str = Field(..., description="Brief description of the server.", examples=["Project management tool"])
    requires_api_key: Optional[bool] = Field(False, description="True if server requires an API key to operate.")
    secure: Optional[bool] = Field(False, description="True if transport is considered secure (TLS, etc.).")
    tags: Optional[List[str]] = Field(None, description="Associated tags for search and grouping.")
    transport: Optional[str] = Field(
        None, description="Preferred transport type (e.g., SSE, STREAMABLEHTTP).", examples=["SSE"]
    )
    logo_url: Optional[str] = Field(None, description="URL to a logo image if available.")
    documentation_url: Optional[str] = Field(None, description="URL to provider documentation if available.")
    is_registered: Optional[bool] = Field(
        False, description="True if this server is already registered in the Gateway."
    )
    is_available: Optional[bool] = Field(True, description="True if the server is currently available.")


class CatalogListResponseDTO(BaseSchema):
    """Paginated catalog/registry listing result.

    API
    ---
    - Method/Path: ``GET /admin/mcp-registry/servers``
    - Query: ``include_inactive``, ``tags``, ``team_id``, ``visibility``

    Fields
    ------
    - ``servers``: list of ``CatalogServerDTO``
    - ``total``: total matching servers
    - Faceting fields (``categories``, ``auth_types``, ``providers``, ``all_tags``)

    Example:
        ```json
        {
           "servers":[
              {
                 "id":"clickup",
                 "name":"clickup",
                 "category":"Utilities",
                 "url":"http://clickup-mcp:8082/sse/sse",
                 "auth_type":"Open",
                 "provider":"E2E",
                 "description":"Project management tool",
                 "requires_api_key":false,
                 "secure":false,
                 "tags":[
                    "project-management",
                    "clickup",
                    "python"
                 ],
                 "transport":"SSE",
                 "logo_url":null,
                 "documentation_url":null,
                 "is_registered":false,
                 "is_available":true
              }
           ],
           "total":1,
           "categories":[
              "Utilities"
           ],
           "auth_types":[
              "Open"
           ],
           "providers":[
              "E2E"
           ],
           "all_tags":[
              "clickup",
              "project-management",
              "python"
           ]
        }
        ```
    """

    servers: List[CatalogServerDTO] = Field(..., description="List of catalog servers matching the query.")
    total: int = Field(..., description="Total number of servers that match the filter criteria.")
    categories: List[str] = Field(..., description="Available categories for faceting.")
    auth_types: List[str] = Field(..., description="Available authentication types for faceting.")
    providers: List[str] = Field(..., description="Available providers for faceting.")
    all_tags: Optional[List[str]] = Field(None, description="All tags seen in the result set for faceting.")


class CatalogServerRegisterResponseDTO(BaseSchema):
    """Response for catalog server registration.

    API
    ---
    - Method/Path: ``POST /admin/mcp-registry/{server_id}/register``

    Behavior
    --------
    Triggers discovery and federation of the underlying MCP server into the Gateway.
    The message usually contains a summary (e.g., tool count discovered).

    Example:
        ```json
        {
           "success":true,
           "server_id":"61a50681abf24f008cf849f857484b12",
           "message":"Successfully registered clickup with 29 tools discovered",
           "error":null
        }
        ```
    """

    success: bool = Field(..., description="True if registration succeeded.")
    server_id: str = Field(..., description="The Gateway-assigned server id after registration.")
    message: str = Field(..., description="Human-readable summary of the registration outcome.")
    error: Optional[str] = Field(None, description="Error message when registration fails.")


class CatalogServerStatusResponseDTO(BaseSchema):
    """Status for a catalog server entry.

    Indicates availability, registration state, and diagnostics.
    """

    server_id: str = Field(..., description="Server identifier in the catalog.")
    is_available: bool = Field(..., description="Current availability of the catalog server.")
    is_registered: bool = Field(..., description="Whether the server has been registered in the Gateway.")
    last_checked: Optional[str] = Field(None, description="Timestamp of last health check.")
    response_time_ms: Optional[float] = Field(None, description="Observed response time during last check (ms).")
    error: Optional[str] = Field(None, description="Diagnostic error message if last check failed.")


class CatalogBulkRegisterResponseDTO(BaseSchema):
    """Bulk registration result for multiple catalog servers.

    Notes
    -----
    The ``failed`` field is normalized from an array of free-form objects into a
    list of ``{server_id, error}`` entries in the pre-validation step.
    """

    successful: List[str] = Field(..., description="List of server ids that were registered successfully.")
    failed: List["CatalogRegisterFailureDTO"] = Field(..., description="List of failures with server ids and reasons.")
    total_attempted: int = Field(..., description="Number of servers attempted to register.")
    total_successful: int = Field(..., description="Number of successful registrations.")

    @model_validator(mode="before")
    @classmethod
    def _coerce_failed(cls, v):
        # API returns failed as a list of objects with additionalProperties: string
        # We normalize into list of {server_id, error}
        if isinstance(v, dict) and isinstance(v.get("failed"), list):
            failed_list = []
            for item in v["failed"]:
                if isinstance(item, dict):
                    for k, err in item.items():
                        failed_list.append({"server_id": k, "error": err})
            v = {**v, "failed": failed_list}
        return v


class CatalogRegisterFailureDTO(BaseSchema):
    """Entry describing a single failure during bulk registration."""

    server_id: str = Field(..., description="Server identifier for which registration failed.")
    error: str = Field(..., description="Reason for failure.")


# -----------------------------
# Admin: Gateways
# -----------------------------


class GatewayCapabilitiesDTO(BaseSchema):
    """Free-form capability map for a Gateway instance.

    The Gateway may expose feature flags or capability subtrees for prompts,
    resources, tools, and experimental areas. The shape can evolve, so the
    model is intentionally permissive.
    """

    model_config = ConfigDict(extra="allow")


class HeaderMapDTO(BaseSchema):
    """Arbitrary header mapping (string → string)."""

    model_config = ConfigDict(extra="allow")


class OAuthConfigDTO(BaseSchema):
    """OAuth 2.0 configuration container.

    Contains well-known OAuth parameters but remains extensible to support
    provider-specific fields.
    """

    grant_type: Optional[str] = Field(
        None, description="OAuth grant type (e.g., client_credentials, authorization_code)."
    )
    client_id: Optional[str] = Field(None, description="OAuth client id.")
    client_secret: Optional[str] = Field(None, description="OAuth client secret (if applicable).")
    authorization_url: Optional[str] = Field(None, description="Authorization endpoint URL.")
    token_url: Optional[str] = Field(None, description="Token endpoint URL.")
    scopes: Optional[List[str]] = Field(None, description="Requested scopes for the token.")
    redirect_uri: Optional[str] = Field(None, description="Redirect URI registered with the provider.")
    audience: Optional[str] = Field(None, description="Intended audience for the token (if required).")
    model_config = ConfigDict(extra="allow")


class GatewayReadDTO(BaseSchema):
    """
    Gateway instance representation as returned by admin endpoints.

    API
    ---
    - Method/Path: ``GET /admin/gateways`` and ``GET /admin/gateways/{gateway_id}``

    Fields
    ------
    - Core: ``id``, ``name``, ``url``, ``description``, ``transport``
    - State: ``enabled``, ``reachable``, ``lastSeen``
    - Auth: ``authType``, ``authValue``, headers/user credentials and OAuth config
    - Audit: created/modified metadata, ownership, visibility
    - Tags and capability maps

    Example:
        ```json
        {
           "id":"61a50681abf24f008cf849f857484b12",
           "name":"clickup",
           "url":"http://clickup-mcp:8082/sse/sse",
           "description":"Project management tool",
           "transport":"SSE",
           "capabilities":{
              "experimental":{

              },
              "prompts":{
                 "listChanged":false
              },
              "resources":{
                 "subscribe":false,
                 "listChanged":false
              },
              "tools":{
                 "listChanged":false
              }
           },
           "createdAt":"2025-12-13T11:41:30.487339",
           "updatedAt":"2025-12-13T11:41:30.487342",
           "enabled":true,
           "reachable":true,
           "lastSeen":"2025-12-13T11:41:30.484488",
           "passthroughHeaders":null,
           "authType":null,
           "authValue":null,
           "authHeaders":null,
           "authHeadersUnmasked":null,
           "oauthConfig":null,
           "authUsername":null,
           "authPassword":null,
           "authToken":null,
           "authHeaderKey":null,
           "authHeaderValue":null,
           "tags":[
              "project-management",
              "clickup",
              "python"
           ],
           "authPasswordUnmasked":null,
           "authTokenUnmasked":null,
           "authHeaderValueUnmasked":null,
           "teamId":null,
           "team":null,
           "ownerEmail":null,
           "visibility":"public",
           "createdBy":null,
           "createdFromIp":null,
           "createdVia":"catalog",
           "createdUserAgent":null,
           "modifiedBy":null,
           "modifiedFromIp":null,
           "modifiedVia":null,
           "modifiedUserAgent":null,
           "importBatchId":null,
           "federationSource":null,
           "version":1,
           "slug":"clickup"
        }
        ```
    """

    id: Optional[str] = Field(None, description="Gateway identifier.")
    name: str = Field(..., description="Gateway name (human-readable).")
    url: str = Field(..., description="Gateway connection URL (e.g., SSE endpoint).")
    description: Optional[str] = Field(None, description="Description of the Gateway instance.")
    transport: str = Field("SSE", description="Transport type used by the Gateway (e.g., SSE).")
    capabilities: Optional[GatewayCapabilitiesDTO] = Field(None, description="Feature flags and capability subtrees.")
    createdAt: Optional[str] = Field(None, description="Creation timestamp (ISO 8601).")
    updatedAt: Optional[str] = Field(None, description="Last update timestamp (ISO 8601).")
    enabled: Optional[bool] = Field(True, description="Whether the Gateway is enabled.")
    reachable: Optional[bool] = Field(True, description="Whether the Gateway is currently reachable.")
    lastSeen: Optional[str] = Field(None, description="Last observed active timestamp.")
    passthroughHeaders: Optional[List[str]] = Field(
        None, description="Header names to pass through to upstream services."
    )
    authType: Optional[str] = Field(None, description="Authentication type configured for the Gateway.")
    authValue: Optional[str] = Field(None, description="Authentication value (masked).")
    authHeaders: Optional[List[HeaderMapDTO]] = Field(None, description="Configured authentication headers (masked).")
    authHeadersUnmasked: Optional[List[HeaderMapDTO]] = Field(None, description="Authentication headers (unmasked).")
    oauthConfig: Optional[OAuthConfigDTO] = Field(None, description="OAuth configuration, when applicable.")
    authUsername: Optional[str] = Field(None, description="Username for basic auth, if configured.")
    authPassword: Optional[str] = Field(None, description="Password for basic auth (masked).")
    authToken: Optional[str] = Field(None, description="Bearer token (masked).")
    authHeaderKey: Optional[str] = Field(None, description="Custom authorization header key, if used.")
    authHeaderValue: Optional[str] = Field(None, description="Custom authorization header value (masked).")
    tags: Optional[List[str]] = Field(None, description="Tags associated with this Gateway instance.")
    authPasswordUnmasked: Optional[str] = Field(None, description="Password (unmasked) – privileged view.")
    authTokenUnmasked: Optional[str] = Field(None, description="Token (unmasked) – privileged view.")
    authHeaderValueUnmasked: Optional[str] = Field(None, description="Custom auth header value (unmasked).")
    teamId: Optional[str] = Field(None, description="Owning team identifier, if applicable.")
    team: Optional[str] = Field(None, description="Owning team name or reference, if available.")
    ownerEmail: Optional[str] = Field(None, description="Owner email address, if tracked.")
    visibility: Optional[str] = Field(None, description="Visibility setting (e.g., public/team/private).")
    createdBy: Optional[str] = Field(None, description="Creator identity, if tracked.")
    createdFromIp: Optional[str] = Field(None, description="Origin IP for creation, if tracked.")
    createdVia: Optional[str] = Field(None, description="Channel/feature used for creation (e.g., catalog).")
    createdUserAgent: Optional[str] = Field(None, description="User agent observed at creation time.")
    modifiedBy: Optional[str] = Field(None, description="Last modifier identity, if tracked.")
    modifiedFromIp: Optional[str] = Field(None, description="Origin IP for last modification, if tracked.")
    modifiedVia: Optional[str] = Field(None, description="Channel/feature used for modification.")
    modifiedUserAgent: Optional[str] = Field(None, description="User agent observed at last modification.")
    importBatchId: Optional[str] = Field(None, description="Import batch identifier, if federated.")
    federationSource: Optional[str] = Field(None, description="Source of federation/import (e.g., catalog).")
    version: Optional[int] = Field(1, description="Version number for the Gateway entity.")
    slug: Optional[str] = Field(None, description="URL/path-safe slug representing the Gateway.")


# -----------------------------
# Admin: Tools
# -----------------------------


class ToolMetricsDTO(BaseSchema):
    """Aggregated execution metrics for a Gateway-federated tool.

    Values may be ``null`` when not yet executed or when the deployment does
    not collect the metric.
    """

    totalExecutions: int = Field(..., description="Total number of executions observed.")
    successfulExecutions: int = Field(..., description="Total number of successful executions.")
    failedExecutions: int = Field(..., description="Total number of failed executions.")
    failureRate: float = Field(..., description="Failure rate as a fraction (0.0 - 1.0).")
    minResponseTime: Optional[float] = Field(None, description="Minimum observed response time (ms).")
    maxResponseTime: Optional[float] = Field(None, description="Maximum observed response time (ms).")
    avgResponseTime: Optional[float] = Field(None, description="Average observed response time (ms).")
    lastExecutionTime: Optional[str] = Field(None, description="Timestamp of most recent execution.")


class AuthenticationValuesDTO(BaseSchema):
    """Authentication material attached to a tool or Gateway entry.

    Depending on ``authType``, different fields may be populated, such as
    ``token`` (for bearer), ``username``/``password`` (basic), or custom headers.
    """

    authType: Optional[str] = Field(None, description="Authentication type (e.g., Bearer, Basic, Custom).")
    authValue: Optional[str] = Field(None, description="Primary auth value (masked).")
    username: Optional[str] = Field(None, description="Username for Basic auth, if applicable.")
    password: Optional[str] = Field(None, description="Password for Basic auth (masked).")
    token: Optional[str] = Field(None, description="Bearer token (masked).")
    authHeaderKey: Optional[str] = Field(None, description="Custom auth header key, when using header-based auth.")
    authHeaderValue: Optional[str] = Field(None, description="Custom auth header value (masked).")


class JSONSchemaDTO(BaseSchema):
    """Arbitrary JSON Schema-like structure for tool I/O definitions.

    The Gateway exposes JSON schemas for tool inputs/outputs. This model keeps
    the structure permissive to accommodate provider-specific schema features.
    """

    model_config = ConfigDict(extra="allow")


class FreeformObjectDTO(BaseSchema):
    """Arbitrary JSON object (metadata, annotations, mappings)."""

    model_config = ConfigDict(extra="allow")


class HeadersDTO(BaseSchema):
    """Arbitrary headers map (string → string)."""

    model_config = ConfigDict(extra="allow")


class ToolReadDTO(BaseSchema):
    """
    Tool definition federated by a Gateway.

    API
    ---
    - Method/Path: ``GET /admin/tools/{tool_id}``
    - Also appears in list responses from ``GET /admin/tools`` under ``data``.

    Fields
    ------
    - Identity: ``id``, ``name``, ``displayName``, ``originalName``, ``gatewaySlug``
    - Schemas: ``inputSchema``, ``outputSchema`` (JSON Schema-like)
    - Execution: ``requestType``, ``integrationType``, ``timeoutMs``, passthrough
    - Auth and headers: ``auth``, ``headers``, mappings
    - Metrics and status: ``metrics``, ``enabled``, ``reachable``
    - Metadata: tags, versions, creation/modification info, and freeform ``_meta``

    Example:
        ```json
        {
           "id":"3b455b4ff78942d6adb41a41b08ed003",
           "originalName":"workspace.list",
           "url":"http://clickup-mcp:8082/sse/sse",
           "description":"List workspaces (teams) the token can access. Use this first to discover team IDs, then call `space.list` to enumerate spaces. HTTP: GET /team. Returns { workspaces: [{ team_id, name }] }.",
           "requestType":"SSE",
           "integrationType":"MCP",
           "headers":null,
           "inputSchema":{
              "properties":{

              },
              "title":"workspace_listArguments",
              "type":"object"
           },
           "outputSchema":{
              "$defs":{
                 "IssueCode":{
                    "enum":[
                       "VALIDATION_ERROR",
                       "PERMISSION_DENIED",
                       "NOT_FOUND",
                       "CONFLICT",
                       "RATE_LIMIT",
                       "TRANSIENT",
                       "INTERNAL"
                    ],
                    "title":"IssueCode",
                    "type":"string"
                 },
                 "ToolIssue":{
                    "description":"Tiny issue object for failures.\n\nKeep token-lean but actionable. Codes are strict.",
                    "examples":[
                       {
                          "code":"RATE_LIMIT",
                          "hint":"Back off and retry",
                          "message":"Rate limit exceeded",
                          "retry_after_ms":1200
                       }
                    ],
                    "properties":{
                       "code":{
                          "$ref":"#/$defs/IssueCode",
                          "description":"Canonical error code"
                       },
                       "message":{
                          "description":"End-user readable short message",
                          "title":"Message",
                          "type":"string"
                       },
                       "hint":{
                          "anyOf":[
                             {
                                "type":"string"
                             },
                             {
                                "type":"null"
                             }
                          ],
                          "default":null,
                          "description":"Optional one-line remediation hint",
                          "title":"Hint"
                       },
                       "retry_after_ms":{
                          "anyOf":[
                             {
                                "minimum":0,
                                "type":"integer"
                             },
                             {
                                "type":"null"
                             }
                          ],
                          "default":null,
                          "description":"Backoff duration in ms (when rate-limited)",
                          "title":"Retry After Ms"
                       }
                    },
                    "required":[
                       "code",
                       "message"
                    ],
                    "title":"ToolIssue",
                    "type":"object"
                 },
                 "WorkspaceListItem":{
                    "description":"Tiny projection for a workspace (team).",
                    "properties":{
                       "team_id":{
                          "description":"Workspace (team) ID",
                          "examples":[
                             "team_1",
                             "9018752317"
                          ],
                          "title":"Team Id",
                          "type":"string"
                       },
                       "name":{
                          "description":"Workspace name",
                          "examples":[
                             "Engineering",
                             "Ops"
                          ],
                          "title":"Name",
                          "type":"string"
                       }
                    },
                    "required":[
                       "team_id",
                       "name"
                    ],
                    "title":"WorkspaceListItem",
                    "type":"object"
                 },
                 "WorkspaceListResult":{
                    "description":"Result for workspace.list tool.",
                    "examples":[
                       {
                          "items":[
                             {
                                "name":"Engineering",
                                "team_id":"team_1"
                             }
                          ]
                       }
                    ],
                    "properties":{
                       "items":{
                          "description":"List of workspaces",
                          "examples":[
                             [
                                {
                                   "name":"Engineering",
                                   "team_id":"team_1"
                                }
                             ]
                          ],
                          "items":{
                             "$ref":"#/$defs/WorkspaceListItem"
                          },
                          "title":"Items",
                          "type":"array"
                       }
                    },
                    "title":"WorkspaceListResult",
                    "type":"object"
                 }
              },
              "examples":[
                 {
                    "issues":[

                    ],
                    "ok":true,
                    "result":null
                 },
                 {
                    "issues":[
                       {
                          "code":"PERMISSION_DENIED",
                          "hint":"Grant the app the required scope",
                          "message":"Missing scope: tasks:write"
                       }
                    ],
                    "ok":false
                 }
              ],
              "properties":{
                 "ok":{
                    "description":"True if the operation succeeded",
                    "title":"Ok",
                    "type":"boolean"
                 },
                 "result":{
                    "anyOf":[
                       {
                          "$ref":"#/$defs/WorkspaceListResult"
                       },
                       {
                          "type":"null"
                       }
                    ],
                    "default":null,
                    "description":"Result payload when ok=true"
                 },
                 "issues":{
                    "description":"Business-level issues",
                    "items":{
                       "$ref":"#/$defs/ToolIssue"
                    },
                    "title":"Issues",
                    "type":"array"
                 }
              },
              "required":[
                 "ok"
              ],
              "title":"ToolResponse[WorkspaceListResult]",
              "type":"object"
           },
           "annotations":{

           },
           "jsonpathFilter":"",
           "auth":null,
           "createdAt":"2025-12-13T11:41:30.494034",
           "updatedAt":"2025-12-13T11:41:30.494035",
           "enabled":true,
           "reachable":true,
           "gatewayId":"61a50681abf24f008cf849f857484b12",
           "executionCount":0,
           "metrics":{
              "totalExecutions":0,
              "successfulExecutions":0,
              "failedExecutions":0,
              "failureRate":0.0,
              "minResponseTime":null,
              "maxResponseTime":null,
              "avgResponseTime":null,
              "lastExecutionTime":null
           },
           "name":"clickup-workspace-list",
           "displayName":"Workspace List",
           "gatewaySlug":"clickup",
           "customName":"workspace.list",
           "customNameSlug":"workspace-list",
           "tags":[

           ],
           "createdBy":"system",
           "createdFromIp":null,
           "createdVia":"federation",
           "createdUserAgent":null,
           "modifiedBy":null,
           "modifiedFromIp":null,
           "modifiedVia":null,
           "modifiedUserAgent":null,
           "importBatchId":null,
           "federationSource":"clickup",
           "version":1,
           "teamId":null,
           "team":null,
           "ownerEmail":null,
           "visibility":"public",
           "baseUrl":null,
           "pathTemplate":null,
           "queryMapping":null,
           "headerMapping":null,
           "timeoutMs":null,
           "exposePassthrough":true,
           "allowlist":null,
           "pluginChainPre":null,
           "pluginChainPost":null,
           "_meta":null
        }
        ```
    """

    id: str = Field(..., description="Tool identifier inside the Gateway.")
    originalName: str = Field(..., description="Original MCP tool name as exposed by the upstream server.")
    url: Optional[str] = Field(None, description="Gateway URL serving this tool.")
    description: Optional[str] = Field(None, description="Description of the tool and its purpose.")
    requestType: str = Field(..., description="Transport type used to execute the tool (e.g., SSE).")
    integrationType: str = Field(..., description="Integration classification (e.g., MCP).")
    headers: Optional[HeadersDTO] = Field(None, description="Custom headers to apply when executing the tool.")
    inputSchema: JSONSchemaDTO = Field(..., description="JSON Schema describing the tool's input parameters.")
    outputSchema: Optional[JSONSchemaDTO] = Field(None, description="JSON Schema for the tool's output payload.")
    annotations: Optional[FreeformObjectDTO] = Field(None, description="Freeform annotations.")
    jsonpathFilter: Optional[str] = Field(None, description="Optional JSONPath filter applied to outputs.")
    auth: Optional[AuthenticationValuesDTO] = Field(None, description="Authentication to use when invoking the tool.")
    createdAt: str = Field(..., description="Creation timestamp (ISO 8601).")
    updatedAt: str = Field(..., description="Last update timestamp (ISO 8601).")
    enabled: bool = Field(..., description="Whether the tool is enabled.")
    reachable: bool = Field(..., description="Whether the tool is reachable.")
    gatewayId: Optional[str] = Field(None, description="Owning Gateway identifier.")
    executionCount: int = Field(..., description="Total observed execution count.")
    metrics: ToolMetricsDTO = Field(..., description="Aggregated execution metrics.")
    name: str = Field(..., description="Stable tool name (slug) within the Gateway.")
    displayName: Optional[str] = Field(None, description="End-user friendly display name.")
    gatewaySlug: str = Field(..., description="Gateway slug where this tool resides.")
    customName: str = Field(..., description="Customized name mapped from upstream, when applicable.")
    customNameSlug: str = Field(..., description="Slugified custom name.")
    tags: Optional[List[str]] = Field(None, description="Tags associated with this tool.")
    createdBy: Optional[str] = Field(None, description="Creator identity, if tracked.")
    createdFromIp: Optional[str] = Field(None, description="Origin IP for creation, if tracked.")
    createdVia: Optional[str] = Field(None, description="Channel/feature used for creation.")
    createdUserAgent: Optional[str] = Field(None, description="User agent at creation time.")
    modifiedBy: Optional[str] = Field(None, description="Last modifier identity, if tracked.")
    modifiedFromIp: Optional[str] = Field(None, description="Origin IP for last modification, if tracked.")
    modifiedVia: Optional[str] = Field(None, description="Channel/feature used for modification.")
    modifiedUserAgent: Optional[str] = Field(None, description="User agent observed at last modification.")
    importBatchId: Optional[str] = Field(None, description="Import batch identifier, if federated.")
    federationSource: Optional[str] = Field(None, description="Federation source for this tool.")
    version: Optional[int] = Field(1, description="Version of the tool entity.")
    teamId: Optional[str] = Field(None, description="Owning team id, if applicable.")
    team: Optional[str] = Field(None, description="Owning team name or reference, if available.")
    ownerEmail: Optional[str] = Field(None, description="Owner email address, if tracked.")
    visibility: Optional[str] = Field(None, description="Visibility setting (e.g., public/team/private).")
    baseUrl: Optional[str] = Field(None, description="Base URL used for HTTP-style adapters, if applicable.")
    pathTemplate: Optional[str] = Field(None, description="Path template for HTTP-style adapters, if applicable.")
    queryMapping: Optional[FreeformObjectDTO] = Field(None, description="Mapping for query parameters.")
    headerMapping: Optional[FreeformObjectDTO] = Field(None, description="Mapping for header parameters.")
    timeoutMs: Optional[int] = Field(20000, description="Timeout for tool execution in milliseconds.")
    exposePassthrough: Optional[bool] = Field(True, description="Whether to expose passthrough behavior to clients.")
    allowlist: Optional[List[str]] = Field(None, description="Explicit allowlist of consumers/principals.")
    pluginChainPre: Optional[List[str]] = Field(None, description="Pre-execution plugin chain identifiers.")
    pluginChainPost: Optional[List[str]] = Field(None, description="Post-execution plugin chain identifiers.")
    meta: Optional[FreeformObjectDTO] = Field(
        default=None,
        validation_alias=AliasChoices("_meta", "meta"),
        serialization_alias="_meta",
    )


class PaginationDTO(BaseSchema):
    """Pagination metadata (extensible)."""

    page: Optional[int] = Field(None, description="Current page number (1-based), if pagination is page-based.")
    perPage: Optional[int] = Field(None, description="Items per page, if pagination is page-based.")
    total: Optional[int] = Field(None, description="Total items matching the filter criteria.")
    totalPages: Optional[int] = Field(None, description="Total number of pages, if known.")
    model_config = ConfigDict(extra="allow")


class LinksDTO(BaseSchema):
    """Pagination links (extensible)."""

    self: Optional[str] = Field(None, description="Self link for the current page.")
    next: Optional[str] = Field(None, description="Link to the next page, if available.")
    prev: Optional[str] = Field(None, description="Link to the previous page, if available.")
    model_config = ConfigDict(extra="allow")


class AdminToolsListResponseDTO(BaseSchema):
    """
    Admin tools listing response wrapper.

    API
    ---
    - Method/Path: ``GET /admin/tools``
    - Query: ``offset``, ``limit``, ``include_inactive``

    Fields
    ------
    - ``data``: list of ``ToolReadDTO``
    - ``pagination``: optional page metadata
    - ``links``: optional navigation links

    Example:
        ```json
        {
           "data":[
              {
                 "id":"3b455b4ff78942d6adb41a41b08ed003",
                 "originalName":"workspace.list",
                 "url":"http://clickup-mcp:8082/sse/sse",
                 "description":"List workspaces (teams) the token can access. Use this first to discover team IDs, then call `space.list` to enumerate spaces. HTTP: GET /team. Returns { workspaces: [{ team_id, name }] }.",
                 "requestType":"SSE",
                 "integrationType":"MCP",
                 "headers":null,
                 "inputSchema":{
                    "properties":{

                    },
                    "title":"workspace_listArguments",
                    "type":"object"
                 },
                 "outputSchema":{
                    "$defs":{
                       "IssueCode":{
                          "enum":[
                             "VALIDATION_ERROR",
                             "PERMISSION_DENIED",
                             "NOT_FOUND",
                             "CONFLICT",
                             "RATE_LIMIT",
                             "TRANSIENT",
                             "INTERNAL"
                          ],
                          "title":"IssueCode",
                          "type":"string"
                       },
                       "ToolIssue":{
                          "description":"Tiny issue object for failures.\n\nKeep token-lean but actionable. Codes are strict.",
                          "examples":[
                             {
                                "code":"RATE_LIMIT",
                                "hint":"Back off and retry",
                                "message":"Rate limit exceeded",
                                "retry_after_ms":1200
                             }
                          ],
                          "properties":{
                             "code":{
                                "$ref":"#/$defs/IssueCode",
                                "description":"Canonical error code"
                             },
                             "message":{
                                "description":"End-user readable short message",
                                "title":"Message",
                                "type":"string"
                             },
                             "hint":{
                                "anyOf":[
                                   {
                                      "type":"string"
                                   },
                                   {
                                      "type":"null"
                                   }
                                ],
                                "default":null,
                                "description":"Optional one-line remediation hint",
                                "title":"Hint"
                             },
                             "retry_after_ms":{
                                "anyOf":[
                                   {
                                      "minimum":0,
                                      "type":"integer"
                                   },
                                   {
                                      "type":"null"
                                   }
                                ],
                                "default":null,
                                "description":"Backoff duration in ms (when rate-limited)",
                                "title":"Retry After Ms"
                             }
                          },
                          "required":[
                             "code",
                             "message"
                          ],
                          "title":"ToolIssue",
                          "type":"object"
                       },
                       "WorkspaceListItem":{
                          "description":"Tiny projection for a workspace (team).",
                          "properties":{
                             "team_id":{
                                "description":"Workspace (team) ID",
                                "examples":[
                                   "team_1",
                                   "9018752317"
                                ],
                                "title":"Team Id",
                                "type":"string"
                             },
                             "name":{
                                "description":"Workspace name",
                                "examples":[
                                   "Engineering",
                                   "Ops"
                                ],
                                "title":"Name",
                                "type":"string"
                             }
                          },
                          "required":[
                             "team_id",
                             "name"
                          ],
                          "title":"WorkspaceListItem",
                          "type":"object"
                       },
                       "WorkspaceListResult":{
                          "description":"Result for workspace.list tool.",
                          "examples":[
                             {
                                "items":[
                                   {
                                      "name":"Engineering",
                                      "team_id":"team_1"
                                   }
                                ]
                             }
                          ],
                          "properties":{
                             "items":{
                                "description":"List of workspaces",
                                "examples":[
                                   [
                                      {
                                         "name":"Engineering",
                                         "team_id":"team_1"
                                      }
                                   ]
                                ],
                                "items":{
                                   "$ref":"#/$defs/WorkspaceListItem"
                                },
                                "title":"Items",
                                "type":"array"
                             }
                          },
                          "title":"WorkspaceListResult",
                          "type":"object"
                       }
                    },
                    "examples":[
                       {
                          "issues":[

                          ],
                          "ok":true,
                          "result":null
                       },
                       {
                          "issues":[
                             {
                                "code":"PERMISSION_DENIED",
                                "hint":"Grant the app the required scope",
                                "message":"Missing scope: tasks:write"
                             }
                          ],
                          "ok":false
                       }
                    ],
                    "properties":{
                       "ok":{
                          "description":"True if the operation succeeded",
                          "title":"Ok",
                          "type":"boolean"
                       },
                       "result":{
                          "anyOf":[
                             {
                                "$ref":"#/$defs/WorkspaceListResult"
                             },
                             {
                                "type":"null"
                             }
                          ],
                          "default":null,
                          "description":"Result payload when ok=true"
                       },
                       "issues":{
                          "description":"Business-level issues",
                          "items":{
                             "$ref":"#/$defs/ToolIssue"
                          },
                          "title":"Issues",
                          "type":"array"
                       }
                    },
                    "required":[
                       "ok"
                    ],
                    "title":"ToolResponse[WorkspaceListResult]",
                    "type":"object"
                 },
                 "annotations":{

                 },
                 "jsonpathFilter":"",
                 "auth":null,
                 "createdAt":"2025-12-13T11:41:30.494034",
                 "updatedAt":"2025-12-13T11:41:30.494035",
                 "enabled":true,
                 "reachable":true,
                 "gatewayId":"61a50681abf24f008cf849f857484b12",
                 "executionCount":0,
                 "metrics":{
                    "totalExecutions":0,
                    "successfulExecutions":0,
                    "failedExecutions":0,
                    "failureRate":0.0,
                    "minResponseTime":null,
                    "maxResponseTime":null,
                    "avgResponseTime":null,
                    "lastExecutionTime":null
                 },
                 "name":"clickup-workspace-list",
                 "displayName":"Workspace List",
                 "gatewaySlug":"clickup",
                 "customName":"workspace.list",
                 "customNameSlug":"workspace-list",
                 "tags":[

                 ],
                 "createdBy":"system",
                 "createdFromIp":null,
                 "createdVia":"federation",
                 "createdUserAgent":null,
                 "modifiedBy":null,
                 "modifiedFromIp":null,
                 "modifiedVia":null,
                 "modifiedUserAgent":null,
                 "importBatchId":null,
                 "federationSource":"clickup",
                 "version":1,
                 "teamId":null,
                 "team":null,
                 "ownerEmail":null,
                 "visibility":"public",
                 "baseUrl":null,
                 "pathTemplate":null,
                 "queryMapping":null,
                 "headerMapping":null,
                 "timeoutMs":null,
                 "exposePassthrough":true,
                 "allowlist":null,
                 "pluginChainPre":null,
                 "pluginChainPost":null,
                 "_meta":null
              },
              {
                 "id":"f9db3322ce4b40f4bc586aa56f7857a5",
                 "originalName":"get_authorized_teams",
                 "url":"http://clickup-mcp:8082/sse/sse",
                 "description":"Retrieve all teams/workspaces that the authenticated user has access to.",
                 "requestType":"SSE",
                 "integrationType":"MCP",
                 "headers":null,
                 "inputSchema":{
                    "properties":{

                    },
                    "title":"get_authorized_teamsArguments",
                    "type":"object"
                 },
                 "outputSchema":{
                    "$defs":{
                       "IssueCode":{
                          "enum":[
                             "VALIDATION_ERROR",
                             "PERMISSION_DENIED",
                             "NOT_FOUND",
                             "CONFLICT",
                             "RATE_LIMIT",
                             "TRANSIENT",
                             "INTERNAL"
                          ],
                          "title":"IssueCode",
                          "type":"string"
                       },
                       "ToolIssue":{
                          "description":"Tiny issue object for failures.\n\nKeep token-lean but actionable. Codes are strict.",
                          "examples":[
                             {
                                "code":"RATE_LIMIT",
                                "hint":"Back off and retry",
                                "message":"Rate limit exceeded",
                                "retry_after_ms":1200
                             }
                          ],
                          "properties":{
                             "code":{
                                "$ref":"#/$defs/IssueCode",
                                "description":"Canonical error code"
                             },
                             "message":{
                                "description":"End-user readable short message",
                                "title":"Message",
                                "type":"string"
                             },
                             "hint":{
                                "anyOf":[
                                   {
                                      "type":"string"
                                   },
                                   {
                                      "type":"null"
                                   }
                                ],
                                "default":null,
                                "description":"Optional one-line remediation hint",
                                "title":"Hint"
                             },
                             "retry_after_ms":{
                                "anyOf":[
                                   {
                                      "minimum":0,
                                      "type":"integer"
                                   },
                                   {
                                      "type":"null"
                                   }
                                ],
                                "default":null,
                                "description":"Backoff duration in ms (when rate-limited)",
                                "title":"Retry After Ms"
                             }
                          },
                          "required":[
                             "code",
                             "message"
                          ],
                          "title":"ToolIssue",
                          "type":"object"
                       },
                       "WorkspaceListItem":{
                          "description":"Tiny projection for a workspace (team).",
                          "properties":{
                             "team_id":{
                                "description":"Workspace (team) ID",
                                "examples":[
                                   "team_1",
                                   "9018752317"
                                ],
                                "title":"Team Id",
                                "type":"string"
                             },
                             "name":{
                                "description":"Workspace name",
                                "examples":[
                                   "Engineering",
                                   "Ops"
                                ],
                                "title":"Name",
                                "type":"string"
                             }
                          },
                          "required":[
                             "team_id",
                             "name"
                          ],
                          "title":"WorkspaceListItem",
                          "type":"object"
                       },
                       "WorkspaceListResult":{
                          "description":"Result for workspace.list tool.",
                          "examples":[
                             {
                                "items":[
                                   {
                                      "name":"Engineering",
                                      "team_id":"team_1"
                                   }
                                ]
                             }
                          ],
                          "properties":{
                             "items":{
                                "description":"List of workspaces",
                                "examples":[
                                   [
                                      {
                                         "name":"Engineering",
                                         "team_id":"team_1"
                                      }
                                   ]
                                ],
                                "items":{
                                   "$ref":"#/$defs/WorkspaceListItem"
                                },
                                "title":"Items",
                                "type":"array"
                             }
                          },
                          "title":"WorkspaceListResult",
                          "type":"object"
                       }
                    },
                    "examples":[
                       {
                          "issues":[

                          ],
                          "ok":true,
                          "result":null
                       },
                       {
                          "issues":[
                             {
                                "code":"PERMISSION_DENIED",
                                "hint":"Grant the app the required scope",
                                "message":"Missing scope: tasks:write"
                             }
                          ],
                          "ok":false
                       }
                    ],
                    "properties":{
                       "ok":{
                          "description":"True if the operation succeeded",
                          "title":"Ok",
                          "type":"boolean"
                       },
                       "result":{
                          "anyOf":[
                             {
                                "$ref":"#/$defs/WorkspaceListResult"
                             },
                             {
                                "type":"null"
                             }
                          ],
                          "default":null,
                          "description":"Result payload when ok=true"
                       },
                       "issues":{
                          "description":"Business-level issues",
                          "items":{
                             "$ref":"#/$defs/ToolIssue"
                          },
                          "title":"Issues",
                          "type":"array"
                       }
                    },
                    "required":[
                       "ok"
                    ],
                    "title":"ToolResponse[WorkspaceListResult]",
                    "type":"object"
                 },
                 "annotations":{

                 },
                 "jsonpathFilter":"",
                 "auth":null,
                 "createdAt":"2025-12-13T11:41:30.494031",
                 "updatedAt":"2025-12-13T11:41:30.494031",
                 "enabled":true,
                 "reachable":true,
                 "gatewayId":"61a50681abf24f008cf849f857484b12",
                 "executionCount":0,
                 "metrics":{
                    "totalExecutions":0,
                    "successfulExecutions":0,
                    "failedExecutions":0,
                    "failureRate":0.0,
                    "minResponseTime":null,
                    "maxResponseTime":null,
                    "avgResponseTime":null,
                    "lastExecutionTime":null
                 },
                 "name":"clickup-get-authorized-teams",
                 "displayName":"Get Authorized Teams",
                 "gatewaySlug":"clickup",
                 "customName":"get_authorized_teams",
                 "customNameSlug":"get-authorized-teams",
                 "tags":[

                 ],
                 "createdBy":"system",
                 "createdFromIp":null,
                 "createdVia":"federation",
                 "createdUserAgent":null,
                 "modifiedBy":null,
                 "modifiedFromIp":null,
                 "modifiedVia":null,
                 "modifiedUserAgent":null,
                 "importBatchId":null,
                 "federationSource":"clickup",
                 "version":1,
                 "teamId":null,
                 "team":null,
                 "ownerEmail":null,
                 "visibility":"public",
                 "baseUrl":null,
                 "pathTemplate":null,
                 "queryMapping":null,
                 "headerMapping":null,
                 "timeoutMs":null,
                 "exposePassthrough":true,
                 "allowlist":null,
                 "pluginChainPre":null,
                 "pluginChainPost":null,
                 "_meta":null
              }
           ],
           "pagination":{
              "page":1,
              "per_page":50,
              "total_items":29,
              "total_pages":1,
              "has_next":false,
              "has_prev":false,
              "next_cursor":null,
              "prev_cursor":null
           },
           "links":{
              "self":"/admin/tools?page=1&per_page=50",
              "first":"/admin/tools?page=1&per_page=50",
              "last":"/admin/tools?page=1&per_page=50",
              "next":null,
              "prev":null
           }
        }
        ```
    """

    data: Optional[List[ToolReadDTO]] = Field(None, description="List of tools for the current page.")
    pagination: Optional[PaginationDTO] = Field(None, description="Pagination metadata for the list response.")
    links: Optional[LinksDTO] = Field(None, description="Navigation links for the list response.")
