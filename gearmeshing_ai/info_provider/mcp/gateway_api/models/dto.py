"""Gateway API DTO models

Pydantic models that define the request/response contracts for the Gateway
management API. These DTOs centralize serialization/deserialization and provide
helpers to map into domain models used by the client/strategy layers.

Guidelines:
- Keep alias mappings aligned with the Gateway OpenAPI spec (e.g., teamId, isActive).
- Normalize flexible wire formats into a single structured model (e.g., servers list).
- Prefer validators over ad-hoc parsing in strategies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import AnyHttpUrl, Field, model_validator, ConfigDict, AliasChoices

from ...schemas.base import BaseSchema
from .domain import GatewayServer, GatewayTransport


class ListServersQuery(BaseSchema):
    """Query params for listing servers from the Gateway.

    Use `to_params()` to serialize to HTTP query parameters (booleans are lowercased strings).

    Examples:
        Build and serialize query params for the Gateway list endpoint.

        >>> q = ListServersQuery(include_inactive=True, tags="prod,search", team_id="team-123", visibility="team")
        >>> q.to_params()
        {'includeInactive': 'true', 'tags': 'prod,search', 'teamId': 'team-123', 'visibility': 'team'}

    References:
        - GatewayApiClient.list_servers uses this model's `to_params()` when making requests.
        - OpenAPI: GET /servers (see docs/openapi_spec/mcp_gateway.json)
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
        description="Team identifier to scope results to a single team.",
        examples=["team-123"],
    )
    visibility: Optional[str] = Field(
        default=None,
        description="Visibility filter for servers. Typical values: 'public', 'team', 'private'.",
        examples=["team"],
    )

    def to_params(self) -> dict[str, str]:
        """Serialize the query into HTTP params, excluding None values."""
        data = self.model_dump(exclude_none=True)
        params: dict[str, str] = {}
        for k, v in data.items():
            if isinstance(v, bool):
                params[k] = str(v).lower()
            else:
                params[k] = str(v)
        return params


class ServerReadDTO(BaseSchema):
    """Server resource returned by the Gateway.

    Includes alias mappings for `teamId` → `team_id` and `isActive` → `is_active`.
    Use `to_gateway_server()` to map to the domain model consumed by strategies.

    Examples:
        A typical server object returned by the Gateway API:

        {
            'id': 's1',
            'name': 'github-mcp',
            'url': 'http://underlying/mcp/',
            'transport': 'STREAMABLEHTTP',
            'description': 'Team search MCP server',
            'tags': ['prod', 'search'],
            'visibility': 'team',
            'teamId': 'team-123',
            'isActive': true,
            'metrics': {'latencyMsP50': 30}
        }

    References:
        - Mapped to domain via `ServerReadDTO.to_gateway_server()`.
        - Consumed by `GatewayApiClient.list_servers` and `GatewayApiClient.get_server`.
        - OpenAPI: GET /servers/{serverId} (see docs/openapi_spec/mcp_gateway.json)
    """

    id: str = Field(
        ..., description="Unique identifier of the server within the Gateway.", examples=["s1", "created-123"]
    )
    name: str = Field(
        ..., description="Human-readable name for the server entry.", examples=["github-mcp", "clickup-mcp"]
    )
    url: AnyHttpUrl = Field(
        ...,
        description="Base URL of the underlying MCP server (source server, not Gateway endpoint).",
        examples=["http://underlying/mcp/"],
    )
    transport: str = Field(
        ...,
        description="Transport type used by the Gateway to reach the server. One of 'STREAMABLEHTTP', 'SSE', 'STDIO'.",
        examples=["STREAMABLEHTTP"],
    )
    description: Optional[str] = Field(
        default=None, description="Optional freeform description of the server.", examples=["Team search MCP server"]
    )
    tags: Optional[List[str]] = Field(
        default=None, description="Optional list of tags associated with the server.", examples=[["prod", "search"]]
    )
    visibility: Optional[str] = Field(
        default=None, description="Visibility setting (e.g., 'team', 'public', 'private').", examples=["team"]
    )
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

    Examples:
        Input variants accepted and the resulting normalized structure:

        - Raw list:
          [ {'id': 's1', 'name': 'a', 'url': 'http://u', 'transport': 'SSE'}, ... ]
          → { 'items': [ {'id': 's1', 'name': 'a', 'url': 'http://u', 'transport': 'SSE'}, ... ] }

        - Object with items:
          { 'items': [ {'id': 's1', 'name': 'a', 'url': 'http://u', 'transport': 'SSE'} ] }
          → preserved

        - Object with servers:
          { 'servers': [ {'id': 's1', 'name': 'a', 'url': 'http://u', 'transport': 'SSE'} ] }
          → { 'items': [ {'id': 's1', 'name': 'a', 'url': 'http://u', 'transport': 'SSE'} ] }

    References:
        - Used by `GatewayApiClient.list_servers` to normalize responses.
        - OpenAPI: GET /servers (see docs/openapi_spec/mcp_gateway.json)
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
    """DTO for creating/registering a server in the Gateway.

    Examples:
        JSON payload for creating a server:

        {
            'name': 'clickup-mcp',
            'url': 'http://clickup-mcp:8000/mcp/',
            'transport': 'STREAMABLEHTTP',
            'authToken': 'Bearer ghp_exampletoken',
            'tags': ['prod', 'tasks'],
            'visibility': 'team',
            'teamId': 'team-123'
        }

    References:
        - Sent by `GatewayApiClient.create_server` to the Gateway API.
        - OpenAPI: POST /servers (see docs/openapi_spec/mcp_gateway.json)
    """

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
    id: str
    name: str
    category: str
    url: str
    auth_type: str
    provider: str
    description: str
    requires_api_key: Optional[bool] = False
    secure: Optional[bool] = False
    tags: Optional[List[str]] = None
    transport: Optional[str] = None
    logo_url: Optional[str] = None
    documentation_url: Optional[str] = None
    is_registered: Optional[bool] = False
    is_available: Optional[bool] = True


class CatalogListResponseDTO(BaseSchema):
    """
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
    servers: List[CatalogServerDTO]
    total: int
    categories: List[str]
    auth_types: List[str]
    providers: List[str]
    all_tags: Optional[List[str]] = None


class CatalogServerRegisterResponseDTO(BaseSchema):
    """
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
    success: bool
    server_id: str
    message: str
    error: Optional[str] = None


class CatalogServerStatusResponseDTO(BaseSchema):
    server_id: str
    is_available: bool
    is_registered: bool
    last_checked: Optional[str] = None
    response_time_ms: Optional[float] = None
    error: Optional[str] = None


class CatalogBulkRegisterResponseDTO(BaseSchema):
    successful: List[str]
    failed: List["CatalogRegisterFailureDTO"]
    total_attempted: int
    total_successful: int

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
    server_id: str
    error: str


# -----------------------------
# Admin: Gateways
# -----------------------------


class GatewayCapabilitiesDTO(BaseSchema):
    # Free-form capability map; keep extensible
    model_config = ConfigDict(extra="allow")


class HeaderMapDTO(BaseSchema):
    # Map-like object with arbitrary string keys and string values
    model_config = ConfigDict(extra="allow")


class OAuthConfigDTO(BaseSchema):
    # Typical OAuth 2.0 fields; remain extensible
    grant_type: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    authorization_url: Optional[str] = None
    token_url: Optional[str] = None
    scopes: Optional[List[str]] = None
    redirect_uri: Optional[str] = None
    audience: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class GatewayReadDTO(BaseSchema):
    """
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

    id: Optional[str] = None
    name: str
    url: str
    description: Optional[str] = None
    transport: str = "SSE"
    capabilities: Optional[GatewayCapabilitiesDTO] = None
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None
    enabled: Optional[bool] = True
    reachable: Optional[bool] = True
    lastSeen: Optional[str] = None
    passthroughHeaders: Optional[List[str]] = None
    authType: Optional[str] = None
    authValue: Optional[str] = None
    authHeaders: Optional[List[HeaderMapDTO]] = None
    authHeadersUnmasked: Optional[List[HeaderMapDTO]] = None
    oauthConfig: Optional[OAuthConfigDTO] = None
    authUsername: Optional[str] = None
    authPassword: Optional[str] = None
    authToken: Optional[str] = None
    authHeaderKey: Optional[str] = None
    authHeaderValue: Optional[str] = None
    tags: Optional[List[str]] = None
    authPasswordUnmasked: Optional[str] = None
    authTokenUnmasked: Optional[str] = None
    authHeaderValueUnmasked: Optional[str] = None
    teamId: Optional[str] = None
    team: Optional[str] = None
    ownerEmail: Optional[str] = None
    visibility: Optional[str] = None
    createdBy: Optional[str] = None
    createdFromIp: Optional[str] = None
    createdVia: Optional[str] = None
    createdUserAgent: Optional[str] = None
    modifiedBy: Optional[str] = None
    modifiedFromIp: Optional[str] = None
    modifiedVia: Optional[str] = None
    modifiedUserAgent: Optional[str] = None
    importBatchId: Optional[str] = None
    federationSource: Optional[str] = None
    version: Optional[int] = 1
    slug: Optional[str] = None


# -----------------------------
# Admin: Tools
# -----------------------------


class ToolMetricsDTO(BaseSchema):
    totalExecutions: int
    successfulExecutions: int
    failedExecutions: int
    failureRate: float
    minResponseTime: Optional[float] = None
    maxResponseTime: Optional[float] = None
    avgResponseTime: Optional[float] = None
    lastExecutionTime: Optional[str] = None


class AuthenticationValuesDTO(BaseSchema):
    authType: Optional[str] = None
    authValue: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    authHeaderKey: Optional[str] = None
    authHeaderValue: Optional[str] = None


class JSONSchemaDTO(BaseSchema):
    # Arbitrary JSON schema-like structure
    model_config = ConfigDict(extra="allow")


class FreeformObjectDTO(BaseSchema):
    # Arbitrary JSON object (metadata, annotations, mappings)
    model_config = ConfigDict(extra="allow")


class HeadersDTO(BaseSchema):
    # Arbitrary headers map (string->string)
    model_config = ConfigDict(extra="allow")


class ToolReadDTO(BaseSchema):
    """
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
    id: str
    originalName: str
    url: Optional[str] = None
    description: Optional[str] = None
    requestType: str
    integrationType: str
    headers: Optional[HeadersDTO] = None
    inputSchema: JSONSchemaDTO
    outputSchema: Optional[JSONSchemaDTO] = None
    annotations: Optional[FreeformObjectDTO] = None
    jsonpathFilter: Optional[str] = None
    auth: Optional[AuthenticationValuesDTO] = None
    createdAt: str
    updatedAt: str
    enabled: bool
    reachable: bool
    gatewayId: Optional[str] = None
    executionCount: int
    metrics: ToolMetricsDTO
    name: str
    displayName: Optional[str] = None
    gatewaySlug: str
    customName: str
    customNameSlug: str
    tags: Optional[List[str]] = None
    createdBy: Optional[str] = None
    createdFromIp: Optional[str] = None
    createdVia: Optional[str] = None
    createdUserAgent: Optional[str] = None
    modifiedBy: Optional[str] = None
    modifiedFromIp: Optional[str] = None
    modifiedVia: Optional[str] = None
    modifiedUserAgent: Optional[str] = None
    importBatchId: Optional[str] = None
    federationSource: Optional[str] = None
    version: Optional[int] = 1
    teamId: Optional[str] = None
    team: Optional[str] = None
    ownerEmail: Optional[str] = None
    visibility: Optional[str] = None
    baseUrl: Optional[str] = None
    pathTemplate: Optional[str] = None
    queryMapping: Optional[FreeformObjectDTO] = None
    headerMapping: Optional[FreeformObjectDTO] = None
    timeoutMs: Optional[int] = 20000
    exposePassthrough: Optional[bool] = True
    allowlist: Optional[List[str]] = None
    pluginChainPre: Optional[List[str]] = None
    pluginChainPost: Optional[List[str]] = None
    meta: Optional[FreeformObjectDTO] = Field(
        default=None,
        validation_alias=AliasChoices("_meta", "meta"),
        serialization_alias="_meta",
    )


class PaginationDTO(BaseSchema):
    # Page info; keep extensible
    page: Optional[int] = None
    perPage: Optional[int] = None
    total: Optional[int] = None
    totalPages: Optional[int] = None
    model_config = ConfigDict(extra="allow")


class LinksDTO(BaseSchema):
    self: Optional[str] = None
    next: Optional[str] = None
    prev: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class AdminToolsListResponseDTO(BaseSchema):
    """
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
    data: Optional[List[ToolReadDTO]] = None
    pagination: Optional[PaginationDTO] = None
    links: Optional[LinksDTO] = None
