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

from pydantic import AnyHttpUrl, Field, model_validator

from gearmeshing_ai.mcp_client.schemas.base import BaseSchema

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
