from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import AnyHttpUrl, Field, model_validator

from gearmeshing_ai.mcp_client.schemas.base import BaseSchema

from .domain import GatewayServer, GatewayTransport


class ListServersQuery(BaseSchema):
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
        data = self.model_dump(exclude_none=True)
        params: dict[str, str] = {}
        for k, v in data.items():
            if isinstance(v, bool):
                params[k] = str(v).lower()
            else:
                params[k] = str(v)
        return params


class ServerReadDTO(BaseSchema):
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
    items: List[ServerReadDTO] = Field(
        ..., description="Normalized list of servers returned by the Gateway list endpoints."
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, v):
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
