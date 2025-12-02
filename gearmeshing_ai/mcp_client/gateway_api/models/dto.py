from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import AnyHttpUrl, Field, model_validator

from gearmeshing_ai.mcp_client.schemas.base import BaseSchema

from .domain import GatewayServer, GatewayTransport


class ListServersQuery(BaseSchema):
    include_inactive: Optional[bool] = None
    tags: Optional[str] = None
    team_id: Optional[str] = None
    visibility: Optional[str] = None

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
    id: str
    name: str
    url: AnyHttpUrl
    transport: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    visibility: Optional[str] = None
    team_id: Optional[str] = Field(default=None, alias="teamId")
    is_active: Optional[bool] = Field(default=None, alias="isActive")
    metrics: Optional[Dict[str, Any]] = None

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
    items: List[ServerReadDTO]

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
