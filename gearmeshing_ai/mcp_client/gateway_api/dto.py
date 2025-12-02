from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import AnyHttpUrl, Field, model_validator

from ..schemas.base import BaseSchema


class ListServersQuery(BaseSchema):
    include_inactive: Optional[bool] = None
    tags: Optional[str] = None
    team_id: Optional[str] = None
    visibility: Optional[str] = None


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


class ServerCreateDTO(BaseSchema):
    name: str
    url: AnyHttpUrl
    transport: str
    auth_token: Optional[str] = None
    tags: Optional[List[str]] = None
    visibility: Optional[str] = None
    team_id: Optional[str] = Field(default=None, alias="teamId")


class ServerCreateResponseDTO(ServerReadDTO):
    pass


class GetServerResponseDTO(ServerReadDTO):
    pass


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
