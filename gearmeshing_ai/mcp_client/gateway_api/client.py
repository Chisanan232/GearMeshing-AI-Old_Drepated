from __future__ import annotations

import logging
from typing import List, Optional

import httpx

from .errors import GatewayApiError, GatewayServerNotFoundError
from .models import GatewayServer, GatewayServerCreate


class GatewayApiClient:
    """
    Thin HTTP client for the IBM Context Forge MCP Gateway management API.

    Responsibilities:
    - list_servers
    - get_server
    - create_server

    Note: This client manages only Gateway metadata. It does not perform MCP tool calls.
    """

    def __init__(
        self,
        base_url: str,
        *,
        auth_token: Optional[str] = None,
        timeout: float = 10.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self._client = client or httpx.Client(timeout=timeout, follow_redirects=True)
        self._logger = logging.getLogger(__name__)

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = self.auth_token
        return headers

    def list_servers(
        self,
        *,
        include_inactive: Optional[bool] = None,
        tags: Optional[str] = None,
        team_id: Optional[str] = None,
        visibility: Optional[str] = None,
    ) -> List[GatewayServer]:
        try:
            params: dict[str, str] = {}
            if include_inactive is not None:
                params["include_inactive"] = str(include_inactive).lower()
            if tags is not None:
                params["tags"] = tags
            if team_id is not None:
                params["team_id"] = team_id
            if visibility is not None:
                params["visibility"] = visibility
            self._logger.debug("GatewayApiClient.list_servers: GET %s/servers params=%s", self.base_url, params)
            r = self._client.get(f"{self.base_url}/servers", headers=self._headers(), params=params or None)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise GatewayApiError(
                f"Gateway list_servers failed: {e.response.status_code}",
                status_code=e.response.status_code,
                details=e.response.text,
            ) from e
        data = r.json()
        items = data if isinstance(data, list) else data.get("items", []) if isinstance(data, dict) else []
        servers: List[GatewayServer] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            servers.append(self._parse_server(item))
        self._logger.debug("GatewayApiClient.list_servers: got %d servers", len(servers))
        return servers

    def get_server(self, server_id: str) -> GatewayServer:
        try:
            self._logger.debug("GatewayApiClient.get_server: GET %s/servers/%s", self.base_url, server_id)
            r = self._client.get(f"{self.base_url}/servers/{server_id}", headers=self._headers())
            if r.status_code == 404:
                raise GatewayServerNotFoundError(server_id)
            r.raise_for_status()
        except GatewayServerNotFoundError:
            raise
        except httpx.HTTPStatusError as e:
            raise GatewayApiError(
                f"Gateway get_server failed: {e.response.status_code}",
                status_code=e.response.status_code,
                details=e.response.text,
            ) from e
        data = r.json()
        if not isinstance(data, dict):
            raise GatewayApiError("Unexpected response shape from get_server", status_code=r.status_code, details=data)
        server = self._parse_server(data)
        self._logger.debug("GatewayApiClient.get_server: resolved id=%s name=%s", server.id, server.name)
        return server

    def create_server(self, payload: GatewayServerCreate) -> GatewayServer:
        try:
            self._logger.debug("GatewayApiClient.create_server: POST %s/servers name=%s", self.base_url, payload.name)
            r = self._client.post(
                f"{self.base_url}/servers",
                headers=self._headers(),
                json=payload.model_dump(by_alias=True, mode="json"),
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise GatewayApiError(
                f"Gateway create_server failed: {e.response.status_code}",
                status_code=e.response.status_code,
                details=e.response.text,
            ) from e
        server = self._parse_server(
            r.json() if r.headers.get("content-type", "application/json").startswith("application/json") else {}
        )
        self._logger.debug("GatewayApiClient.create_server: created id=%s name=%s", server.id, server.name)
        return server

    def _parse_server(self, item: dict) -> GatewayServer:
        # Map only fields we care about; ignore extra fields from ServerRead schema
        subset = {
            "id": item.get("id"),
            "name": item.get("name"),
            "url": item.get("url"),
            "transport": item.get("transport"),
        }
        return GatewayServer.model_validate(subset)
