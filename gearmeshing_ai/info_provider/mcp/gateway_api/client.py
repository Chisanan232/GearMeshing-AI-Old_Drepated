"""Gateway management API client.

Thin, focused HTTP client for interacting with the MCP Gateway's management API
(list/get/create servers). This client does not perform tool invocations; it is
used by strategies to discover or register servers and propagate auth.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import httpx

from .errors import GatewayApiError, GatewayServerNotFoundError
from .models import (
    GatewayServer,
    GatewayServerCreate,
    ListServersQuery,
    ServerReadDTO,
    ServersListPayloadDTO,
)


class GatewayApiClient:
    """Thin HTTP client for the MCP Gateway management API.

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
        """Build standard JSON headers and include Authorization when provided.

        Returns:
            A dictionary with `Content-Type` and optional `Authorization` header.
        """
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
        """List servers from the Gateway, returning normalized domain models.

        Arguments map to `ListServersQuery`; booleans are serialized as lowercase
        strings per HTTP query conventions.

        Args:
            include_inactive: Include inactive servers if True.
            tags: Optional tags filter string.
            team_id: Optional team filter.
            visibility: Optional visibility filter.

        Returns:
            A list of `GatewayServer` domain objects.

        Raises:
            GatewayApiError: When the Gateway responds with non-success status.
        """
        try:
            q = ListServersQuery(
                include_inactive=include_inactive,
                tags=tags,
                team_id=team_id,
                visibility=visibility,
            )
            params: dict[str, str] = q.to_params()
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
        payload = ServersListPayloadDTO.model_validate(data)
        servers: List[GatewayServer] = []
        for dto in payload.items:
            servers.append(dto.to_gateway_server())
        self._logger.debug("GatewayApiClient.list_servers: got %d servers", len(servers))
        return servers

    def get_server(self, server_id: str) -> GatewayServer:
        """Get a single server by ID, mapping the DTO to the domain model.

        Args:
            server_id: The Gateway server resource ID.

        Returns:
            The `GatewayServer` domain object.

        Raises:
            GatewayServerNotFoundError: When the server responds with 404.
            GatewayApiError: When a non-404 error occurs.
        """
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
        dto = ServerReadDTO.model_validate(data)
        server = dto.to_gateway_server()
        self._logger.debug("GatewayApiClient.get_server: resolved id=%s name=%s", server.id, server.name)
        return server

    def create_server(self, payload: GatewayServerCreate) -> GatewayServer:
        """Create/register a server in the Gateway and return the created resource.

        Args:
            payload: The request payload describing the server to create.

        Returns:
            The created `GatewayServer` resource as a domain object.

        Raises:
            GatewayApiError: When the Gateway responds with non-success status.
        """
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
        raw = r.json() if r.headers.get("content-type", "application/json").startswith("application/json") else {}
        dto = ServerReadDTO.model_validate(raw if isinstance(raw, dict) else {})
        server = dto.to_gateway_server()
        self._logger.debug("GatewayApiClient.create_server: created id=%s name=%s", server.id, server.name)
        return server
