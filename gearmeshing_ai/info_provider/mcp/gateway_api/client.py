"""Gateway management API client.

Thin, focused HTTP client for interacting with the MCP Gateway's management API
(list/get/create servers). This client does not perform tool invocations; it is
used by strategies to discover or register servers and propagate auth.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional
import os
import subprocess
import sys

import httpx

from .errors import GatewayApiError


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
        token_provider: Optional[Callable[[], str]] = None,
        timeout: float = 10.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self._token_provider = token_provider
        self._client = client or httpx.Client(timeout=timeout, follow_redirects=True)
        self._logger = logging.getLogger(__name__)

        # ----------------------
        # Admin endpoints
        # ----------------------
        self._admin = _AdminNamespace(self)

    def _ensure_token(self) -> None:
        """Ensure an auth token is available by invoking the token provider if set."""
        if self._token_provider is not None:
            try:
                self.auth_token = self._token_provider()
            except Exception as e:
                self._logger.warning("GatewayApiClient token_provider failed: %s", e)

    def _headers(self) -> dict[str, str]:
        """Build standard JSON headers and include Authorization when provided.

        Returns:
            A dictionary with `Content-Type` and optional `Authorization` header.
        """
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = self.auth_token
        return headers

    @staticmethod
    def basic_auth_token(username: str, password: str) -> str:
        """Generate a Basic auth token string for Authorization header.

        Returns a string like "Basic base64(user:pass)".
        """
        import base64

        raw = f"{username}:{password}".encode("utf-8")
        return "Basic " + base64.b64encode(raw).decode("ascii")

    @staticmethod
    def generate_bearer_token(jwt_secret_key: Optional[str] = None, *, extra_env: Optional[Dict[str, str]] = None, timeout: float = 5.0) -> str:
        """Generate a Bearer JWT using mcpgateway.utils.create_jwt_token.

        Args:
            jwt_secret_key: Optional secret key. If not provided, uses env JWT_SECRET_KEY.
            extra_env: Optional env vars to inject for the token generation process.
            timeout: Subprocess timeout seconds.

        Returns:
            Authorization header value like: "Bearer <jwt>".
        """
        env = os.environ.copy()
        if jwt_secret_key is not None:
            env["JWT_SECRET_KEY"] = jwt_secret_key
        if extra_env:
            env.update(extra_env)
        try:
            out = subprocess.check_output([sys.executable, "-m", "mcpgateway.utils.create_jwt_token"], env=env, timeout=timeout)
        except Exception as e:
            raise GatewayApiError(f"Failed to generate JWT via mcpgateway: {e}") from e
        token = out.decode("utf-8").strip()
        return f"Bearer {token}"

    @property
    def admin(self) -> "_AdminNamespace":
        """Namespaced admin API access: mcp_registry, gateway, tools."""
        return self._admin

    # Backward-compatible admin_* wrappers have been removed in favor of the
    # descriptive namespaced interface available via `client.admin`.

    def health(self) -> Dict[str, Any]:
        """GET /health to check MCP Gateway service health.

        Returns the JSON payload from the health endpoint. Raises GatewayApiError
        on non-success responses.
        """
        try:
            self._ensure_token()
            r = self._client.get(f"{self.base_url}/health", headers=self._headers())
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise GatewayApiError(
                f"Gateway health failed: {e.response.status_code}",
                status_code=e.response.status_code,
                details=e.response.text,
            ) from e
        try:
            return r.json()
        except Exception:
            return {"status": r.text}


class _McpRegistryNamespace:
    def __init__(self, client: GatewayApiClient) -> None:
        self._client = client

    def list(
        self,
        include_inactive: Optional[bool] = None,
        tags: Optional[str] = None,
        team_id: Optional[str] = None,
        visibility: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._client._ensure_token()
        params: Dict[str, Any] = {}
        if include_inactive is not None:
            params["include_inactive"] = str(include_inactive).lower()
        if tags is not None:
            params["tags"] = tags
        if team_id is not None:
            params["team_id"] = team_id
        if visibility is not None:
            params["visibility"] = visibility
        r = self._client._client.get(
            f"{self._client.base_url}/admin/mcp-registry/servers",
            headers=self._client._headers(),
            params=params or None,
        )
        r.raise_for_status()
        return r.json()

    def register(self, server_id: str) -> Dict[str, Any]:
        self._client._ensure_token()
        r = self._client._client.post(
            f"{self._client.base_url}/admin/mcp-registry/{server_id}/register",
            headers=self._client._headers(),
        )
        r.raise_for_status()
        return r.json()


class _GatewayMgmtNamespace:
    def __init__(self, client: GatewayApiClient) -> None:
        self._client = client

    def list(self, include_inactive: Optional[bool] = None) -> Dict[str, Any]:
        self._client._ensure_token()
        params: Dict[str, Any] = {}
        if include_inactive is not None:
            params["include_inactive"] = str(include_inactive).lower()
        r = self._client._client.get(
            f"{self._client.base_url}/admin/gateways",
            headers=self._client._headers(),
            params=params or None,
        )
        r.raise_for_status()
        return r.json()

    def get(self, gateway_id: str) -> Dict[str, Any]:
        self._client._ensure_token()
        r = self._client._client.get(
            f"{self._client.base_url}/admin/gateways/{gateway_id}", headers=self._client._headers()
        )
        r.raise_for_status()
        return r.json()


class _ToolsNamespace:
    def __init__(self, client: GatewayApiClient) -> None:
        self._client = client

    def list(self, offset: int = 0, limit: int = 50, include_inactive: Optional[bool] = None) -> Dict[str, Any]:
        self._client._ensure_token()
        params: Dict[str, Any] = {"offset": offset, "limit": limit}
        if include_inactive is not None:
            params["include_inactive"] = str(include_inactive).lower()
        r = self._client._client.get(
            f"{self._client.base_url}/admin/tools",
            headers=self._client._headers(),
            params=params,
        )
        r.raise_for_status()
        return r.json()

    def get(self, tool_id: str) -> Dict[str, Any]:
        self._client._ensure_token()
        r = self._client._client.get(
            f"{self._client.base_url}/admin/tools/{tool_id}", headers=self._client._headers()
        )
        r.raise_for_status()
        return r.json()


class _AdminNamespace:
    def __init__(self, client: GatewayApiClient) -> None:
        self._client = client
        self._mcp_registry = _McpRegistryNamespace(client)
        self._gateway = _GatewayMgmtNamespace(client)
        self._tools = _ToolsNamespace(client)

    @property
    def mcp_registry(self) -> _McpRegistryNamespace:
        return self._mcp_registry

    @property
    def gateway(self) -> _GatewayMgmtNamespace:
        return self._gateway

    @property
    def tools(self) -> _ToolsNamespace:
        return self._tools
