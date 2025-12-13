"""MCP Gateway Management API client

Overview
--------
Thin, focused HTTP client for the MCP Gateway management API. This client is
primarily used by strategies to discover Gateway-managed MCP servers and tools,
register servers in the Gateway's catalog/registry, and propagate authentication.
It deliberately does not perform MCP tool invocations – those are done via the
official ``mcp`` Python client once a server endpoint is known.

Key features
------------
- Namespaced admin surface under ``client.admin``:
  - ``mcp_registry``: catalog/registry of MCP servers (list/register/status)
  - ``gateway``: list/get Gateway instances
  - ``tools``: list/get tools federated through the Gateway
- Typed DTO responses (Pydantic models), aligning with the OpenAPI spec in
  ``docs/openapi_spec/mcp_gateway.json`` and the interactive docs at
  ``http://127.0.0.1:4444/docs``.
- Optional automatic Bearer token generation using
  ``mcpgateway.utils.create_jwt_token`` for local development.

Authentication
--------------
- Provide ``auth_token`` directly (e.g., ``"Bearer <jwt>"``), or
- Provide a ``token_provider`` callable that returns the token on demand, or
- Enable ``auto_bearer=True`` to spawn ``python -m mcpgateway.utils.create_jwt_token``.
  The secret can be supplied via ``jwt_secret_key`` or environment variable
  ``MCPGATEWAY_JWT_SECRET``. Additional environment variables may be injected via
  ``token_env``.

Errors
------
All HTTP errors are raised as ``GatewayApiError`` with status code and details
where applicable. The ``health`` method returns JSON when possible and falls
back to a text payload.

Usage
-----
Example creating a client and listing catalog servers/tools:

>>> client = GatewayApiClient("http://localhost:4444", auto_bearer=True)
>>> servers = client.admin.mcp_registry.list(include_inactive=True)
>>> tools = client.admin.tools.list(limit=10)

See method docstrings for specific endpoints, HTTP methods, parameters, and
typed return DTOs.
"""

from __future__ import annotations

import logging
import importlib
from typing import Any, Callable, Dict, List, Optional
import os
import subprocess
import sys

import httpx

from .errors import GatewayApiError
from .models.dto import (
    AdminToolsListResponseDTO,
    CatalogListResponseDTO,
    CatalogServerRegisterResponseDTO,
    GatewayReadDTO,
    ToolReadDTO,
)


class GatewayApiClient:
    """Thin HTTP client for the MCP Gateway management API.

    Design
    ------
    - Keeps a small, explicit surface that mirrors the Gateway's admin API.
    - Groups endpoints into namespaces for clarity (``mcp_registry``, ``gateway``, ``tools``).
    - Returns typed DTOs for all responses.
    - Delegates token acquisition to a provider or a local helper for DX.

    Responsibilities
    ----------------
    - Authenticate requests using a static token or a provider.
    - Expose read and registration endpoints of the Gateway.
    - Normalize responses and map them to DTOs.

    This client manages only Gateway metadata. It does not perform MCP tool calls.
    """

    def __init__(
        self,
        base_url: str,
        *,
        auth_token: Optional[str] = None,
        token_provider: Optional[Callable[[], str]] = None,
        timeout: float = 10.0,
        client: Optional[httpx.Client] = None,
        # Auto bearer token generation controls
        auto_bearer: bool = False,
        jwt_secret_key: Optional[str] = None,
        token_env: Optional[Dict[str, str]] = None,
        token_timeout: float = 5.0,
    ) -> None:
        """Create a Gateway API client.

        Args:
            base_url: Base URL of the Gateway service (e.g., ``http://localhost:4444``).
            auth_token: Authorization header value (e.g., ``"Bearer <jwt>"``). If not
                provided, ``token_provider`` or ``auto_bearer`` may be used.
            token_provider: Callable invoked to fetch a token before each request.
            timeout: Default HTTP timeout for the internal client.
            client: Optional preconfigured ``httpx.Client`` to use.
            auto_bearer: If ``True`` and no token is provided, attempt to generate a
                token via ``python -m mcpgateway.utils.create_jwt_token``.
            jwt_secret_key: Secret injected as ``MCPGATEWAY_JWT_SECRET`` when generating
                a token. If omitted, environment is used.
            token_env: Extra environment variables for the token generation subprocess.
            token_timeout: Subprocess timeout for token generation in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self._token_provider = token_provider
        self._client = client or httpx.Client(timeout=timeout, follow_redirects=True)
        self._logger = logging.getLogger(__name__)

        # ----------------------
        # Admin endpoints
        # ----------------------
        self._admin = _AdminNamespace(self)

        # Optionally generate a Bearer token immediately for convenience
        if auth_token is None and token_provider is None and auto_bearer:
            try:
                self.auth_token = self.generate_bearer_token(
                    jwt_secret_key, extra_env=token_env, timeout=token_timeout
                )
            except Exception as e:  # pragma: no cover - logged, not raised
                self._logger.warning("GatewayApiClient auto_bearer failed: %s", e)

    def _ensure_token(self) -> None:
        """Ensure an auth token is available.

        If a ``token_provider`` was supplied, it is invoked prior to making a
        request. Any exception raised by the provider is logged as a warning and
        the request proceeds without altering the token, allowing callers to
        implement soft-fail behavior.
        """
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
    def generate_bearer_token(jwt_secret_key: Optional[str] = None, *, extra_env: Optional[Dict[str, str]] = None, timeout: float = 5.0) -> str:
        """Generate a Bearer JWT via ``mcpgateway.utils.create_jwt_token``.

        Behavior
        --------
        Spawns ``python -m mcpgateway.utils.create_jwt_token`` in a subprocess,
        merging the current environment with ``extra_env`` and injecting
        ``MCPGATEWAY_JWT_SECRET`` when ``jwt_secret_key`` is provided. On success,
        returns a string suitable for the ``Authorization`` header: ``"Bearer <jwt>"``.

        Pre-checks
        ----------
        To provide early, actionable errors, this method first imports
        ``mcpgateway`` and ``mcpgateway.utils.create_jwt_token``. If import fails,
        a ``GatewayApiError`` is raised.

        Args:
            jwt_secret_key: Secret used by the JWT generator. If ``None``, the
                environment variable ``MCPGATEWAY_JWT_SECRET`` must be set by the caller.
            extra_env: Additional environment variables to pass to the subprocess.
            timeout: Subprocess timeout in seconds.

        Returns:
            The value to use as ``Authorization`` header, e.g. ``"Bearer ey..."``.

        Raises:
            GatewayApiError: If the generator is not importable or the subprocess fails.
        """
        # Ensure mcpgateway tooling is importable before attempting to execute it
        try:
            # Import the package and the module used by `-m`
            importlib.import_module("mcpgateway")
            importlib.import_module("mcpgateway.utils.create_jwt_token")
        except Exception as e:
            raise GatewayApiError(
                "mcpgateway.utils.create_jwt_token is not importable. Please install/configure 'mcpgateway'."
            ) from e

        env = os.environ.copy()
        if jwt_secret_key is not None:
            env["MCPGATEWAY_JWT_SECRET"] = jwt_secret_key
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
        """Namespaced admin API access: ``mcp_registry``, ``gateway``, ``tools``.

        Returns:
            ``_AdminNamespace``: Holder exposing the Gateway admin endpoints.
        """
        return self._admin

    # Backward-compatible admin_* wrappers have been removed in favor of the
    # descriptive namespaced interface available via `client.admin`.

    def health(self) -> Dict[str, Any]:
        """Check Gateway service health.

        API
        ---
        - Method/Path: ``GET /health``
        - Auth: Optional (depends on deployment)

        Returns:
            Parsed JSON when available, otherwise ``{"status": <raw text>}``.

        Raises:
            GatewayApiError: When the response status is non-2xx.
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
    ) -> CatalogListResponseDTO:
        """List catalog/registry servers.

        API
        ---
        - Method/Path: ``GET /admin/mcp-registry/servers``
        - Query:
          - ``include_inactive`` (bool): include inactive entries.
          - ``tags`` (str): CSV of tags to filter by.
          - ``team_id`` (str): scope to a team.
          - ``visibility`` (str): e.g., ``public``, ``team``, ``private``.

        Returns:
            ``CatalogListResponseDTO``: Typed listing payload (servers, totals, facets).
        """
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
        return CatalogListResponseDTO.model_validate(r.json())

    def register(self, server_id: str) -> CatalogServerRegisterResponseDTO:
        """Register a catalog server by identifier.

        API
        ---
        - Method/Path: ``POST /admin/mcp-registry/{server_id}/register``
        - Path params:
          - ``server_id``: Identifier from the catalog list.

        Side-effects
        ------------
        Triggers discovery against the referenced MCP server and records it as
        registered in the Gateway. The response indicates success and summary.

        Returns:
            ``CatalogServerRegisterResponseDTO`` with success flag, server_id, and message.
        """
        self._client._ensure_token()
        r = self._client._client.post(
            f"{self._client.base_url}/admin/mcp-registry/{server_id}/register",
            headers=self._client._headers(),
        )
        r.raise_for_status()
        return CatalogServerRegisterResponseDTO.model_validate(r.json())


class _GatewayMgmtNamespace:
    def __init__(self, client: GatewayApiClient) -> None:
        self._client = client

    def list(self, include_inactive: Optional[bool] = None) -> List[GatewayReadDTO]:
        """List Gateway instances.

        API
        ---
        - Method/Path: ``GET /admin/gateways``
        - Query:
          - ``include_inactive`` (bool): include inactive gateways in results.

        Notes
        -----
        Some deployments return a list directly, others an object with ``items``;
        both shapes are normalized here and returned as a flat list of DTOs.

        Returns:
            ``List[GatewayReadDTO]``
        """
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
        data = r.json()
        if isinstance(data, list):
            return [GatewayReadDTO.model_validate(x) for x in data]
        items = data.get("items", []) if isinstance(data, dict) else []
        return [GatewayReadDTO.model_validate(x) for x in items]

    def get(self, gateway_id: str) -> GatewayReadDTO:
        """Get a single Gateway instance by id.

        API
        ---
        - Method/Path: ``GET /admin/gateways/{gateway_id}``
        - Path params:
          - ``gateway_id``: Unique identifier of the Gateway instance.

        Returns:
            ``GatewayReadDTO``
        """
        self._client._ensure_token()
        r = self._client._client.get(
            f"{self._client.base_url}/admin/gateways/{gateway_id}", headers=self._client._headers()
        )
        r.raise_for_status()
        return GatewayReadDTO.model_validate(r.json())


class _ToolsNamespace:
    def __init__(self, client: GatewayApiClient) -> None:
        self._client = client

    def list(self, offset: int = 0, limit: int = 50, include_inactive: Optional[bool] = None) -> AdminToolsListResponseDTO:
        """List federated tools available via the Gateway.

        API
        ---
        - Method/Path: ``GET /admin/tools``
        - Query:
          - ``offset`` (int): pagination offset.
          - ``limit`` (int): page size.
          - ``include_inactive`` (bool): include inactive tools.

        Returns:
            ``AdminToolsListResponseDTO`` containing ``data`` (list of tools) and
            optional paging/links metadata depending on deployment.
        """
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
        return AdminToolsListResponseDTO.model_validate(r.json())

    def get(self, tool_id: str) -> ToolReadDTO:
        """Get tool details by identifier.

        API
        ---
        - Method/Path: ``GET /admin/tools/{tool_id}``
        - Path params:
          - ``tool_id``: Unique identifier of the tool (not the MCP method name).

        Returns:
            ``ToolReadDTO`` with definition, schemas, metrics, and metadata.
        """
        self._client._ensure_token()
        r = self._client._client.get(
            f"{self._client.base_url}/admin/tools/{tool_id}", headers=self._client._headers()
        )
        r.raise_for_status()
        return ToolReadDTO.model_validate(r.json())


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
