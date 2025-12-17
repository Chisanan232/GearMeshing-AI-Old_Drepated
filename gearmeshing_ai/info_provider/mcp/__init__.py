"""MCP info provider facade and supporting components.

This subpackage implements a policy-aware *metadata discovery* layer for MCP
(Model Context Protocol) servers.

Scope
-----

This layer is intentionally focused on *read-only* metadata operations:

- discovering configured servers,
- listing tools available on a server,
- optionally listing tool metadata with pagination.

It is designed to support AI agent frameworks by supplying tool schemas and
names, while keeping tool execution and streaming responsibilities in lower
layers (transport/strategies) and higher layers (agent runtime/capabilities).

Key modules
-----------

- ``base``: Protocols and shared helpers for the info-provider facade.
- ``provider``: ``MCPInfoProvider`` and ``AsyncMCPInfoProvider`` implementations.
- ``strategy``: direct and gateway-backed strategies (sync/async).
- ``schemas``: Pydantic schemas for configuration and MCP DTOs.
- ``gateway_api``: HTTP client and DTOs for the MCP gateway management service.

Policy
------

Providers can apply per-agent policies:

- server allow-lists,
- tool allow-lists,
- read-only mode (blocks mutating tools).

These policies are enforced in the info-provider facade when an ``agent_id`` is
supplied.
"""

from .provider import AsyncMCPInfoProvider, MCPInfoProvider

__all__ = [
    "MCPInfoProvider",
    "AsyncMCPInfoProvider",
]
