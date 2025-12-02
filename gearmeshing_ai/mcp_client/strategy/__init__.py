"""MCP strategies for interacting with servers.

Provides concrete strategies for:
- Direct server access (`DirectMcpStrategy`)
- Gateway-backed access (`GatewayMcpStrategy`, `AsyncGatewayMcpStrategy`)

These classes encapsulate transport details and expose a uniform
interface consumed by the high-level client facades.
"""

from .direct import DirectMcpStrategy
from .gateway import GatewayMcpStrategy
from .gateway_async import AsyncGatewayMcpStrategy

__all__ = [
    "DirectMcpStrategy",
    "GatewayMcpStrategy",
    "AsyncGatewayMcpStrategy",
]
