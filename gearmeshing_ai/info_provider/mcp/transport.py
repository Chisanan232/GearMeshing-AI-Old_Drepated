from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncContextManager, AsyncIterator, Protocol

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client


class AsyncMCPTransport(Protocol):
    """Protocol for creating MCP ClientSession connections asynchronously.

    Implementations return an async context manager via ``session(endpoint_url)``
    that yields an initialized ``ClientSession``.
    """

    def session(self, endpoint_url: str) -> AsyncContextManager[ClientSession]:
        """
        Establish a session with the given endpoint.

        Args:
            endpoint_url: The URL of the MCP server (HTTP or SSE).

        Returns:
            AsyncContextManager[ClientSession]: An async context manager yielding a connected session.
        """
        ...


class StreamableHttpMCPTransport(AsyncMCPTransport):
    """MCP transport using the streamable HTTP client."""

    def session(self, endpoint_url: str) -> AsyncContextManager[ClientSession]:
        """
        Create a session over Streamable HTTP.

        Args:
            endpoint_url: The HTTP URL of the MCP endpoint.

        Returns:
            AsyncContextManager[ClientSession]: An initialized session over HTTP.
        """
        @asynccontextmanager
        async def _cm() -> AsyncIterator[ClientSession]:
            async with streamablehttp_client(endpoint_url) as (read_stream, write_stream, _close_fn):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session

        return _cm()


class SseMCPTransport(AsyncMCPTransport):
    """MCP transport using the SSE client (MCP over SSE)."""

    def session(self, endpoint_url: str) -> AsyncContextManager[ClientSession]:
        """
        Create a session over Server-Sent Events (SSE).

        Args:
            endpoint_url: The SSE URL of the MCP endpoint.

        Returns:
            AsyncContextManager[ClientSession]: An initialized session over SSE.
        """
        @asynccontextmanager
        async def _cm() -> AsyncIterator[ClientSession]:
            async with sse_client(endpoint_url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session

        return _cm()
