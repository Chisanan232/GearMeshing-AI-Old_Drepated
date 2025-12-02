"""Transport interfaces for MCP client communication.

  Defines lightweight Protocols used by strategies and clients to abstract
  transport implementations (HTTP streaming, Server-Sent Events, stdio).
  Concrete transports live alongside these interfaces (e.g., `sse.py`).
  """

from __future__ import annotations

from typing import Any, Protocol


class HttpStreamTransport(Protocol):
    """Protocol for synchronous HTTP-like request/response transports.

    Purpose:
    - Abstract single-request/single-response interactions used by direct
      strategies (e.g., JSON-RPC over HTTP).

    Examples:
        >>> resp = transport.send_request({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        >>> assert "result" in resp
    """

    def send_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a request payload and return the parsed response.

        Args:
            payload: JSON-serializable request body, already including any
                headers or RPC metadata required by the strategy.

        Returns:
            Parsed response body as a JSON-compatible dictionary.

        Raises:
            Exception: Implementations may raise transport-specific errors
                (e.g., network failures, timeouts).
        
        Examples:
            >>> resp = transport.send_request({"jsonrpc": "2.0", "id": 7, "method": "status"})
            >>> print(resp.get("result"))
        """
        ...


class SseTransport(Protocol):
    """Protocol for async Server-Sent Events (SSE) transports.

    Implementations manage the lifecycle of a persistent HTTP stream
    delivering text/event-stream data to strategy consumers.

    Examples:
        >>> await transport.connect("/sse")
        >>> # consume via a strategy helper yielding events
        >>> await transport.close()
    """

    async def connect(self, path: str) -> None:
        """Open a streaming connection.

        Args:
            path: HTTP path to connect to (e.g., "/sse"), relative to the
                configured base URL.

        Raises:
            Exception: If the connection cannot be established.
        
        Examples:
            >>> await transport.connect("/events")
        """
        ...

    async def close(self) -> None:
        """Close the current streaming connection and release resources.

        Raises:
            Exception: If an error occurs during teardown.
        
        Examples:
            >>> await transport.close()
        """
        ...


class StdioTransport(Protocol):
    """Protocol for stdio-based process transports.

    Examples:
        >>> transport.start()
        >>> # interact with the process via stdio
        >>> transport.stop()
    """

    def start(self) -> None:
        """Start the underlying process and connect IO streams.

        Raises:
            Exception: If the process cannot be started.
        
        Examples:
            >>> transport.start()
        """
        ...

    def stop(self) -> None:
        """Stop the process and clean up resources.

        Raises:
            Exception: If an error occurs while stopping.
        
        Examples:
            >>> transport.stop()
        """
        ...
