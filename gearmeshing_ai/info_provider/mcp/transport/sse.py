"""Server-Sent Events (SSE) transport utilities.

Provides a minimal SSE transport built on `httpx.AsyncClient` with optional
reconnect, backoff, idle timeout, and maximum total time controls. Used by
async strategies to implement streaming APIs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator, Optional

import httpx


class BasicSseTransport:
    """Minimal SSE transport using httpx.AsyncClient.

    - connect(path): start a streamed GET request
    - aiter(): yield raw SSE lines (already decoded)
    - close(): close the underlying response and client

    Usage guidelines:
    - Set `include_blank_lines=True` when you need explicit event boundaries.
    - Use reconnect/backoff parameters to handle flaky networks.
    - Prefer a caller-provided AsyncClient when you need custom timeouts/proxies.
    """

    def __init__(
        self,
        base_url: str,
        *,
        client: Optional[httpx.AsyncClient] = None,
        auth_token: Optional[str] = None,
        include_blank_lines: bool = False,
        reconnect: bool = False,
        max_retries: int = 3,
        backoff_initial: float = 0.5,
        backoff_factor: float = 2.0,
        backoff_max: float = 8.0,
        idle_timeout: Optional[float] = None,
        max_total_seconds: Optional[float] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=10.0, follow_redirects=True)
        self._auth_token = auth_token
        self._response: Optional[httpx.Response] = None
        self._include_blank = include_blank_lines
        self._reconnect = reconnect
        self._max_retries = max_retries
        self._backoff_initial = backoff_initial
        self._backoff_factor = backoff_factor
        self._backoff_max = backoff_max
        self._path: Optional[str] = None
        self._logger = logging.getLogger(__name__)
        self._idle_timeout = idle_timeout
        self._max_total = max_total_seconds

    def _headers(self) -> dict[str, str]:
        """Build headers for SSE requests.

        Returns:
            A dictionary with `Accept: text/event-stream` and optional `Authorization`.
        """
        h = {"Accept": "text/event-stream"}
        if self._auth_token:
            h["Authorization"] = self._auth_token
        return h

    async def connect(self, path: str) -> None:
        """Establish the streaming connection to the given path (relative).

        Args:
            path: Relative path under the base URL (e.g., "/sse").

        Returns:
            None.

        Raises:
            httpx.HTTPStatusError: If the initial request returns a non-2xx response.
            httpx.TransportError: For transport-level HTTP issues.
        """
        self._path = path
        await self._do_connect()

    async def _do_connect(self) -> None:
        """Internal helper to (re)connect the streaming HTTP request.

        Returns:
            None.

        Raises:
            httpx.HTTPStatusError: If the request returns a non-2xx response.
            httpx.TransportError: For transport-level HTTP issues.
        """
        assert self._path is not None
        url = f"{self._base_url}/{self._path.lstrip('/')}"
        self._logger.debug("SSE connect: GET %s", url)
        req = self._client.build_request("GET", url, headers=self._headers())
        self._response = await self._client.send(req, stream=True)
        self._response.raise_for_status()

    async def aiter(self) -> AsyncIterator[str]:
        """Iterate over raw SSE lines with optional reconnect/backoff behavior.

        Returns:
            An async iterator of decoded SSE lines. When `include_blank_lines` is False,
            blank lines (event boundaries) are skipped.

        Raises:
            RuntimeError: If `connect()` was not called before iteration.
            httpx.HTTPStatusError: If a reconnect attempt fails with non-2xx.
            httpx.TransportError: For transport-level HTTP issues during reads/reconnects.
        """
        if self._response is None:
            raise RuntimeError("SSE not connected. Call connect() first.")
        retries = 0
        start = time.monotonic()
        while True:
            try:
                line_iter = self._response.aiter_lines()
                while True:
                    if self._max_total is not None and (time.monotonic() - start) >= self._max_total:
                        return
                    try:
                        nxt = line_iter.__anext__()
                        if self._idle_timeout is not None:
                            line = await asyncio.wait_for(nxt, timeout=self._idle_timeout)
                        else:
                            line = await nxt
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        if not self._reconnect or retries >= self._max_retries:
                            return
                        sleep_s = min(self._backoff_initial * (self._backoff_factor**retries), self._backoff_max)
                        self._logger.warning(
                            "SSE idle timeout; reconnecting in %ss (attempt %s/%s)",
                            sleep_s,
                            retries + 1,
                            self._max_retries,
                        )
                        retries += 1
                        await asyncio.sleep(sleep_s)
                        await self._do_connect()
                        break
                    if (not line) and not self._include_blank:
                        continue
                    yield line
                if not self._reconnect or retries >= self._max_retries:
                    break
                sleep_s = min(self._backoff_initial * (self._backoff_factor**retries), self._backoff_max)
                self._logger.warning(
                    "SSE stream ended; reconnecting in %ss (attempt %s/%s)", sleep_s, retries + 1, self._max_retries
                )
                retries += 1
                await asyncio.sleep(sleep_s)
                await self._do_connect()
                continue
            except Exception:
                if not self._reconnect or retries >= self._max_retries:
                    self._logger.error("SSE stream error; giving up", exc_info=True)
                    raise
                sleep_s = min(self._backoff_initial * (self._backoff_factor**retries), self._backoff_max)
                self._logger.warning(
                    "SSE error; reconnecting in %ss (attempt %s/%s)", sleep_s, retries + 1, self._max_retries
                )
                retries += 1
                await asyncio.sleep(sleep_s)
                await self._do_connect()
                continue

    async def close(self) -> None:
        """Close the current SSE response (if any) and underlying client.

        Returns:
            None.
        """
        if self._response is not None:
            await self._response.aclose()
            self._response = None
        await self._client.aclose()
