from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

import httpx


class BasicSseTransport:
    """
    Minimal SSE transport using httpx.AsyncClient.

    - connect(path): starts a streamed GET request
    - aiter(): yields raw SSE lines (already decoded)
    - close(): closes the underlying response
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

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "text/event-stream"}
        if self._auth_token:
            h["Authorization"] = self._auth_token
        return h

    async def connect(self, path: str) -> None:
        self._path = path
        await self._do_connect()

    async def _do_connect(self) -> None:
        assert self._path is not None
        url = f"{self._base_url}/{self._path.lstrip('/')}"
        self._logger.debug("SSE connect: GET %s", url)
        self._response = await self._client.get(url, headers=self._headers(), timeout=None)
        self._response.raise_for_status()

    async def aiter(self) -> AsyncIterator[str]:
        if self._response is None:
            raise RuntimeError("SSE not connected. Call connect() first.")
        retries = 0
        while True:
            try:
                async for line in self._response.aiter_lines():
                    if (not line) and not self._include_blank:
                        continue
                    yield line
                # Stream ended gracefully
                if not self._reconnect or retries >= self._max_retries:
                    break
                sleep_s = min(self._backoff_initial * (self._backoff_factor ** retries), self._backoff_max)
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
                sleep_s = min(self._backoff_initial * (self._backoff_factor ** retries), self._backoff_max)
                self._logger.warning(
                    "SSE error; reconnecting in %ss (attempt %s/%s)", sleep_s, retries + 1, self._max_retries
                )
                retries += 1
                await asyncio.sleep(sleep_s)
                await self._do_connect()
                continue

    async def close(self) -> None:
        if self._response is not None:
            await self._response.aclose()
            self._response = None
        await self._client.aclose()
