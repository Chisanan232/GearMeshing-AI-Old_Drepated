from __future__ import annotations

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
        self, base_url: str, *, client: Optional[httpx.AsyncClient] = None, auth_token: Optional[str] = None
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=10.0, follow_redirects=True)
        self._auth_token = auth_token
        self._response: Optional[httpx.Response] = None

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "text/event-stream"}
        if self._auth_token:
            h["Authorization"] = self._auth_token
        return h

    async def connect(self, path: str) -> None:
        url = f"{self._base_url}/{path.lstrip('/')}"
        self._response = await self._client.get(url, headers=self._headers(), timeout=None)
        self._response.raise_for_status()

    async def aiter(self) -> AsyncIterator[str]:
        if self._response is None:
            raise RuntimeError("SSE not connected. Call connect() first.")
        async for line in self._response.aiter_lines():
            if not line:
                continue
            yield line

    async def close(self) -> None:
        if self._response is not None:
            await self._response.aclose()
            self._response = None
        await self._client.aclose()
