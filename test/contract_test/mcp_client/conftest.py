from __future__ import annotations

from typing import Iterable

import httpx
import pytest


@pytest.fixture
def offline_http_guard(monkeypatch: pytest.MonkeyPatch):
    """Prevent external network during tests in this package.

    - Allows requests to mock base URLs (http://mock, https://mock).
    - Short-circuits known LiteLLM remote price fetch path with empty JSON.
    - Raises on any other external HTTP request.
    """

    allowed_prefixes: Iterable[str] = ("http://mock", "https://mock")
    litellm_prices_prefix = "https://raw.githubusercontent.com/BerriAI/litellm/"

    orig_sync = httpx._client.Client.request
    orig_async = httpx._client.AsyncClient.request

    def _is_allowed(url_str: str) -> bool:
        return any(url_str.startswith(p) for p in allowed_prefixes)

    def _should_stub_empty(url_str: str) -> bool:
        return url_str.startswith(litellm_prices_prefix)

    def offline_sync(self, method, url, *args, **kwargs):  # type: ignore[no-redef]
        url_str = str(url)
        if _is_allowed(url_str):
            return orig_sync(self, method, url, *args, **kwargs)
        if _should_stub_empty(url_str):
            return httpx.Response(200, json={})
        raise RuntimeError(f"External HTTP blocked by offline_http_guard: {url_str}")

    async def offline_async(self, method, url, *args, **kwargs):  # type: ignore[no-redef]
        url_str = str(url)
        if _is_allowed(url_str):
            return await orig_async(self, method, url, *args, **kwargs)
        if _should_stub_empty(url_str):
            return httpx.Response(200, json={})
        raise RuntimeError(f"External HTTP blocked by offline_http_guard (async): {url_str}")

    monkeypatch.setattr(httpx._client.Client, "request", offline_sync, raising=True)
    monkeypatch.setattr(httpx._client.AsyncClient, "request", offline_async, raising=True)

    # Provide a small context object if a test wants to update allowed prefixes later
    class Guard:
        def allow_prefix(self, prefix: str) -> None:
            nonlocal allowed_prefixes
            allowed_prefixes = tuple(list(allowed_prefixes) + [prefix])

    yield Guard()
