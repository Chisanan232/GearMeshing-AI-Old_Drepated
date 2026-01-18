from __future__ import annotations

from typing import Iterable

import httpx
import pytest


@pytest.fixture(scope="session")
def test_config():
    """Fixture providing test configuration from Pydantic settings model.

    Returns:
        TestSettings: Test configuration with all environment variables loaded
    """
    from test.settings import test_settings

    return test_settings


@pytest.fixture(autouse=True)
def _global_offline_http_guard(monkeypatch: pytest.MonkeyPatch):
    allowed_prefixes: Iterable[str] = (
        "http://mock",
        "https://mock",
        "http://localhost",
        "http://127.0.0.1",
        "http://0.0.0.0",
        "/",  # Allow relative paths (used by ASGI transport)
    )
    litellm_prices_prefix = "https://raw.githubusercontent.com/BerriAI/litellm/"

    orig_sync = httpx._client.Client.request
    orig_async = httpx._client.AsyncClient.request

    def _is_allowed(url_str: str) -> bool:
        return any(url_str.startswith(p) for p in allowed_prefixes)

    def _should_stub_empty(url_str: str) -> bool:
        return url_str.startswith(litellm_prices_prefix)

    def offline_sync(self, method, url, *args, **kwargs):
        url_str = str(url)
        if _is_allowed(url_str):
            return orig_sync(self, method, url, *args, **kwargs)
        if _should_stub_empty(url_str):
            return httpx.Response(200, json={})
        raise RuntimeError(f"External HTTP blocked by global offline guard: {url_str}")

    async def offline_async(self, method, url, *args, **kwargs):
        url_str = str(url)
        if _is_allowed(url_str):
            return await orig_async(self, method, url, *args, **kwargs)
        if _should_stub_empty(url_str):
            return httpx.Response(200, json={})
        raise RuntimeError(f"External HTTP blocked by global offline guard (async): {url_str}")

    monkeypatch.setattr(httpx._client.Client, "request", offline_sync, raising=True)
    monkeypatch.setattr(httpx._client.AsyncClient, "request", offline_async, raising=True)
