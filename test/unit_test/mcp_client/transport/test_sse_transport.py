from __future__ import annotations

import asyncio

import httpx
import pytest

from gearmeshing_ai.mcp_client.transport.sse import BasicSseTransport


def _mock_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/sse":
            body = (
                b"data: one\n\n"
                b": comment line\n\n"
                b"data: two\n\n"
            )
            headers = {"content-type": "text/event-stream"}
            return httpx.Response(200, headers=headers, content=body)
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_basic_sse_transport_connect_iter_close() -> None:
    transport = _mock_transport()
    async_client = httpx.AsyncClient(transport=transport, base_url="http://mock")

    sse = BasicSseTransport("http://mock", client=async_client, auth_token="Bearer t")
    await sse.connect("/sse")

    results: list[str] = []
    async for line in sse.aiter():
        results.append(line)
        if len(results) >= 3:
            break

    # We expect to see raw lines including the comment line
    assert results[0].startswith("data: one")
    assert results[1].startswith(": comment")
    assert results[2].startswith("data: two")

    await sse.close()
    await async_client.aclose()
