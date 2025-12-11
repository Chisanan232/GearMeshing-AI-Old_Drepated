from __future__ import annotations

import httpx
import pytest

from gearmeshing_ai.info_provider.mcp.transport.sse import BasicSseTransport


def _mock_transport_reconnect() -> httpx.MockTransport:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/sse":
            calls["count"] += 1
            if calls["count"] == 1:
                # First segment
                body = b"data: first\n\n"
            elif calls["count"] == 2:
                # Second segment after reconnect
                body = b"data: second\n\n"
            else:
                # Any further reconnects receive empty stream
                body = b""
            headers = {"content-type": "text/event-stream"}
            return httpx.Response(200, headers=headers, content=body)
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_basic_sse_transport_reconnect_two_segments() -> None:
    transport = _mock_transport_reconnect()
    async_client = httpx.AsyncClient(transport=transport, base_url="http://mock")

    sse = BasicSseTransport(
        "http://mock",
        client=async_client,
        reconnect=True,
        max_retries=1,
        backoff_initial=0.0,
        include_blank_lines=False,
    )
    await sse.connect("/sse")

    results: list[str] = []
    async for line in sse.aiter():
        results.append(line)
        if len(results) >= 2:
            break

    # We should see lines from both segments across reconnect
    assert results[0].startswith("data: first")
    assert results[1].startswith("data: second")

    await sse.close()
    await async_client.aclose()


@pytest.mark.asyncio
async def test_basic_sse_transport_skip_blank_lines_by_default() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/sse":
            body = b"data: a\n\n:\n\ndata: b\n\n"
            headers = {"content-type": "text/event-stream"}
            return httpx.Response(200, headers=headers, content=body)
        return httpx.Response(404, json={"error": "not found"})

    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport, base_url="http://mock")

    sse = BasicSseTransport("http://mock", client=async_client)  # include_blank_lines defaults to False
    await sse.connect("/sse")

    results: list[str] = []
    async for line in sse.aiter():
        results.append(line)
        if len(results) >= 3:
            break

    # Blank lines should be skipped by default, but comments (":" lines) are yielded
    assert results[0].startswith("data: a")
    assert results[1].startswith(":")
    assert results[2].startswith("data: b")

    await sse.close()
    await async_client.aclose()
