from __future__ import annotations

import asyncio

import httpx
import pytest

from gearmeshing_ai.info_provider.transport.sse import BasicSseTransport


class SlowAsyncStream(httpx.AsyncByteStream):
    def __init__(self, parts: list[tuple[bytes, float]]) -> None:
        self._parts = parts

    async def aiter_bytes(self):
        for chunk, delay in self._parts:
            if delay:
                await asyncio.sleep(delay)
            yield chunk

    async def aclose(self) -> None:
        return None

    def __aiter__(self):
        return self.aiter_bytes()


@pytest.mark.asyncio
async def test_basic_sse_transport_idle_timeout_triggers_stop_without_reconnect() -> None:
    # Stream yields one line, then delays longer than idle_timeout
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/sse":
            stream = SlowAsyncStream(
                [
                    (b"data: first\n", 0.0),
                    (b"data: second\n", 0.2),  # long delay to exceed idle_timeout
                ]
            )
            headers = {"content-type": "text/event-stream"}
            return httpx.Response(200, headers=headers, stream=stream)
        return httpx.Response(404, json={"error": "not found"})

    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport, base_url="http://mock")

    sse = BasicSseTransport(
        "http://mock",
        client=async_client,
        include_blank_lines=False,
        reconnect=False,  # no reconnect, should stop on idle timeout
        idle_timeout=0.05,
    )
    await sse.connect("/sse")

    results: list[str] = []
    async for line in sse.aiter():
        results.append(line)
        # We should only receive the first line, then timeout stops the stream
        if len(results) > 1:
            break

    assert results == ["data: first"]

    await sse.close()
    await async_client.aclose()


@pytest.mark.asyncio
async def test_basic_sse_transport_max_total_seconds_stops_immediately() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/sse":
            # Long stream but we'll cut it off via max_total_seconds
            stream = SlowAsyncStream(
                [
                    (b"data: a\n", 0.0),
                    (b"data: b\n", 0.0),
                    (b"data: c\n", 0.0),
                ]
            )
            headers = {"content-type": "text/event-stream"}
            return httpx.Response(200, headers=headers, stream=stream)
        return httpx.Response(404, json={"error": "not found"})

    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport, base_url="http://mock")

    sse = BasicSseTransport(
        "http://mock",
        client=async_client,
        include_blank_lines=False,
        reconnect=False,
        max_total_seconds=0.0,  # should stop immediately
    )
    await sse.connect("/sse")

    results: list[str] = []
    async for line in sse.aiter():
        results.append(line)

    assert results == []

    await sse.close()
    await async_client.aclose()
