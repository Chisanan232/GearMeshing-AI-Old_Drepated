from __future__ import annotations

import asyncio
from typing import List, Dict, Any

import httpx
import pytest

from gearmeshing_ai.mcp_client.gateway_api.client import GatewayApiClient
from gearmeshing_ai.mcp_client.strategy.gateway_async import AsyncGatewayMcpStrategy


def _mock_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        # SSE will be requested at /servers/s1/mcp/sse
        if request.method == "GET" and request.url.path == "/servers/s1/mcp/sse":
            body = (
                b": comment should be ignored\n\n"
                b"id: 1\n"
                b"event: token\n"
                b"data: part1\n"
                b"data: part2\n\n"
                b"id: 2\n"
                b"event: done\n"
                b"data: finished\n\n"
            )
            headers = {"content-type": "text/event-stream"}
            return httpx.Response(200, headers=headers, content=body)
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_gateway_async_stream_events_parsed() -> None:
    # We only need the base_url and token for headers
    gw = GatewayApiClient("http://mock", auth_token="Bearer sse")

    sse_client = httpx.AsyncClient(transport=_mock_transport(), base_url="http://mock")

    strat = AsyncGatewayMcpStrategy(gw, sse_client=sse_client)

    results: List[Dict[str, Any]] = []
    async for evt in strat.stream_events_parsed("s1", path="/sse"):
        results.append(evt)
        if len(results) >= 2:
            break

    assert results[0] == {"id": "1", "event": "token", "data": "part1\npart2"}
    assert results[1] == {"id": "2", "event": "done", "data": "finished"}

    await sse_client.aclose()
