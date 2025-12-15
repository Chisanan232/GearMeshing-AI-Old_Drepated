from __future__ import annotations

import httpx


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
