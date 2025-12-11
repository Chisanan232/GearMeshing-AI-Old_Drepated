from __future__ import annotations

from typing import Any, Dict, List

import httpx

from gearmeshing_ai.info_provider.schemas.config import ServerConfig
from gearmeshing_ai.info_provider.strategy.direct import DirectMcpStrategy


def _mock_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        # Expecting base_url http://mock and endpoint_url http://mock/mcp
        # So tools path -> /mcp/tools
        if request.method == "GET" and request.url.path == "/mcp/tools":
            data: List[Dict[str, Any]] = [
                {
                    "name": "echo",
                    "description": "Echo tool",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"text": {"type": "string", "description": "Text to echo"}},
                        "required": ["text"],
                    },
                }
            ]
            return httpx.Response(200, json=data)
        if request.method == "POST" and request.url.path == "/mcp/a2a/echo/invoke":
            body: Dict[str, Any] = {}
            if request.content:
                try:
                    body = httpx.RequestContent.decode(request).json()
                except Exception:
                    try:
                        body = httpx._models.jsonlib.loads(request.content.decode("utf-8"))
                    except Exception:
                        body = {}
            params = (body or {}).get("parameters") or {}
            return httpx.Response(200, json={"ok": True, "echo": params.get("text")})
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


def test_direct_strategy_list_and_call() -> None:
    transport = _mock_transport()
    client = httpx.Client(transport=transport, base_url="http://mock")
    servers = [ServerConfig(name="direct1", endpoint_url="http://mock/mcp")]

    strategy = DirectMcpStrategy(servers, client=client)

    servers_list = list(strategy.list_servers())
    assert len(servers_list) == 1
    assert servers_list[0].id == "direct1"

    tools = list(strategy.list_tools("direct1"))
    assert len(tools) == 1
    assert tools[0].name == "echo"
    assert tools[0].raw_parameters_schema["properties"]["text"]["type"] == "string"

    res = strategy.call_tool("direct1", "echo", {"text": "hi"})
    assert res.ok is True
    assert res.data.get("echo") == "hi"
