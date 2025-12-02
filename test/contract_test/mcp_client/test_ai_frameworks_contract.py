from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Sequence

import httpx
import pytest

from gearmeshing_ai.mcp_client.client_async import AsyncMcpClient
from gearmeshing_ai.mcp_client.client_sync import McpClient
from gearmeshing_ai.mcp_client.gateway_api.client import GatewayApiClient
from gearmeshing_ai.mcp_client.schemas.config import GatewayConfig, McpClientConfig, ServerConfig
from gearmeshing_ai.mcp_client.schemas.core import McpTool


# ------------------------------
# Mock transports for Direct and Gateway
# ------------------------------

def _mock_transport_direct() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/mcp/tools":
            data: List[Dict[str, Any]] = [
                {
                    "name": "echo",
                    "description": "Echo tool",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                }
            ]
            return httpx.Response(200, json=data)
        if request.method == "POST" and request.url.path == "/mcp/a2a/echo/invoke":
            return httpx.Response(200, json={"ok": True, "result": "ok"})
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


def _mock_transport_gateway() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/servers":
            data = [
                {
                    "id": "s1",
                    "name": "gateway-s1",
                    "url": "http://underlying/mcp/",
                    "transport": "STREAMABLEHTTP",
                }
            ]
            return httpx.Response(200, json=data)
        if request.method == "GET" and request.url.path == "/servers/s1/mcp/tools":
            tools: List[Dict[str, Any]] = [
                {
                    "name": "echo",
                    "description": "Echo tool",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                }
            ]
            return httpx.Response(200, json=tools)
        if request.method == "POST" and request.url.path == "/servers/s1/mcp/a2a/echo/invoke":
            return httpx.Response(200, json={"ok": True, "result": "ok"})
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


# ------------------------------
# Adapters to framework-friendly tool descriptors
# ------------------------------

def to_openai_function_tools(tools: Sequence[McpTool]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in tools:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.raw_parameters_schema or {"type": "object"},
                },
            }
        )
    return out


def to_langchain_tools(tools: Sequence[McpTool]) -> List[Any]:
    lc = pytest.importorskip("langchain_core.tools", reason="langchain not installed")
    Tool = getattr(lc, "Tool")

    out: List[Any] = []
    for t in tools:
        def _fn(**kwargs: Any) -> Dict[str, Any]:
            return {"called": t.name, "args": kwargs}

        out.append(Tool(name=t.name, description=t.description or "", func=_fn))
    return out


# ------------------------------
# Sync: client + direct
# ------------------------------

def test_framework_adapters_sync_direct_servers_and_tools() -> None:
    transport = _mock_transport_direct()
    http_client = httpx.Client(transport=transport, base_url="http://mock")

    cfg = McpClientConfig(servers=[ServerConfig(name="s1", endpoint_url="http://mock/mcp")])

    client = McpClient.from_config(cfg, direct_http_client=http_client)

    servers = client.list_servers()
    assert [s.id for s in servers] == ["s1"]

    tools = client.list_tools("s1")
    assert tools and tools[0].name == "echo"

    # OpenAI function tools
    oa_tools = to_openai_function_tools(tools)
    assert oa_tools and oa_tools[0]["function"]["name"] == "echo"
    assert oa_tools[0]["function"]["parameters"]["type"] == "object"

    # LangChain tools
    lc_tools = to_langchain_tools(tools)
    assert lc_tools and getattr(lc_tools[0], "name", None) == "echo"


# ------------------------------
# Sync: client + gateway
# ------------------------------

def test_framework_adapters_sync_gateway_servers_and_tools() -> None:
    transport = _mock_transport_gateway()
    mgmt_client = httpx.Client(transport=transport, base_url="http://mock")
    http_client = httpx.Client(transport=transport, base_url="http://mock")

    cfg = McpClientConfig(gateway=GatewayConfig(base_url="http://mock"))

    # Build a gateway-only client via from_config
    client = McpClient.from_config(
        cfg,
        gateway_mgmt_client=mgmt_client,
        gateway_http_client=http_client,
    )

    servers = client.list_servers()
    assert [s.id for s in servers] == ["s1"]

    tools = client.list_tools("s1")
    assert tools and tools[0].name == "echo"

    oa_tools = to_openai_function_tools(tools)
    assert oa_tools and oa_tools[0]["function"]["name"] == "echo"

    lc_tools = to_langchain_tools(tools)
    assert lc_tools and getattr(lc_tools[0], "name", None) == "echo"


# ------------------------------
# Async: async client + async direct
# ------------------------------
@pytest.mark.asyncio
async def test_framework_adapters_async_direct_tools() -> None:
    # Build AsyncMcpClient with AsyncDirectMcpStrategy using Mock AsyncClient
    from gearmeshing_ai.mcp_client.strategy.direct_async import AsyncDirectMcpStrategy

    atransport = _mock_transport_direct()
    async_client = httpx.AsyncClient(transport=atransport, base_url="http://mock")

    strat = AsyncDirectMcpStrategy([ServerConfig(name="s1", endpoint_url="http://mock/mcp")], client=async_client)
    client = AsyncMcpClient(strategies=[strat])

    tools = await client.list_tools("s1")
    assert tools and tools[0].name == "echo"

    oa_tools = to_openai_function_tools(tools)
    assert oa_tools and oa_tools[0]["function"]["name"] == "echo"

    lc_tools = to_langchain_tools(tools)
    assert lc_tools and getattr(lc_tools[0], "name", None) == "echo"

    await async_client.aclose()


# ------------------------------
# Async: async client + async gateway
# ------------------------------
@pytest.mark.asyncio
async def test_framework_adapters_async_gateway_tools() -> None:
    atransport = _mock_transport_gateway()
    mgmt_client = httpx.Client(transport=atransport, base_url="http://mock")
    http_client = httpx.AsyncClient(transport=atransport, base_url="http://mock")
    sse_client = httpx.AsyncClient(transport=atransport, base_url="http://mock")

    cfg = McpClientConfig(gateway=GatewayConfig(base_url="http://mock"))

    client = await AsyncMcpClient.from_config(
        cfg,
        gateway_mgmt_client=mgmt_client,
        gateway_http_client=http_client,
        gateway_sse_client=sse_client,
    )

    tools = await client.list_tools("s1")
    assert tools and tools[0].name == "echo"

    oa_tools = to_openai_function_tools(tools)
    assert oa_tools and oa_tools[0]["function"]["name"] == "echo"

    lc_tools = to_langchain_tools(tools)
    assert lc_tools and getattr(lc_tools[0], "name", None) == "echo"

    await http_client.aclose()
    await sse_client.aclose()


# ------------------------------
# Framework-specific adapters (best-effort, import-or-skip)
# ------------------------------

def to_crewai_tools(tools: Sequence[McpTool]) -> List[Any]:
    pytest.importorskip("crewai", reason="crewai not installed")
    # CrewAI commonly interoperates with LangChain tools; reuse LC mapping.
    return to_langchain_tools(tools)


def to_llamaindex_tools(tools: Sequence[McpTool]) -> List[Any]:
    li_tools = pytest.importorskip("llama_index.core.tools", reason="llama-index not installed")
    FunctionTool = getattr(li_tools, "FunctionTool")

    out: List[Any] = []
    for t in tools:
        def _fn(**kwargs: Any) -> Dict[str, Any]:
            return {"called": t.name, "args": kwargs}

        out.append(FunctionTool.from_defaults(fn=_fn, name=t.name, description=t.description or ""))
    return out


def to_phidata_tools(tools: Sequence[McpTool]) -> List[Any]:
    # phidata installs top-level module 'phi'
    phi = pytest.importorskip("phi", reason="phidata not installed")
    tools_mod = getattr(phi, "tools", None)
    if tools_mod is None:
        pytest.skip("phi.tools not available")
    ToolCls = getattr(tools_mod, "Tool", None)
    if ToolCls is None:
        pytest.skip("phi.tools.Tool class not available in this version")

    out: List[Any] = []
    for t in tools:
        def _run(**kwargs: Any) -> Dict[str, Any]:
            return {"called": t.name, "args": kwargs}

        out.append(ToolCls(name=t.name, description=t.description or "", run=_run))
    return out


def to_semantic_kernel_tools(tools: Sequence[McpTool]) -> List[Any]:
    sk = pytest.importorskip("semantic_kernel", reason="semantic-kernel not installed")
    # For contract purposes, simply return OpenAI function tools which SK can consume via connectors.
    return to_openai_function_tools(tools)


def to_autogen_tools(tools: Sequence[McpTool]) -> List[Dict[str, Any]]:
    pytest.importorskip("autogen_agentchat", reason="autogen-agentchat not installed")
    # Many frameworks accept OpenAI function tool schema; return that representation.
    return to_openai_function_tools(tools)


def to_ag2_tools(tools: Sequence[McpTool]) -> List[Dict[str, Any]]:
    pytest.importorskip("ag2", reason="ag2 not installed")
    # Represent tools using OpenAI function schema as common denominator.
    return to_openai_function_tools(tools)


# Per-framework sanity checks (sync direct as representative)

def test_autogen_adapter_sync_direct_tools() -> None:
    transport = _mock_transport_direct()
    http_client = httpx.Client(transport=transport, base_url="http://mock")
    cfg = McpClientConfig(servers=[ServerConfig(name="s1", endpoint_url="http://mock/mcp")])
    client = McpClient.from_config(cfg, direct_http_client=http_client)
    tools = client.list_tools("s1")
    oa_tools = to_autogen_tools(tools)
    assert oa_tools and oa_tools[0]["function"]["name"] == "echo"


def test_ag2_adapter_sync_direct_tools() -> None:
    transport = _mock_transport_direct()
    http_client = httpx.Client(transport=transport, base_url="http://mock")
    cfg = McpClientConfig(servers=[ServerConfig(name="s1", endpoint_url="http://mock/mcp")])
    client = McpClient.from_config(cfg, direct_http_client=http_client)
    tools = client.list_tools("s1")
    oa_tools = to_ag2_tools(tools)
    assert oa_tools and oa_tools[0]["function"]["name"] == "echo"


def test_langgraph_adapter_sync_direct_tools() -> None:
    pytest.importorskip("langgraph", reason="langgraph not installed")
    # Use LangChain tool mapping as underlying representation for LangGraph nodes.
    transport = _mock_transport_direct()
    http_client = httpx.Client(transport=transport, base_url="http://mock")
    cfg = McpClientConfig(servers=[ServerConfig(name="s1", endpoint_url="http://mock/mcp")])
    client = McpClient.from_config(cfg, direct_http_client=http_client)
    tools = client.list_tools("s1")
    lc_tools = to_langchain_tools(tools)
    assert lc_tools and getattr(lc_tools[0], "name", None) == "echo"


def test_crewai_adapter_sync_direct_tools() -> None:
    transport = _mock_transport_direct()
    http_client = httpx.Client(transport=transport, base_url="http://mock")
    cfg = McpClientConfig(servers=[ServerConfig(name="s1", endpoint_url="http://mock/mcp")])
    client = McpClient.from_config(cfg, direct_http_client=http_client)
    tools = client.list_tools("s1")
    cr_tools = to_crewai_tools(tools)
    assert cr_tools and getattr(cr_tools[0], "name", None) == "echo"


def test_llamaindex_adapter_sync_direct_tools() -> None:
    transport = _mock_transport_direct()
    http_client = httpx.Client(transport=transport, base_url="http://mock")
    cfg = McpClientConfig(servers=[ServerConfig(name="s1", endpoint_url="http://mock/mcp")])
    client = McpClient.from_config(cfg, direct_http_client=http_client)
    tools = client.list_tools("s1")
    li_tools = to_llamaindex_tools(tools)
    assert li_tools and len(li_tools) > 0


def test_phidata_adapter_sync_direct_tools() -> None:
    transport = _mock_transport_direct()
    http_client = httpx.Client(transport=transport, base_url="http://mock")
    cfg = McpClientConfig(servers=[ServerConfig(name="s1", endpoint_url="http://mock/mcp")])
    client = McpClient.from_config(cfg, direct_http_client=http_client)
    tools = client.list_tools("s1")
    pd_tools = to_phidata_tools(tools)
    # Ensure we created objects with a name attribute when available
    first = pd_tools[0]
    name = getattr(first, "name", None) or getattr(first, "tool_name", None)
    assert name == "echo"


def test_semantic_kernel_adapter_sync_direct_tools() -> None:
    transport = _mock_transport_direct()
    http_client = httpx.Client(transport=transport, base_url="http://mock")
    cfg = McpClientConfig(servers=[ServerConfig(name="s1", endpoint_url="http://mock/mcp")])
    client = McpClient.from_config(cfg, direct_http_client=http_client)
    tools = client.list_tools("s1")
    sk_tools = to_semantic_kernel_tools(tools)
    assert sk_tools and sk_tools[0]["function"]["name"] == "echo"
