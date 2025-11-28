from gearmeshing_ai.mcp_client.strategy.direct import DirectStrategy
from gearmeshing_ai.mcp_client.config import MCPConfig


def test_direct_strategy_skeleton():
    s = DirectStrategy(MCPConfig())
    assert s.list_tools() == []
    res_iter = list(s.stream_call_tool("echo", {"text": "hi"}))
    assert len(res_iter) == 1
    assert res_iter[0].ok is True
