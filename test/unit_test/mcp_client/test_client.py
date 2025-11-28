from typing import Any

from gearmeshing_ai.mcp_client.client import MCPClient
from gearmeshing_ai.mcp_client.config import MCPConfig
from gearmeshing_ai.mcp_client.models import ToolResult
from gearmeshing_ai.mcp_client.policy import Policy


def test_client_constructs_with_defaults() -> None:
    c: MCPClient = MCPClient()
    assert c.list_tools() == []


def test_client_policy_blocks_call(monkeypatch: Any) -> None:
    class DenyAll(Policy):
        def can_call(self, role: str, tool_name: str):  # type: ignore[override]
            from gearmeshing_ai.mcp_client.policy import PolicyDecision

            return PolicyDecision(allowed=False, reason="denied")

    c: MCPClient = MCPClient(config=MCPConfig(strategy="gateway"), policy=DenyAll())
    res: ToolResult = c.call_tool("echo", {"text": "hi"})
    assert res.ok is False
    assert res.error == "denied"
