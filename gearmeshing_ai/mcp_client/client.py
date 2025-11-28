from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Protocol, Sequence

from .config import MCPConfig
from .models import ToolMetadata, ToolResult
from .policy import Policy, PolicyDecision
from .strategy import DirectStrategy, GatewayStrategy


class Strategy(Protocol):
    def list_tools(self) -> Sequence[ToolMetadata]: ...

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult: ...

    def stream_call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Iterable[ToolResult]: ...


class MCPClient:
    """Entry-point client that applies policy and delegates to a transport strategy."""

    def __init__(
        self, config: Optional[MCPConfig] = None, policy: Optional[Policy] = None, strategy: Optional[Strategy] = None
    ) -> None:
        self._config = config or MCPConfig.from_env()
        self._policy = policy or Policy.from_env()
        self._strategy: Strategy = strategy or self._make_strategy(self._config)
        self._role = self._detect_role()

    def _make_strategy(self, config: MCPConfig) -> Strategy:
        if config.strategy == "direct":
            return DirectStrategy(config)
        return GatewayStrategy(config)

    def _detect_role(self) -> str:
        # In real usage, derive from caller identity/session/context
        return "dev"

    def list_tools(self) -> Sequence[ToolMetadata]:
        return self._strategy.list_tools()

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        decision: PolicyDecision = self._policy.can_call(self._role, tool_name)
        if not decision.allowed:
            return ToolResult(ok=False, error=decision.reason or "Blocked by policy")
        return self._strategy.call_tool(tool_name, arguments)

    def stream_call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Iterable[ToolResult]:
        decision: PolicyDecision = self._policy.can_call(self._role, tool_name)
        if not decision.allowed:
            yield ToolResult(ok=False, error=decision.reason or "Blocked by policy")
            return
        yield from self._strategy.stream_call_tool(tool_name, arguments)
