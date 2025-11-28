from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Set


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: Optional[str] = None


class Policy:
    """Simple policy engine to allow/deny tool calls based on role and tool name."""

    def __init__(self, allow_by_role: Optional[Dict[str, Set[str]]] = None, default_allow: bool = True) -> None:
        # Map role -> allowed tool names (exact matches)
        self._allow_by_role: Dict[str, Set[str]] = allow_by_role or {}
        self._default_allow = default_allow

    def can_call(self, role: str, tool_name: str) -> PolicyDecision:
        allowed_set = self._allow_by_role.get(role)
        if allowed_set is None:
            return PolicyDecision(allowed=self._default_allow)
        return PolicyDecision(
            allowed=(tool_name in allowed_set),
            reason=None if tool_name in allowed_set else f"Tool '{tool_name}' not allowed for role '{role}'",
        )

    @staticmethod
    def from_env(default_allow: bool = True) -> "Policy":
        # Optionally parse a CSV env var like MCP_ALLOWED_TOOLS=dev:tool_a|tool_b;prod:tool_c
        mapping: Dict[str, Set[str]] = {}
        raw = os.getenv("MCP_ALLOWED_TOOLS", "").strip()
        if raw:
            for part in raw.split(";"):
                if not part:
                    continue
                role, _, tools = part.partition(":")
                toolset = set(t for t in tools.split("|") if t)
                if role:
                    mapping[role] = toolset
        return Policy(mapping, default_allow=default_allow)
