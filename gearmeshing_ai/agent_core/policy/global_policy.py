from __future__ import annotations

"""Global policy decisions for Action execution.

``GlobalPolicy`` is the runtime authority used by ``AgentEngine`` to decide how
to handle a side-effecting step.

Design goals
------------

- Centralize allow/deny decisions outside of prompts.
- Provide a single place to implement:

  - capability allow-lists,
  - MCP server allow/block lists,
  - risk classification and approval gating,
  - basic safety validation (size limits),
  - redaction and prompt injection heuristics.

The policy layer only applies to Action steps. Thought steps never call policy.
"""

import json
import re
from typing import Any, Dict, Optional

from ..schemas.domain import CapabilityName, RiskLevel
from .models import PolicyConfig, PolicyDecision, risk_requires_approval

_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{10,}"),
    re.compile(r"ghp_[A-Za-z0-9]{10,}"),
    re.compile(r"xoxb-[A-Za-z0-9-]{10,}"),
)


class GlobalPolicy:
    """Aggregate policy decisions for a single run.

    ``GlobalPolicy`` is configured by ``PolicyConfig`` and provides helper
    methods used by the runtime engine.
    """

    def __init__(self, config: PolicyConfig) -> None:
        self._cfg = config

    @property
    def config(self) -> PolicyConfig:
        """Return the underlying configuration object."""
        return self._cfg

    def classify_risk(
        self,
        capability: CapabilityName,
        *,
        args: Dict[str, Any],
        logical_tool: str | None = None,
    ) -> RiskLevel:
        """Classify the risk level for a capability invocation.

        This is a coarse-grained classifier used to determine whether approval
        is required under the configured autonomy profile.
        """
        if logical_tool is not None:
            override = self._cfg.approval_policy.tool_risk_overrides.get(logical_tool)
            if override is not None:
                return override

        if capability in {CapabilityName.shell_exec, CapabilityName.codegen, CapabilityName.code_execution}:
            return RiskLevel.high
        if capability in {CapabilityName.mcp_call}:
            return RiskLevel.medium
        return RiskLevel.low

    def decide(self, capability: CapabilityName, *, args: Dict[str, Any], logical_tool: str | None = None) -> PolicyDecision:
        """Compute a policy decision for an action.

        The decision includes:

        - whether the action is blocked,
        - a risk level,
        - whether approval is required.
        """
        if self._cfg.tool_policy.allowed_capabilities is not None:
            if capability not in self._cfg.tool_policy.allowed_capabilities:
                return PolicyDecision(
                    risk=RiskLevel.low,
                    require_approval=False,
                    block=True,
                    block_reason=f"capability not allowed: {capability}",
                )

        if logical_tool is not None:
            if logical_tool in self._cfg.tool_policy.blocked_tools:
                return PolicyDecision(
                    risk=RiskLevel.low,
                    require_approval=False,
                    block=True,
                    block_reason=f"tool blocked: {logical_tool}",
                )
            if self._cfg.tool_policy.allowed_tools is not None and logical_tool not in self._cfg.tool_policy.allowed_tools:
                return PolicyDecision(
                    risk=RiskLevel.low,
                    require_approval=False,
                    block=True,
                    block_reason=f"tool not allowed: {logical_tool}",
                )

        if capability == CapabilityName.mcp_call:
            server_id = str(args.get("server_id") or "")
            if server_id and server_id in self._cfg.tool_policy.blocked_mcp_servers:
                return PolicyDecision(
                    risk=RiskLevel.low,
                    require_approval=False,
                    block=True,
                    block_reason=f"mcp server blocked: {server_id}",
                )
            if self._cfg.tool_policy.allowed_mcp_servers is not None:
                if server_id not in self._cfg.tool_policy.allowed_mcp_servers:
                    return PolicyDecision(
                        risk=RiskLevel.low,
                        require_approval=False,
                        block=True,
                        block_reason=f"mcp server not allowed: {server_id}",
                    )

        risk = self.classify_risk(capability, args=args, logical_tool=logical_tool)
        require = risk_requires_approval(risk, profile=self._cfg.autonomy_profile, policy=self._cfg.approval_policy)
        return PolicyDecision(risk=risk, require_approval=require, block=False, block_reason=None)

    def validate_tool_args(self, args: Dict[str, Any]) -> Optional[str]:
        """Validate tool args against safety constraints.

        Returns an error string if validation fails, otherwise ``None``.
        """
        raw = json.dumps(args, default=str).encode("utf-8")
        if len(raw) > self._cfg.safety_policy.max_tool_args_bytes:
            return "tool args too large"
        return None

    def redact(self, text: str) -> str:
        """Redact secrets from text according to the configured safety policy."""
        if not self._cfg.safety_policy.redact_secrets:
            return text
        out = text
        for pat in _SECRET_PATTERNS:
            out = pat.sub("<redacted>", out)
        return out

    def detect_prompt_injection(self, text: str) -> bool:
        """Heuristic prompt-injection detection.

        This is intentionally lightweight and is used as a guardrail for
        user-provided inputs and tool outputs.
        """
        if not self._cfg.safety_policy.block_prompt_injection:
            return False
        lowered = text.lower()
        return "ignore previous" in lowered or "system prompt" in lowered
