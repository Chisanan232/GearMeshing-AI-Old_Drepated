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
from .models import PolicyConfig, PolicyDecision, risk_from_kind, risk_requires_approval

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
        """
        Classify the risk level for a capability invocation.

        This method determines the inherent risk of an action. It checks for:
        1. Explicit overrides in ``approval_policy.tool_risk_overrides``.
        2. Risk kind (read/write) in ``approval_policy.tool_risk_kinds``.
        3. Built-in defaults based on the capability type (e.g., shell_exec is high risk).

        Args:
            capability: The capability being invoked.
            args: The arguments passed to the capability.
            logical_tool: The logical tool name (optional, for MCP tools).

        Returns:
            The calculated RiskLevel (low, medium, high).
        """
        if logical_tool is not None:
            override = self._cfg.approval_policy.tool_risk_overrides.get(logical_tool)
            if override is not None:
                return override

            kind = self._cfg.approval_policy.tool_risk_kinds.get(logical_tool)
            if kind is not None:
                return risk_from_kind(kind)

        if capability == CapabilityName.mcp_call:
            mut = args.get("_mcp_tool_mutating")
            if mut is False:
                return RiskLevel.low
            return RiskLevel.medium

        if capability in {CapabilityName.shell_exec, CapabilityName.codegen, CapabilityName.code_execution}:
            return RiskLevel.high
        if capability in {CapabilityName.mcp_call}:
            return RiskLevel.medium
        return RiskLevel.low

    def decide(
        self, capability: CapabilityName, *, args: Dict[str, Any], logical_tool: str | None = None
    ) -> PolicyDecision:
        """
        Compute a comprehensive policy decision for an action.

        This is the main entry point for policy enforcement. It evaluates:
        1. Allow/block lists for capabilities.
        2. Allow/block lists for logical tools.
        3. Allow/block lists for MCP servers.
        4. Risk classification.
        5. Approval requirements based on autonomy profile.

        Args:
            capability: The capability to evaluate.
            args: The capability arguments.
            logical_tool: Optional logical tool name.

        Returns:
            A PolicyDecision object containing the verdict (block, approval needed, risk).
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
            if (
                self._cfg.tool_policy.allowed_tools is not None
                and logical_tool not in self._cfg.tool_policy.allowed_tools
            ):
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
        """
        Validate tool arguments against basic safety constraints.

        Checks generic safety rules, such as maximum argument size, to prevent
        DoS or buffer overflow issues in downstream systems.

        Args:
            args: The dictionary of arguments to validate.

        Returns:
            An error string if validation fails, otherwise None.
        """
        raw = json.dumps(args, default=str).encode("utf-8")
        if len(raw) > self._cfg.safety_policy.max_tool_args_bytes:
            return "tool args too large"
        return None

    def redact(self, text: str) -> str:
        """
        Redact known secrets from text.

        Applies regex patterns configured in the system to scrub API keys and
        credentials from text (e.g., logs, summaries).

        Args:
            text: The input text.

        Returns:
            The sanitized text with secrets replaced by '<redacted>'.
        """
        if not self._cfg.safety_policy.redact_secrets:
            return text
        out = text
        for pat in _SECRET_PATTERNS:
            out = pat.sub("<redacted>", out)
        return out

    def detect_prompt_injection(self, text: str) -> bool:
        """
        Heuristic detection of prompt injection attacks.

        Scans text for common injection patterns (e.g., "ignore previous instructions").
        Used to guard against malicious user inputs or tool outputs.

        Args:
            text: The text to analyze.

        Returns:
            True if potential injection is detected, False otherwise.
        """
        if not self._cfg.safety_policy.block_prompt_injection:
            return False
        lowered = text.lower()
        return "ignore previous" in lowered or "system prompt" in lowered
