from __future__ import annotations

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
    def __init__(self, config: PolicyConfig) -> None:
        self._cfg = config

    @property
    def config(self) -> PolicyConfig:
        return self._cfg

    def classify_risk(self, capability: CapabilityName, *, args: Dict[str, Any]) -> RiskLevel:
        if capability in {CapabilityName.shell_exec, CapabilityName.codegen}:
            return RiskLevel.high
        if capability in {CapabilityName.mcp_call}:
            return RiskLevel.medium
        return RiskLevel.low

    def decide(self, capability: CapabilityName, *, args: Dict[str, Any]) -> PolicyDecision:
        if self._cfg.tool_policy.allowed_capabilities is not None:
            if capability not in self._cfg.tool_policy.allowed_capabilities:
                return PolicyDecision(
                    risk=RiskLevel.low,
                    require_approval=False,
                    block=True,
                    block_reason=f"capability not allowed: {capability}",
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

        risk = self.classify_risk(capability, args=args)
        require = risk_requires_approval(risk, profile=self._cfg.autonomy_profile, policy=self._cfg.approval_policy)
        return PolicyDecision(risk=risk, require_approval=require, block=False, block_reason=None)

    def validate_tool_args(self, args: Dict[str, Any]) -> Optional[str]:
        raw = json.dumps(args, default=str).encode("utf-8")
        if len(raw) > self._cfg.safety_policy.max_tool_args_bytes:
            return "tool args too large"
        return None

    def redact(self, text: str) -> str:
        if not self._cfg.safety_policy.redact_secrets:
            return text
        out = text
        for pat in _SECRET_PATTERNS:
            out = pat.sub("<redacted>", out)
        return out

    def detect_prompt_injection(self, text: str) -> bool:
        if not self._cfg.safety_policy.block_prompt_injection:
            return False
        lowered = text.lower()
        return "ignore previous" in lowered or "system prompt" in lowered
