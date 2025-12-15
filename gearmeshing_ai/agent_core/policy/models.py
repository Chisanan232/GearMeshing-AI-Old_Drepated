from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set

from pydantic import Field

from ..schemas.base import BaseSchema
from ..schemas.domain import AutonomyProfile, CapabilityName, RiskLevel


class ToolPolicy(BaseSchema):
    allowed_capabilities: Optional[set[CapabilityName]] = Field(
        default=None,
        description="If set, only these capabilities may be executed.",
    )


class ApprovalPolicy(BaseSchema):
    require_for_risk_at_or_above: RiskLevel = RiskLevel.high
    approval_ttl_seconds: float = Field(default=900.0, ge=0.0, le=86400.0)


class SafetyPolicy(BaseSchema):
    block_prompt_injection: bool = True
    redact_secrets: bool = True
    max_tool_args_bytes: int = Field(default=64_000, ge=1, le=5_000_000)


class BudgetPolicy(BaseSchema):
    max_total_tokens: Optional[int] = Field(default=None, ge=1)


class PolicyConfig(BaseSchema):
    version: str = Field(default="policy-v1")

    autonomy_profile: AutonomyProfile = AutonomyProfile.balanced
    tool_policy: ToolPolicy = Field(default_factory=ToolPolicy)
    approval_policy: ApprovalPolicy = Field(default_factory=ApprovalPolicy)
    safety_policy: SafetyPolicy = Field(default_factory=SafetyPolicy)
    budget_policy: BudgetPolicy = Field(default_factory=BudgetPolicy)


@dataclass(frozen=True)
class PolicyDecision:
    risk: RiskLevel
    require_approval: bool
    block: bool
    block_reason: Optional[str]


def _risk_ge(a: RiskLevel, b: RiskLevel) -> bool:
    order = {RiskLevel.low: 0, RiskLevel.medium: 1, RiskLevel.high: 2}
    return order[a] >= order[b]


def risk_requires_approval(risk: RiskLevel, *, profile: AutonomyProfile, policy: ApprovalPolicy) -> bool:
    if profile == AutonomyProfile.unrestricted:
        return False
    if profile == AutonomyProfile.strict:
        return True
    return _risk_ge(risk, policy.require_for_risk_at_or_above)
