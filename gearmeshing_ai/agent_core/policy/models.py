from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pydantic import Field

from ..schemas.base import BaseSchema
from ..schemas.domain import AutonomyProfile, CapabilityName, RiskLevel


class ToolRiskKind(str, Enum):
    """
    Classification of tool operations for risk assessment.
    
    Attributes:
        read: Read-only operations (low risk).
        write: State-modifying operations (medium risk).
        high: Critical or dangerous operations (high risk).
    """
    read = "read"
    write = "write"
    high = "high"


class ToolPolicy(BaseSchema):
    """
    Configuration for tool allow/deny lists.

    Controls which capabilities and tools the agent is permitted to invoke.
    Default behavior is permissive unless allow-lists are configured.
    """
    allowed_capabilities: Optional[set[CapabilityName]] = Field(
        default=None,
        description="If set, only these capabilities may be executed.",
    )

    allowed_tools: Optional[set[str]] = Field(
        default=None,
        description=(
            "If set, only these logical tool names may be executed. "
            "Logical tool names are free-form strings like 'scm.create_pr'."
        ),
    )
    blocked_tools: set[str] = Field(
        default_factory=set,
        description="Logical tool names in this set will be blocked.",
    )
    allowed_mcp_servers: Optional[set[str]] = Field(
        default=None,
        description="If set, MCP calls must target one of these server ids/slugs.",
    )
    blocked_mcp_servers: set[str] = Field(
        default_factory=set,
        description="MCP calls targeting these server ids/slugs will be blocked.",
    )


class ApprovalPolicy(BaseSchema):
    """
    Configuration for human-in-the-loop approval gates.

    Determines when an action requires explicit human authorization based on its
    risk level.
    """
    require_for_risk_at_or_above: RiskLevel = RiskLevel.medium
    approval_ttl_seconds: float = Field(default=900.0, ge=0.0, le=86400.0)

    tool_risk_overrides: dict[str, RiskLevel] = Field(
        default_factory=dict,
        description=(
            "Optional per-logical-tool risk overrides. Keys are logical tool names "
            "(e.g. 'scm.merge_pr') and values are RiskLevel."
        ),
    )

    tool_risk_kinds: dict[str, ToolRiskKind] = Field(
        default_factory=dict,
        description=(
            "Optional per-logical-tool risk kind classification. Keys are logical tool names "
            "(e.g. 'tracker.update_task') and values are ToolRiskKind (read/write/high)."
        ),
    )


class SafetyPolicy(BaseSchema):
    """
    Configuration for safety guardrails.

    Includes settings for prompt injection detection, secret redaction, and
    resource limits to prevent abuse or accidents.
    """
    block_prompt_injection: bool = True
    redact_secrets: bool = True
    max_tool_args_bytes: int = Field(default=64_000, ge=1, le=5_000_000)


class BudgetPolicy(BaseSchema):
    """
    Configuration for resource budgeting.

    Limits the consumption of tokens or costs for an agent run.
    """
    max_total_tokens: Optional[int] = Field(default=None, ge=1)


class PolicyConfig(BaseSchema):
    """
    Aggregate configuration object for all policy aspects.

    This is the root configuration object used to instantiate a ``GlobalPolicy``.
    """
    version: str = Field(default="policy-v1")

    autonomy_profile: AutonomyProfile = AutonomyProfile.balanced
    tool_policy: ToolPolicy = Field(default_factory=ToolPolicy)
    approval_policy: ApprovalPolicy = Field(default_factory=ApprovalPolicy)
    safety_policy: SafetyPolicy = Field(default_factory=SafetyPolicy)
    budget_policy: BudgetPolicy = Field(default_factory=BudgetPolicy)


@dataclass(frozen=True)
class PolicyDecision:
    """
    Result of a policy evaluation for a specific action.

    Attributes:
        risk: The assessed risk level of the action.
        require_approval: Whether human approval is needed before execution.
        block: Whether the action is explicitly blocked by policy.
        block_reason: Human-readable reason if the action is blocked.
    """
    risk: RiskLevel
    require_approval: bool
    block: bool
    block_reason: Optional[str]


def _risk_ge(a: RiskLevel, b: RiskLevel) -> bool:
    """Check if risk level 'a' is greater than or equal to 'b'."""
    order = {RiskLevel.low: 0, RiskLevel.medium: 1, RiskLevel.high: 2}
    return order[a] >= order[b]


def risk_from_kind(kind: ToolRiskKind) -> RiskLevel:
    """Map a generic tool risk kind (read/write) to a specific RiskLevel."""
    if kind == ToolRiskKind.read:
        return RiskLevel.low
    if kind == ToolRiskKind.write:
        return RiskLevel.medium
    return RiskLevel.high


def risk_requires_approval(risk: RiskLevel, *, profile: AutonomyProfile, policy: ApprovalPolicy) -> bool:
    """
    Determine if approval is required based on risk and autonomy profile.

    Args:
        risk: The assessed risk level of the action.
        profile: The autonomy profile for the run (e.g. strict, balanced).
        policy: The approval policy configuration.

    Returns:
        True if approval is required, False otherwise.
    """
    if profile == AutonomyProfile.unrestricted:
        return _risk_ge(risk, RiskLevel.high)
    if profile == AutonomyProfile.strict:
        return True
    return _risk_ge(risk, policy.require_for_risk_at_or_above)
