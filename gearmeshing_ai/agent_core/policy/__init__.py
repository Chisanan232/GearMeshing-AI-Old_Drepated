"""Policy subsystem for agent autonomy, safety, and approvals.

The policy layer provides *runtime* decisions for action execution. It is
intentionally separate from prompting and planning so that:

- Thought steps remain LLM-only and never touch tools.
- Action steps are centrally governed by configuration.

Components
----------

- ``ToolPolicy``: allow/deny which capabilities (and optionally which MCP
  servers) are permitted.
- ``ApprovalPolicy`` and ``AutonomyProfile``: decide whether a particular
  action requires human approval.
- ``SafetyPolicy``: performs basic safety checks (e.g. size limits, prompt
  injection detection, secret redaction).
- ``BudgetPolicy``: placeholder for token/cost budgeting.

``GlobalPolicy`` aggregates these configurations and exposes methods used by
the engine such as risk classification, allow/deny decisions, approval gating,
and redaction.
"""

from .global_policy import GlobalPolicy
from gearmeshing_ai.core.models.domain.policy import (
    ApprovalPolicy,
    BudgetPolicy,
    PolicyConfig,
    SafetyPolicy,
    ToolPolicy,
)
from .provider import PolicyProvider, StaticPolicyProvider

__all__ = [
    "GlobalPolicy",
    "PolicyConfig",
    "ToolPolicy",
    "ApprovalPolicy",
    "SafetyPolicy",
    "BudgetPolicy",
    "PolicyProvider",
    "StaticPolicyProvider",
]
