"""Schemas and DTOs for the agent core.

This package re-exports models from the centralized core.models package
for backward compatibility. New code should import directly from core.models.

The models are intentionally explicit and serializable so they can be:

- persisted to SQL storage,
- emitted as audit events,
- used as stable interfaces between subsystems.
"""

from gearmeshing_ai.core.models import (
    AgentEvent,
    AgentEventType,
    AgentRun,
    AgentRunStatus,
    Approval,
    ApprovalDecision,
    AutonomyProfile,
    BaseSchema,
    Checkpoint,
    ModelConfig,
    RiskLevel,
    RoleConfig,
    ToolInvocation,
    UsageLedgerEntry,
)

__all__ = [
    "BaseSchema",
    "ModelConfig",
    "RoleConfig",
    "AgentRun",
    "AgentEvent",
    "Approval",
    "ApprovalDecision",
    "AutonomyProfile",
    "RiskLevel",
    "Checkpoint",
    "ToolInvocation",
    "UsageLedgerEntry",
    "AgentRunStatus",
    "AgentEventType",
]
