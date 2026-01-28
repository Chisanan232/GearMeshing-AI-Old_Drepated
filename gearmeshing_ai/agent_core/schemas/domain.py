"""Domain models for agent core.

This module re-exports domain models from the centralized core.models.domain package
for backward compatibility. New code should import directly from core.models.domain.
"""

from __future__ import annotations

from gearmeshing_ai.core.models.domain import (
    AgentEvent,
    AgentEventType,
    AgentRun,
    AgentRunStatus,
    Approval,
    ApprovalDecision,
    AutonomyProfile,
    Checkpoint,
    RiskLevel,
    ToolInvocation,
    UsageLedgerEntry,
)

__all__ = [
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


# Rebuild models to resolve forward references
# This is called after all imports are resolved
def _resolve_forward_references():
    """Resolve forward references in models."""
    from gearmeshing_ai.info_provider import (  # noqa: F401  # Required for Pydantic forward reference resolution
        CapabilityName,
    )

    Approval.model_rebuild()
