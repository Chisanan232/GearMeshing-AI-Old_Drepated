"""Domain enums for agent core models."""

from __future__ import annotations

from enum import Enum


class AutonomyProfile(str, Enum):
    """
    Defines the level of autonomy granted to an agent run.

    This profile controls approval requirements in the ``GlobalPolicy``.
    """

    unrestricted = "unrestricted"  # Requires approval only for high-risk actions.
    balanced = "balanced"  # Default: requires approval for medium/high risk.
    strict = "strict"  # Requires approval for almost all side effects.


class RiskLevel(str, Enum):
    """
    Risk classification for capabilities and tools.

    Used by the policy engine to determine if an action requires approval.
    """

    low = "low"  # Read-only or safe operations.
    medium = "medium"  # State-modifying but reversible or low-impact operations.
    high = "high"  # Critical, expensive, or irreversible operations (e.g. shell exec).


class ApprovalDecision(str, Enum):
    """Possible outcomes for an approval request."""

    approved = "approved"
    rejected = "rejected"
    expired = "expired"


class AgentRunStatus(str, Enum):
    """Lifecycle status of an agent run."""

    pending = "pending"
    running = "running"
    paused_for_approval = "paused_for_approval"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class AgentEventType(str, Enum):
    """
    Types of events emitted during an agent run.

    These events form the audit log and the event stream.
    """

    run_started = "run.started"
    run_completed = "run.completed"
    run_failed = "run.failed"
    run_cancelled = "run.cancelled"
    state_transition = "state.transition"
    plan_created = "plan.created"
    thought_executed = "thought.executed"
    artifact_created = "artifact.created"
    capability_requested = "capability.requested"
    capability_executed = "capability.executed"
    tool_invoked = "tool.invoked"
    approval_requested = "approval.requested"
    approval_resolved = "approval.resolved"
    checkpoint_saved = "checkpoint.saved"
    usage_recorded = "usage.recorded"
