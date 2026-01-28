"""Domain models for agent core entities."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional
from uuid import uuid4

from pydantic import Field

from ..base import BaseSchema
from .enums import AgentEventType, AgentRunStatus, ApprovalDecision, AutonomyProfile, RiskLevel

if TYPE_CHECKING:
    from gearmeshing_ai.info_provider import CapabilityName


def _utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


class AgentRun(BaseSchema):
    """
    Represents a single execution session of an agent.

    A run is scoped to a specific objective and role. It maintains its own state,
    history, and policy configuration.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None

    role: str
    autonomy_profile: AutonomyProfile = AutonomyProfile.balanced

    objective: str
    done_when: Optional[str] = None

    prompt_provider_version: Optional[str] = None

    status: AgentRunStatus = AgentRunStatus.running
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)


class AgentEvent(BaseSchema):
    """
    An immutable event record in the run's history.

    Events capture everything significant that happens during execution, from
    thoughts and plan creation to tool outputs and state changes.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str

    type: AgentEventType
    created_at: datetime = Field(default_factory=_utc_now)

    correlation_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class ToolInvocation(BaseSchema):
    """
    Audit record for a side-effecting tool call.

    Logs the inputs (args) and outputs (result) of a capability execution.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str

    server_id: str
    tool_name: str

    args: Dict[str, Any] = Field(default_factory=dict)
    ok: bool
    result: Dict[str, Any] = Field(default_factory=dict)

    risk: RiskLevel = RiskLevel.low

    created_at: datetime = Field(default_factory=_utc_now)


class Approval(BaseSchema):
    """
    A request for human approval.

    Created when the policy engine blocks an action due to risk.
    The run pauses until a decision is recorded.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str

    risk: RiskLevel
    capability: "CapabilityName"  # Forward reference to avoid circular import

    reason: str
    requested_at: datetime = Field(default_factory=_utc_now)

    expires_at: Optional[datetime] = None

    decision: Optional[ApprovalDecision] = None
    decided_at: Optional[datetime] = None
    decided_by: Optional[str] = None


class Checkpoint(BaseSchema):
    """
    Snapshot of the execution state.

    Stores the serialized LangGraph state to allow pausing and resuming the run across
    process restarts or approval waits.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str

    node: str
    state: Dict[str, Any]

    created_at: datetime = Field(default_factory=_utc_now)


class UsageLedgerEntry(BaseSchema):
    """
    Record of token consumption and cost.

    Used for accounting and billing. Tracks usage per model invocation.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str

    provider: Optional[str] = None
    model: Optional[str] = None

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    cost_usd: Optional[float] = None

    created_at: datetime = Field(default_factory=_utc_now)


def _resolve_forward_references() -> None:
    """
    Resolve forward references in models after all imports are complete.

    This function is called after all imports are resolved to ensure that
    forward references (like CapabilityName in Approval) are properly resolved.
    """
    try:
        from gearmeshing_ai.info_provider import CapabilityName  # noqa: F401

        Approval.model_rebuild()
    except ImportError:
        # If CapabilityName is not available, skip rebuilding
        pass
