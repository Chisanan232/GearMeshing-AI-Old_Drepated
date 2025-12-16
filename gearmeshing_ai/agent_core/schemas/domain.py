from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import Field

from .base import BaseSchema


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AutonomyProfile(str, Enum):
    unrestricted = "unrestricted"
    balanced = "balanced"
    strict = "strict"


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class AgentRole(str, Enum):
    planner = "planner"
    market = "market"
    dev = "dev"
    dev_lead = "dev_lead"
    qa = "qa"
    sre = "sre"


class ApprovalDecision(str, Enum):
    approved = "approved"
    rejected = "rejected"
    expired = "expired"


class CapabilityName(str, Enum):
    web_search = "web_search"
    web_fetch = "web_fetch"
    docs_read = "docs_read"
    summarize = "summarize"
    mcp_call = "mcp_call"
    codegen = "codegen"
    code_execution = "code_execution"
    shell_exec = "shell_exec"


class AgentRunStatus(str, Enum):
    running = "running"
    paused_for_approval = "paused_for_approval"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class AgentEventType(str, Enum):
    run_started = "run.started"
    run_completed = "run.completed"
    run_failed = "run.failed"
    state_transition = "state.transition"
    plan_created = "plan.created"
    capability_requested = "capability.requested"
    capability_executed = "capability.executed"
    tool_invoked = "tool.invoked"
    approval_requested = "approval.requested"
    approval_resolved = "approval.resolved"
    checkpoint_saved = "checkpoint.saved"
    usage_recorded = "usage.recorded"


class AgentRun(BaseSchema):
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
    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str

    type: AgentEventType
    created_at: datetime = Field(default_factory=_utc_now)

    correlation_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class ToolInvocation(BaseSchema):
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
    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str

    risk: RiskLevel
    capability: CapabilityName

    reason: str
    requested_at: datetime = Field(default_factory=_utc_now)

    expires_at: Optional[datetime] = None

    decision: Optional[ApprovalDecision] = None
    decided_at: Optional[datetime] = None
    decided_by: Optional[str] = None


class Checkpoint(BaseSchema):
    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str

    node: str
    state: Dict[str, Any]

    created_at: datetime = Field(default_factory=_utc_now)


class UsageLedgerEntry(BaseSchema):
    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str

    provider: Optional[str] = None
    model: Optional[str] = None

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    cost_usd: Optional[float] = None

    created_at: datetime = Field(default_factory=_utc_now)
