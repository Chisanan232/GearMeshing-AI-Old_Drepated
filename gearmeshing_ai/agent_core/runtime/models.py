from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, NotRequired, Optional, Required, TypedDict

from ..capabilities import CapabilityRegistry
from ..repos import (
    ApprovalRepository,
    CheckpointRepository,
    EventRepository,
    RunRepository,
    ToolInvocationRepository,
)


@dataclass(frozen=True)
class EngineDeps:
    runs: RunRepository
    events: EventRepository
    approvals: ApprovalRepository
    checkpoints: CheckpointRepository
    tool_invocations: ToolInvocationRepository

    capabilities: CapabilityRegistry


class _GraphState(TypedDict):
    run_id: Required[str]
    plan: Required[List[Dict[str, Any]]]
    idx: Required[int]
    awaiting_approval_id: Required[Optional[str]]
    _finished: NotRequired[bool]
    _terminal_status: NotRequired[str]
    _resume_skip_approval: NotRequired[bool]
