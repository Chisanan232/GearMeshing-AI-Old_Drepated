from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

from ..capabilities import CapabilityRegistry
from ..repos import RunRepository, EventRepository, ApprovalRepository, CheckpointRepository, \
    ToolInvocationRepository


@dataclass(frozen=True)
class EngineDeps:
    runs: RunRepository
    events: EventRepository
    approvals: ApprovalRepository
    checkpoints: CheckpointRepository
    tool_invocations: ToolInvocationRepository

    capabilities: CapabilityRegistry


class _GraphState(TypedDict, total=False):
    run_id: str
    plan: List[Dict[str, Any]]
    idx: int
    awaiting_approval_id: Optional[str]
    _finished: bool
    _terminal_status: str
