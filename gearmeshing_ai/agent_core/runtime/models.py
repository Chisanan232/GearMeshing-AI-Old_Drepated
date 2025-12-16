from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

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


class _GraphState(Dict[str, Any]):
    pass
