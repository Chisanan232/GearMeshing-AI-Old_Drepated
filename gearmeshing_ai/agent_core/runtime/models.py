from __future__ import annotations

"""Runtime dependency bundle and LangGraph state types.

The runtime engine is designed to be dependency-injected.

- ``EngineDeps`` collects the repositories and registries the engine needs.
- ``_GraphState`` is the mutable state passed between LangGraph nodes.

These structures are intentionally small and serializable so they can be
checkpointed for pause/resume.
"""

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
    """Dependency bundle for ``AgentEngine``.

    This object is typically constructed by application wiring code and passed
    into the engine (or ``AgentService``). It holds:

    - persistence repositories (runs, events, approvals, checkpoints, tool
      invocations)
    - the capability registry used to resolve Action steps.
    """
    runs: RunRepository
    events: EventRepository
    approvals: ApprovalRepository
    checkpoints: CheckpointRepository
    tool_invocations: ToolInvocationRepository

    capabilities: CapabilityRegistry


class _GraphState(TypedDict):
    """Mutable LangGraph state for a single engine run.

    Required keys:

    - ``run_id``: current run identifier.
    - ``plan``: normalized list of step dicts.
    - ``idx``: current plan index.
    - ``awaiting_approval_id``: set when the run is paused for approval.

    Optional keys:

    - ``_finished`` / ``_terminal_status``: used to terminate the graph.
    - ``_resume_skip_approval``: internal one-shot flag used to avoid re-pausing
      immediately after a resume.
    """
    run_id: Required[str]
    plan: Required[List[Dict[str, Any]]]
    idx: Required[int]
    awaiting_approval_id: Required[Optional[str]]
    _finished: NotRequired[bool]
    _terminal_status: NotRequired[str]
    _resume_skip_approval: NotRequired[bool]
