"""LangGraph-based execution runtime for agent runs.

 The runtime is responsible for taking a normalized plan (a list of step
 dictionaries produced by the planning subsystem) and executing it with strong
 guarantees:

 - ``ThoughtStep`` steps are *non-side-effecting* and are recorded as events and
   artifacts.
 - ``ActionStep`` steps are side-effecting and are routed through policy,
   optional approval, and capability execution.

 The main entry point is ``AgentEngine``.

 Execution is persisted and auditable via ``EngineDeps``, which provides
 repository interfaces for runs, events, approvals, checkpoints, and tool
 invocations.
 """

from .engine import AgentEngine
from .models import EngineDeps

__all__ = [
    "AgentEngine",
    "EngineDeps",
]
