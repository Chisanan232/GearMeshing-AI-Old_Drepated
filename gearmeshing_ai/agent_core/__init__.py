"""Core AI agent runtime, policies, and persistence abstractions.

This package contains the “engine room” of the agent system.

Design overview
---------------

The agent core is built around a strict separation of *planning/thought* vs
*execution/action*:

- Planning produces a plan consisting of two step kinds:

  - ``ThoughtStep``: cognitive/LLM-only work. These steps never call tools,
    never require approval, and only emit structured artifacts/events.
  - ``ActionStep``: side-effecting execution. These steps pass through policy
    checks (allow/deny, safety validation) and may require approval.

- Execution is performed by ``agent_core.runtime.AgentEngine`` using LangGraph.
  The engine persists an auditable trail (events, approvals, checkpoints, tool
  invocations) via repository interfaces.

Typical usage
-------------

Most applications should use ``agent_core.service.AgentService`` to orchestrate
the full run:

1. Create an ``AgentRun``.
2. Use the planner to build a plan.
3. Execute via the engine.
4. If approval is required, resume after approval resolution.
"""

from .agent_registry import AgentRegistry
from .model_provider import (
    ModelProvider,
    create_model_for_role,
    get_model_provider,
)
from .router import Router
from .schemas.domain import (
    AgentEvent,
    AgentRun,
    Approval,
    ApprovalDecision,
    AutonomyProfile,
    RiskLevel,
)
from gearmeshing_ai.info_provider import CapabilityName

__all__ = [
    "AgentRun",
    "AgentEvent",
    "Approval",
    "ApprovalDecision",
    "AutonomyProfile",
    "CapabilityName",
    "RiskLevel",
    "AgentRegistry",
    "Router",
    # Model provider
    "ModelProvider",
    "get_model_provider",
    "create_model_for_role",
]
