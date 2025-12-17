"""Planning components.

 The planning subsystem is responsible for producing a *plan* from a user
 objective and role. A plan is a list of step dictionaries that conform to the
 step schemas defined in ``gearmeshing_ai.agent_core.planning.steps``.

 Output model
 ------------

 The system enforces a strict split between:

 - ``ThoughtStep``: LLM-only / cognitive steps.
 - ``ActionStep``: side-effecting steps that may invoke capabilities/tools and
   may require policy checks and approvals.

 The planner itself does not execute tools; it only emits structured steps that
 are later consumed by ``gearmeshing_ai.agent_core.runtime.AgentEngine``.
 """

from .planner import StructuredPlanner
from .steps import ActionStep, PlanStep, ThoughtStep

__all__ = [
    "ActionStep",
    "PlanStep",
    "StructuredPlanner",
    "ThoughtStep",
]
