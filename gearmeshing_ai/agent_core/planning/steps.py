from __future__ import annotations

"""Plan step schemas and normalization.

A *plan* is a list of dictionaries that represent the agent’s intended work.
The runtime treats plans as untrusted input and normalizes/validates them.

Thought vs Action
-----------------

The system enforces a strict boundary:

- ``ThoughtStep``: cognitive operations that must never trigger side effects.
  These steps are executed by emitting events/artifacts only.
- ``ActionStep``: side-effecting operations that are routed through policy,
  optional approval, and capability execution.

Normalization
-------------

``normalize_plan`` exists to:

- provide a stable runtime representation (always includes a ``kind`` field),
- enforce that thought steps cannot contain action/tool fields,
- preserve backward compatibility for legacy plans that used the older
  ``{"capability": ..., "args": ...}`` structure.
"""

from typing import Any, Dict, List, Literal, Union

from pydantic import Field

from gearmeshing_ai.info_provider import CapabilityName
from gearmeshing_ai.core.models.base import BaseSchema


class ThoughtStep(BaseSchema):
    """Cognitive (LLM-only) plan step.

    A thought step expresses intent for non-side-effecting work such as
    reasoning, planning, or producing a structured artifact.

    Runtime rules:

    - Must not invoke tools/capabilities.
    - Must not require approval.
    - May only emit events/artifacts.
    """

    kind: Literal["thought"] = "thought"
    thought: str
    args: Dict[str, Any] = Field(default_factory=dict)


class ActionStep(BaseSchema):
    """Side-effecting plan step.

    Action steps execute through the capability system and are centrally
    governed by policy.

    Fields
    ------
    capability:
        Logical capability name (resolved through ``CapabilityRegistry``).
    args:
        JSON-serializable arguments passed to the capability implementation.
    server_id/tool_name:
        Optional transport identifiers used when an action maps to an MCP tool.
        These are persisted in tool invocation logs for auditability.
    """

    kind: Literal["action"] = "action"
    capability: CapabilityName
    args: Dict[str, Any] = Field(default_factory=dict)
    logical_tool: str | None = None
    server_id: str | None = None
    tool_name: str | None = None


PlanStep = Union[ThoughtStep, ActionStep]
"""
PlanStep:
    A type union representing any valid step in an agent plan.
    Can be either a ``ThoughtStep`` (cognitive) or an ``ActionStep`` (side-effecting).
"""


def normalize_plan(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize and validate a plan.

    Parameters
    ----------
    plan:
        A list of dictionaries that represent plan steps. These dicts may be
        produced by the planner, tests, or external callers.

    Returns
    -------
    list[dict[str, Any]]
        A normalized list of step dictionaries.

    Raises
    ------
    ValueError
        - If a step has an unknown ``kind``.
        - If a thought step contains action/tool fields.

    Notes
    -----
    Legacy compatibility: if ``kind`` is missing, the step is treated as a
    legacy action step and converted to an ``ActionStep``.
    """
    out: List[Dict[str, Any]] = []
    for raw in plan:
        kind = raw.get("kind")
        if kind is None:
            out.append(
                ActionStep(
                    kind="action",
                    capability=CapabilityName(raw["capability"]),
                    args=dict(raw.get("args") or {}),
                    logical_tool=raw.get("logical_tool"),
                    server_id=raw.get("server_id"),
                    tool_name=raw.get("tool_name"),
                ).model_dump()
            )
            continue
        if kind == "thought":
            forbidden = {"capability", "logical_tool", "server_id", "tool_name"}
            present = forbidden.intersection(raw.keys())
            if present:
                raise ValueError(f"thought step cannot contain action fields: {sorted(present)}")
            out.append(ThoughtStep(**raw).model_dump())
            continue
        if kind == "action":
            out.append(ActionStep(**raw).model_dump())
            continue
        raise ValueError(f"unknown step kind: {kind}")
    return out
