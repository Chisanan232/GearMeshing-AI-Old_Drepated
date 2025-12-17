from __future__ import annotations

from typing import Any, Dict, List, Literal, Union

from pydantic import Field

from ..schemas.base import BaseSchema
from ..schemas.domain import CapabilityName


class ThoughtStep(BaseSchema):
    kind: Literal["thought"] = "thought"
    thought: str
    args: Dict[str, Any] = Field(default_factory=dict)


class ActionStep(BaseSchema):
    kind: Literal["action"] = "action"
    capability: CapabilityName
    args: Dict[str, Any] = Field(default_factory=dict)
    server_id: str | None = None
    tool_name: str | None = None


PlanStep = Union[ThoughtStep, ActionStep]


def normalize_plan(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in plan:
        kind = raw.get("kind")
        if kind is None:
            out.append(
                ActionStep(
                    kind="action",
                    capability=CapabilityName(raw["capability"]),
                    args=dict(raw.get("args") or {}),
                    server_id=raw.get("server_id"),
                    tool_name=raw.get("tool_name"),
                ).model_dump()
            )
            continue
        if kind == "thought":
            forbidden = {"capability", "server_id", "tool_name"}
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
