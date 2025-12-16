from __future__ import annotations

from typing import Any, Dict, List

from pydantic import Field
from pydantic_ai import Agent

from ..schemas.base import BaseSchema
from ..schemas.domain import CapabilityName


class PlanStep(BaseSchema):
    capability: CapabilityName
    args: Dict[str, Any] = Field(default_factory=dict)


class StructuredPlanner:
    def __init__(self, *, model: Any | None = None) -> None:
        self._model = model

    async def plan(self, *, objective: str, role: str) -> List[Dict[str, Any]]:
        if self._model is None:
            return [
                {
                    "capability": CapabilityName.summarize,
                    "args": {"text": objective, "role": role},
                }
            ]

        agent: Agent = Agent(
            self._model,
            output_type=List[PlanStep],
            system_prompt=(
                "You are an expert planner for an autonomous software engineering agent. "
                "Return a minimal, safe sequence of capability steps as JSON."
            ),
        )

        result = await agent.run(
            (
                "Create a short plan for this objective. "
                "Use only the supported capabilities.\n\n"
                f"role={role}\n"
                f"objective={objective}\n"
            )
        )
        steps = result.output
        return [s.model_dump() for s in steps]
