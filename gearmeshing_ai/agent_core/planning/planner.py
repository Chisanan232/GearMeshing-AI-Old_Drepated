from __future__ import annotations

from typing import Any, Dict, List

from pydantic_ai import Agent

from .steps import ActionStep, ThoughtStep


class StructuredPlanner:
    def __init__(self, *, model: Any | None = None) -> None:
        self._model = model

    async def plan(self, *, objective: str, role: str) -> List[Dict[str, Any]]:
        if self._model is None:
            return [ThoughtStep(thought="summarize", args={"text": objective, "role": role}).model_dump()]

        agent: Agent = Agent(
            self._model,
            output_type=List[ActionStep],
            system_prompt=(
                "You are an expert planner for an autonomous software engineering agent. "
                "Return a minimal, safe sequence of action steps as JSON."
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
