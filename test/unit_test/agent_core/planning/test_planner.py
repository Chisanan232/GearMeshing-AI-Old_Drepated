from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import pytest

from gearmeshing_ai.agent_core.planning import ActionStep, StructuredPlanner, ThoughtStep
from gearmeshing_ai.agent_core.schemas.domain import CapabilityName


@pytest.mark.asyncio
async def test_planner_fallback_when_model_is_none() -> None:
    planner = StructuredPlanner(model=None)
    steps = await planner.plan(objective="do x", role="dev")

    assert steps == [
        {
            "kind": "thought",
            "thought": "summarize",
            "args": {"text": "do x", "role": "dev"},
        }
    ]


@pytest.mark.asyncio
async def test_planner_uses_pydantic_ai_agent_and_model_dump(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pydantic_ai")

    @dataclass
    class _FakeResult:
        output: List[ActionStep]

    class _FakeAgent:
        def __init__(self, model: Any, *, output_type: Any, system_prompt: str):
            self.model = model
            self.output_type = output_type
            self.system_prompt = system_prompt
            self.last_prompt: str | None = None

        async def run(self, prompt: str) -> _FakeResult:
            self.last_prompt = prompt
            return _FakeResult(
                output=[
                    ActionStep(capability=CapabilityName.web_search, args={"query": "hello"}),
                    ActionStep(capability=CapabilityName.summarize, args={"text": "done"}),
                ]
            )

    import gearmeshing_ai.agent_core.planning.planner as planner_mod

    monkeypatch.setattr(planner_mod, "Agent", _FakeAgent)

    planner = StructuredPlanner(model=object())
    out = await planner.plan(objective="Find info", role="market")

    assert out == [
        {"kind": "action", "capability": CapabilityName.web_search, "args": {"query": "hello"}, "server_id": None, "tool_name": None},
        {"kind": "action", "capability": CapabilityName.summarize, "args": {"text": "done"}, "server_id": None, "tool_name": None},
    ]


@pytest.mark.asyncio
async def test_planner_with_testmodel_produces_valid_steps() -> None:
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    planner = StructuredPlanner(model=TestModel())
    out = await planner.plan(objective="Research pricing", role="market")

    assert isinstance(out, list)
    assert len(out) >= 1

    step0 = out[0]
    assert isinstance(step0, dict)
    assert "kind" in step0
    assert "args" in step0
    assert step0["kind"] in {"thought", "action"}
    if step0["kind"] == "action":
        assert isinstance(step0["capability"], CapabilityName)
    assert isinstance(step0["args"], dict)


@pytest.mark.asyncio
async def test_planner_with_functionmodel_returns_expected_plan() -> None:
    pytest.importorskip("pydantic_ai")

    from pydantic_ai import ModelMessage, ModelResponse, ToolCallPart, models
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    models.ALLOW_MODEL_REQUESTS = False

    def call_planner(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "final_result",
                    {
                        "response": [
                            {"kind": "action", "capability": "web_search", "args": {"query": "pydantic ai testing"}},
                            {"kind": "action", "capability": "summarize", "args": {"text": "summary me"}},
                        ]
                    },
                )
            ]
        )

    planner = StructuredPlanner(model=FunctionModel(call_planner))
    out = await planner.plan(objective="Write tests", role="dev")

    assert out == [
        {"kind": "action", "capability": CapabilityName.web_search, "args": {"query": "pydantic ai testing"}, "server_id": None, "tool_name": None},
        {"kind": "action", "capability": CapabilityName.summarize, "args": {"text": "summary me"}, "server_id": None, "tool_name": None},
    ]
