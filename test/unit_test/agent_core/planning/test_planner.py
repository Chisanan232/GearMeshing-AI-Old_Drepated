from __future__ import annotations

from typing import Any

import pytest

from gearmeshing_ai.agent_core.planning import (
    ActionStep,
    StructuredPlanner,
)
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

    from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

    class _FakeAgent:
        def __init__(self, config: Any) -> None:
            self.config = config
            self._initialized = False

        async def initialize(self) -> None:
            self._initialized = True

        async def invoke(self, input_text: str, **kwargs: Any) -> AIAgentResponse:
            return AIAgentResponse(
                content=[
                    ActionStep(capability=CapabilityName.web_search, args={"query": "hello"}),
                    ActionStep(capability=CapabilityName.summarize, args={"text": "done"}),
                ],
                success=True,
            )

        async def cleanup(self) -> None:
            pass

    class _FakeProvider:
        async def create_agent(self, config: Any, use_cache: bool = False) -> _FakeAgent:
            agent = _FakeAgent(config)
            await agent.initialize()
            return agent
        
        async def create_agent_from_config_source(self, config_source: Any, use_cache: bool = False) -> _FakeAgent:
            # Mock the config source to return an AIAgentConfig object
            from gearmeshing_ai.agent_core.abstraction import AIAgentConfig
            mock_config = AIAgentConfig(
                name="test-planner",
                framework="pydantic_ai",
                model="gpt-4o",
                system_prompt="You are an expert planner...",
                temperature=0.7,
                max_tokens=4096,
                top_p=0.9,
                metadata={"output_type": list},
            )
            agent = _FakeAgent(mock_config)
            await agent.initialize()
            return agent

    import gearmeshing_ai.agent_core.planning.planner as planner_mod

    monkeypatch.setattr(planner_mod, "get_agent_provider", lambda: _FakeProvider())

    planner = StructuredPlanner(model=object())
    out = await planner.plan(objective="Find info", role="market")

    assert out == [
        {
            "kind": "action",
            "capability": CapabilityName.web_search,
            "args": {"query": "hello"},
            "logical_tool": None,
            "server_id": None,
            "tool_name": None,
        },
        {
            "kind": "action",
            "capability": CapabilityName.summarize,
            "args": {"text": "done"},
            "logical_tool": None,
            "server_id": None,
            "tool_name": None,
        },
    ]


@pytest.mark.asyncio
async def test_planner_with_testmodel_produces_valid_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

    class _FakeAgent:
        def __init__(self, config: Any) -> None:
            self.config = config
            self._initialized = False

        async def initialize(self) -> None:
            self._initialized = True

        async def invoke(self, input_text: str, **kwargs: Any) -> AIAgentResponse:
            return AIAgentResponse(
                content=[
                    ActionStep(capability=CapabilityName.web_search, args={"query": "test"}),
                ],
                success=True,
            )

        async def cleanup(self) -> None:
            pass

    class _FakeProvider:
        async def create_agent(self, config: Any, use_cache: bool = False) -> _FakeAgent:
            agent = _FakeAgent(config)
            await agent.initialize()
            return agent
        
        async def create_agent_from_config_source(self, config_source: Any, use_cache: bool = False) -> _FakeAgent:
            # Mock the config source to return an AIAgentConfig object
            from gearmeshing_ai.agent_core.abstraction import AIAgentConfig
            mock_config = AIAgentConfig(
                name="test-planner",
                framework="pydantic_ai",
                model="gpt-4o",
                system_prompt="You are an expert planner...",
                temperature=0.7,
                max_tokens=4096,
                top_p=0.9,
                metadata={"output_type": list},
            )
            agent = _FakeAgent(mock_config)
            await agent.initialize()
            return agent

    import gearmeshing_ai.agent_core.planning.planner as planner_mod

    monkeypatch.setattr(planner_mod, "get_agent_provider", lambda: _FakeProvider())

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
async def test_planner_with_functionmodel_returns_expected_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pydantic_ai")

    from pydantic_ai import ModelMessage, ModelResponse, ToolCallPart, models
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

    models.ALLOW_MODEL_REQUESTS = False

    class _FakeAgent:
        def __init__(self, config: Any) -> None:
            self.config = config
            self._initialized = False

        async def initialize(self) -> None:
            self._initialized = True

        async def invoke(self, input_text: str, **kwargs: Any) -> AIAgentResponse:
            return AIAgentResponse(
                content=[
                    ActionStep(capability=CapabilityName.web_search, args={"query": "pydantic ai testing"}),
                    ActionStep(capability=CapabilityName.summarize, args={"text": "summary me"}),
                ],
                success=True,
            )

        async def cleanup(self) -> None:
            pass

    class _FakeProvider:
        async def create_agent(self, config: Any, use_cache: bool = False) -> _FakeAgent:
            agent = _FakeAgent(config)
            await agent.initialize()
            return agent
        
        async def create_agent_from_config_source(self, config_source: Any, use_cache: bool = False) -> _FakeAgent:
            # Mock the config source to return an AIAgentConfig object
            from gearmeshing_ai.agent_core.abstraction import AIAgentConfig
            mock_config = AIAgentConfig(
                name="test-planner",
                framework="pydantic_ai",
                model="gpt-4o",
                system_prompt="You are an expert planner...",
                temperature=0.7,
                max_tokens=4096,
                top_p=0.9,
                metadata={"output_type": list},
            )
            agent = _FakeAgent(mock_config)
            await agent.initialize()
            return agent

    import gearmeshing_ai.agent_core.planning.planner as planner_mod

    monkeypatch.setattr(planner_mod, "get_agent_provider", lambda: _FakeProvider())

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
        {
            "kind": "action",
            "capability": CapabilityName.web_search,
            "args": {"query": "pydantic ai testing"},
            "logical_tool": None,
            "server_id": None,
            "tool_name": None,
        },
        {
            "kind": "action",
            "capability": CapabilityName.summarize,
            "args": {"text": "summary me"},
            "logical_tool": None,
            "server_id": None,
            "tool_name": None,
        },
    ]
