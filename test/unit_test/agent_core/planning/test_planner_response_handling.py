"""Tests for StructuredPlanner response handling edge cases."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from gearmeshing_ai.agent_core.planning import StructuredPlanner
from gearmeshing_ai.agent_core.planning.steps import ThoughtStep


@pytest.mark.asyncio
async def test_planner_handles_non_list_response_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test planner returns thought step when agent response content is not a list."""
    from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

    class _FakeAgent:
        def __init__(self, config: Any) -> None:
            self.config = config
            self._initialized = False

        async def initialize(self) -> None:
            self._initialized = True

        async def invoke(self, input_text: str, **kwargs: Any) -> AIAgentResponse:
            # Return non-list content (e.g., dict or string)
            return AIAgentResponse(content={"error": "invalid response"}, success=True)

        async def cleanup(self) -> None:
            pass

    class _FakeProvider:
        async def create_agent(self, config: Any, use_cache: bool = False) -> _FakeAgent:
            agent = _FakeAgent(config)
            await agent.initialize()
            return agent

    import gearmeshing_ai.agent_core.planning.planner as planner_mod

    monkeypatch.setattr(planner_mod, "get_agent_provider", lambda: _FakeProvider())

    mock_model = MagicMock()
    mock_model.model_name = "test-model"
    planner = StructuredPlanner(model=mock_model)

    objective = "Test objective"
    role = "dev"

    plan = await planner.plan(objective=objective, role=role)

    # Should return a thought step when response.content is not a list
    assert len(plan) == 1
    assert plan[0]["kind"] == "thought"
    assert plan[0]["thought"] == "summarize"
    assert plan[0]["args"]["text"] == objective
    assert plan[0]["args"]["role"] == role


@pytest.mark.asyncio
async def test_planner_handles_dict_response_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test planner returns thought step when agent response content is a dict."""
    from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

    class _FakeAgent:
        def __init__(self, config: Any) -> None:
            self.config = config
            self._initialized = False

        async def initialize(self) -> None:
            self._initialized = True

        async def invoke(self, input_text: str, **kwargs: Any) -> AIAgentResponse:
            # Return dict content instead of list
            return AIAgentResponse(
                content={"result": "some result", "status": "ok"},
                success=True,
            )

        async def cleanup(self) -> None:
            pass

    class _FakeProvider:
        async def create_agent(self, config: Any, use_cache: bool = False) -> _FakeAgent:
            agent = _FakeAgent(config)
            await agent.initialize()
            return agent

    import gearmeshing_ai.agent_core.planning.planner as planner_mod

    monkeypatch.setattr(planner_mod, "get_agent_provider", lambda: _FakeProvider())

    mock_model = MagicMock()
    mock_model.model_name = "test-model"
    planner = StructuredPlanner(model=mock_model)

    plan = await planner.plan(objective="Find data", role="analyst")

    # Should return a thought step
    assert len(plan) == 1
    assert plan[0]["kind"] == "thought"
    assert plan[0]["thought"] == "summarize"


@pytest.mark.asyncio
async def test_planner_handles_string_response_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test planner returns thought step when agent response content is a string."""
    from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

    class _FakeAgent:
        def __init__(self, config: Any) -> None:
            self.config = config
            self._initialized = False

        async def initialize(self) -> None:
            self._initialized = True

        async def invoke(self, input_text: str, **kwargs: Any) -> AIAgentResponse:
            # Return string content instead of list
            return AIAgentResponse(content="This is a string response", success=True)

        async def cleanup(self) -> None:
            pass

    class _FakeProvider:
        async def create_agent(self, config: Any, use_cache: bool = False) -> _FakeAgent:
            agent = _FakeAgent(config)
            await agent.initialize()
            return agent

    import gearmeshing_ai.agent_core.planning.planner as planner_mod

    monkeypatch.setattr(planner_mod, "get_agent_provider", lambda: _FakeProvider())

    mock_model = MagicMock()
    mock_model.model_name = "test-model"
    planner = StructuredPlanner(model=mock_model)

    plan = await planner.plan(objective="Process text", role="processor")

    # Should return a thought step
    assert len(plan) == 1
    assert plan[0]["kind"] == "thought"
    assert plan[0]["thought"] == "summarize"


@pytest.mark.asyncio
async def test_planner_non_list_response_includes_objective_and_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that non-list response thought step includes objective and role."""
    from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

    class _FakeAgent:
        def __init__(self, config: Any) -> None:
            self.config = config
            self._initialized = False

        async def initialize(self) -> None:
            self._initialized = True

        async def invoke(self, input_text: str, **kwargs: Any) -> AIAgentResponse:
            return AIAgentResponse(content=None, success=True)

        async def cleanup(self) -> None:
            pass

    class _FakeProvider:
        async def create_agent(self, config: Any, use_cache: bool = False) -> _FakeAgent:
            agent = _FakeAgent(config)
            await agent.initialize()
            return agent

    import gearmeshing_ai.agent_core.planning.planner as planner_mod

    monkeypatch.setattr(planner_mod, "get_agent_provider", lambda: _FakeProvider())

    mock_model = MagicMock()
    mock_model.model_name = "test-model"
    planner = StructuredPlanner(model=mock_model)

    objective = "Implement feature X"
    role = "developer"

    plan = await planner.plan(objective=objective, role=role)

    # Verify the thought step has correct args
    assert plan[0]["args"]["text"] == objective
    assert plan[0]["args"]["role"] == role


@pytest.mark.asyncio
async def test_planner_non_list_response_is_serializable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that non-list response plan is JSON serializable."""
    from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

    class _FakeAgent:
        def __init__(self, config: Any) -> None:
            self.config = config
            self._initialized = False

        async def initialize(self) -> None:
            self._initialized = True

        async def invoke(self, input_text: str, **kwargs: Any) -> AIAgentResponse:
            return AIAgentResponse(content={"unexpected": "format"}, success=True)

        async def cleanup(self) -> None:
            pass

    class _FakeProvider:
        async def create_agent(self, config: Any, use_cache: bool = False) -> _FakeAgent:
            agent = _FakeAgent(config)
            await agent.initialize()
            return agent

    import gearmeshing_ai.agent_core.planning.planner as planner_mod

    monkeypatch.setattr(planner_mod, "get_agent_provider", lambda: _FakeProvider())

    mock_model = MagicMock()
    mock_model.model_name = "test-model"
    planner = StructuredPlanner(model=mock_model)

    plan = await planner.plan(objective="Test", role="dev")

    # Should be JSON serializable
    import json

    json_str = json.dumps(plan)
    assert json_str is not None

    deserialized = json.loads(json_str)
    assert deserialized == plan
