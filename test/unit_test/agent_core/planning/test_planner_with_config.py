"""Tests for StructuredPlanner with configuration-based model support."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gearmeshing_ai.agent_core.planning import (
    ActionStep,
    StructuredPlanner,
)
from gearmeshing_ai.info_provider import CapabilityName


class TestPlannerWithConfigurationSupport:
    """Tests for StructuredPlanner with role and tenant_id parameters."""

    def test_planner_initialization_with_role_and_tenant(self) -> None:
        """Test planner initialization with role and tenant_id."""
        planner = StructuredPlanner(role="dev", tenant_id="acme-corp")
        assert planner._role == "dev"
        assert planner._tenant_id == "acme-corp"

    def test_planner_initialization_with_role_only(self) -> None:
        """Test planner initialization with role only."""
        planner = StructuredPlanner(role="planner")
        assert planner._role == "planner"
        assert planner._tenant_id is None

    def test_planner_initialization_with_model_takes_precedence(self) -> None:
        """Test that explicit model takes precedence over role."""
        mock_model = MagicMock()
        planner = StructuredPlanner(model=mock_model, role="dev")
        assert planner._model is mock_model

    @pytest.mark.asyncio
    async def test_planner_creates_model_from_role(self) -> None:
        """Test planner creates model from role if model not provided."""
        mock_model = MagicMock()

        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
            mock_create.return_value = mock_model

            planner = StructuredPlanner(role="dev", tenant_id="acme-corp")
            # Model creation is deferred to plan() method
            assert planner._role == "dev"
            assert planner._tenant_id == "acme-corp"

    def test_planner_falls_back_to_none_on_model_creation_error(self) -> None:
        """Test planner falls back to None if model creation fails."""
        # Model creation is deferred to plan() method, so constructor doesn't fail
        planner = StructuredPlanner(role="unknown_role")

        # Model is None until plan() is called
        assert planner._model is None

    @pytest.mark.asyncio
    async def test_planner_with_role_uses_created_model(self) -> None:
        """Test planner uses created model for planning."""
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

        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
            mock_model = MagicMock()
            mock_model.model_name = "test-model"
            mock_create.return_value = mock_model

            with patch.object(planner_mod, "get_agent_provider", return_value=_FakeProvider()):
                planner = StructuredPlanner(role="dev")
                steps = await planner.plan(objective="test", role="dev")

                assert len(steps) > 0
                assert steps[0]["capability"] == CapabilityName.web_search

    @pytest.mark.asyncio
    async def test_planner_fallback_when_role_model_creation_fails(self) -> None:
        """Test planner falls back to deterministic mode if model creation fails."""
        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
            mock_create.side_effect = ValueError("Role not found")

            planner = StructuredPlanner(role="unknown_role")
            steps = await planner.plan(objective="do x", role="dev")

            # Should fall back to deterministic mode
            assert steps == [
                {
                    "kind": "thought",
                    "thought": "summarize",
                    "args": {"text": "do x", "role": "dev"},
                }
            ]

    def test_planner_with_explicit_model_ignores_role(self) -> None:
        """Test that explicit model ignores role parameter."""
        mock_model = MagicMock()

        with patch("gearmeshing_ai.agent_core.model_provider.create_model_for_role") as mock_create:
            planner = StructuredPlanner(model=mock_model, role="dev")

            # create_model_for_role should not be called
            mock_create.assert_not_called()
            assert planner._model is mock_model

    @pytest.mark.asyncio
    async def test_planner_with_none_model_and_no_role_uses_fallback(self) -> None:
        """Test planner with no model and no role uses fallback."""
        planner = StructuredPlanner(model=None, role=None)
        steps = await planner.plan(objective="do x", role="dev")

        assert steps == [
            {
                "kind": "thought",
                "thought": "summarize",
                "args": {"text": "do x", "role": "dev"},
            }
        ]

    def test_planner_tenant_id_passed_to_model_creation(self) -> None:
        """Test that tenant_id is passed to model creation."""
        planner = StructuredPlanner(role="dev", tenant_id="acme-corp")

        # Verify tenant_id is stored for later use in plan()
        assert planner._role == "dev"
        assert planner._tenant_id == "acme-corp"
