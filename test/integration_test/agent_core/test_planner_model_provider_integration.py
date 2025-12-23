"""Integration tests for StructuredPlanner with model provider."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gearmeshing_ai.agent_core.planning.planner import StructuredPlanner
from gearmeshing_ai.agent_core.planning.steps import ActionStep, ThoughtStep


class TestPlannerModelProviderIntegration:
    """Integration tests for StructuredPlanner with model provider."""

    def test_planner_initialization_without_model(self):
        """Test planner initializes in deterministic mode without model."""
        planner = StructuredPlanner(model=None)
        assert planner._model is None
        assert planner._role is None
        assert planner._tenant_id is None

    def test_planner_initialization_with_model(self):
        """Test planner initializes with provided model."""
        mock_model = MagicMock()
        planner = StructuredPlanner(model=mock_model)
        assert planner._model is mock_model

    def test_planner_initialization_with_role_no_model(self):
        """Test planner initialization with role but no model."""
        # When role is provided but model creation fails, should fall back to None
        with patch("gearmeshing_ai.agent_core.model_provider.create_model_for_role") as mock_create:
            mock_create.side_effect = ValueError("Role not found")
            
            planner = StructuredPlanner(role="dev", tenant_id="acme-corp")
            
            # Should fall back to None model
            assert planner._model is None
            assert planner._role == "dev"
            assert planner._tenant_id == "acme-corp"

    @pytest.mark.asyncio
    async def test_planner_plan_deterministic_mode(self):
        """Test planner in deterministic mode (no model)."""
        planner = StructuredPlanner(model=None)
        
        plan = await planner.plan(objective="Test objective", role="dev")
        
        # Should return a single thought step
        assert len(plan) == 1
        assert plan[0]["kind"] == "thought"
        assert plan[0]["thought"] == "summarize"
        assert "text" in plan[0]["args"]
        assert "role" in plan[0]["args"]

    @pytest.mark.asyncio
    async def test_planner_plan_with_model(self):
        """Test planner with model generates plan."""
        mock_model = MagicMock()
        planner = StructuredPlanner(model=mock_model)
        
        # Mock the agent result with a valid capability
        mock_action_step = ActionStep(
            kind="action",
            capability="web_search",
            args={"query": "test query"},
        )
        
        with patch("gearmeshing_ai.agent_core.planning.planner.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent
            
            mock_result = MagicMock()
            mock_result.output = [mock_action_step]
            mock_agent.run = AsyncMock(return_value=mock_result)
            
            plan = await planner.plan(objective="Test objective", role="dev")
            
            # Should return the action steps as dictionaries
            assert len(plan) == 1
            assert plan[0]["kind"] == "action"
            assert plan[0]["capability"] == "web_search"

    def test_planner_stores_role_and_tenant(self):
        """Test planner stores role and tenant_id for later use."""
        planner = StructuredPlanner(role="planner", tenant_id="test-tenant")
        
        assert planner._role == "planner"
        assert planner._tenant_id == "test-tenant"

    @pytest.mark.asyncio
    async def test_planner_deterministic_includes_objective_and_role(self):
        """Test deterministic plan includes objective and role in args."""
        planner = StructuredPlanner(model=None)
        
        objective = "Implement feature X"
        role = "developer"
        
        plan = await planner.plan(objective=objective, role=role)
        
        assert len(plan) == 1
        assert plan[0]["args"]["text"] == objective
        assert plan[0]["args"]["role"] == role

    def test_planner_initialization_with_all_parameters(self):
        """Test planner initialization with all parameters."""
        mock_model = MagicMock()
        planner = StructuredPlanner(
            model=mock_model,
            role="qa",
            tenant_id="acme-corp",
        )
        
        assert planner._model is mock_model
        assert planner._role == "qa"
        assert planner._tenant_id == "acme-corp"

    @pytest.mark.asyncio
    async def test_planner_agent_system_prompt(self):
        """Test planner creates agent with correct system prompt."""
        mock_model = MagicMock()
        planner = StructuredPlanner(model=mock_model)
        
        with patch("gearmeshing_ai.agent_core.planning.planner.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent
            
            mock_result = MagicMock()
            mock_result.output = []
            mock_agent.run = AsyncMock(return_value=mock_result)
            
            await planner.plan(objective="Test", role="dev")
            
            # Verify Agent was created with correct parameters
            mock_agent_class.assert_called_once()
            call_kwargs = mock_agent_class.call_args[1]
            assert "system_prompt" in call_kwargs
            assert "planner" in call_kwargs["system_prompt"].lower()

    @pytest.mark.asyncio
    async def test_planner_agent_output_type(self):
        """Test planner creates agent with correct output type."""
        mock_model = MagicMock()
        planner = StructuredPlanner(model=mock_model)
        
        with patch("gearmeshing_ai.agent_core.planning.planner.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent
            
            mock_result = MagicMock()
            mock_result.output = []
            mock_agent.run = AsyncMock(return_value=mock_result)
            
            await planner.plan(objective="Test", role="dev")
            
            # Verify output_type is set
            call_kwargs = mock_agent_class.call_args[1]
            assert "output_type" in call_kwargs

    @pytest.mark.asyncio
    async def test_planner_passes_objective_to_agent(self):
        """Test planner passes objective to agent.run."""
        mock_model = MagicMock()
        planner = StructuredPlanner(model=mock_model)
        
        with patch("gearmeshing_ai.agent_core.planning.planner.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent
            
            mock_result = MagicMock()
            mock_result.output = []
            mock_agent.run = AsyncMock(return_value=mock_result)
            
            objective = "Implement new feature"
            role = "dev"
            
            await planner.plan(objective=objective, role=role)
            
            # Verify agent.run was called with objective and role
            mock_agent.run.assert_called_once()
            call_args = mock_agent.run.call_args[0][0]
            assert objective in call_args
            assert role in call_args

    def test_planner_model_creation_failure_graceful_fallback(self):
        """Test planner gracefully falls back when model creation fails."""
        with patch("gearmeshing_ai.agent_core.model_provider.create_model_for_role") as mock_create:
            # Simulate various failure modes
            mock_create.side_effect = RuntimeError("API key not set")
            
            # Should not raise, should fall back to None
            planner = StructuredPlanner(role="dev")
            assert planner._model is None

    @pytest.mark.asyncio
    async def test_planner_returns_serializable_plan(self):
        """Test planner returns JSON-serializable plan."""
        planner = StructuredPlanner(model=None)
        
        plan = await planner.plan(objective="Test", role="dev")
        
        # Should be serializable to JSON
        import json
        json_str = json.dumps(plan)
        assert json_str is not None
        
        # Should be able to deserialize back
        deserialized = json.loads(json_str)
        assert deserialized == plan
