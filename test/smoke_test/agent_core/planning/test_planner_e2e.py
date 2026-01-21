"""
End-to-end tests for AI agent planning features with real LLM calls.

These tests verify the core planning logic using real AI model providers
while mocking all other dependencies (database, cache, etc.).
"""

from __future__ import annotations

import asyncio
import os
from test.settings import test_settings
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from gearmeshing_ai.agent_core.model_provider import async_create_model_for_role
from gearmeshing_ai.agent_core.planning.planner import StructuredPlanner


class TestStructuredPlannerE2E:
    """End-to-end tests for StructuredPlanner with real AI models."""

    @pytest.fixture
    def mock_repos(self) -> Dict[str, AsyncMock]:
        """Mock all repository dependencies."""
        return {
            "runs": AsyncMock(),
            "events": AsyncMock(),
            "approvals": AsyncMock(),
            "checkpoints": AsyncMock(),
            "tool_invocations": AsyncMock(),
            "usage": AsyncMock(),
        }

    @pytest.fixture
    def mock_capabilities(self) -> MagicMock:
        """Mock capabilities registry."""
        capabilities = MagicMock()
        capabilities.list_all.return_value = [
            {
                "name": "read_file",
                "description": "Read a file from the filesystem",
                "parameters": {"file_path": "string"},
            },
            {
                "name": "write_file",
                "description": "Write content to a file",
                "parameters": {"file_path": "string", "content": "string"},
            },
            {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {"query": "string", "max_results": "integer"},
            },
        ]
        return capabilities

    @pytest.mark.asyncio
    async def test_planner_with_openai_real_model(
        self,
        mock_repos: Dict[str, AsyncMock],
        mock_capabilities: MagicMock,
        mock_database_access: Any,
        mock_settings_for_ai: Any,
    ) -> None:
        """Test planner with real OpenAI model for planning."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        # Debug: Check environment variables
        print(f"Debug - OPENAI_API_KEY in env: {bool(os.getenv('OPENAI_API_KEY'))}")
        print(f"Debug - AI_PROVIDER__OPENAI__API_KEY in env: {bool(os.getenv('AI_PROVIDER__OPENAI__API_KEY'))}")

        # Create real OpenAI model
        model = await async_create_model_for_role("planner")

        planner = StructuredPlanner(model=model, role="planner")

        # Test planning with a concrete objective
        objective = "Create a Python script that reads a CSV file and analyzes the data"
        role = "data_analyst"

        plan = await planner.plan(objective=objective, role=role)

        # Verify plan structure
        assert isinstance(plan, list)
        assert len(plan) > 0

        # Each step should be a dictionary with required fields
        for step in plan:
            assert isinstance(step, dict)
            assert "kind" in step
            assert step["kind"] in ["thought", "action"]

            if step["kind"] == "thought":
                assert "thought" in step
                assert "args" in step
            elif step["kind"] == "action":
                assert "capability" in step
                assert "args" in step

    @pytest.mark.asyncio
    async def test_planner_with_anthropic_real_model(self, mock_repos, mock_capabilities):
        """Test planner with real Anthropic model for planning."""
        if not test_settings.ai_provider.anthropic.api_key:
            pytest.skip("Anthropic API key not configured")

        # Create real Anthropic model
        model = await async_create_model_for_role("planner")

        planner = StructuredPlanner(model=model, role="planner")

        # Test planning with a complex objective
        objective = "Design a REST API for user management with authentication"
        role = "api_architect"

        plan = await planner.plan(objective=objective, role=role)

        # Verify plan contains logical steps
        assert isinstance(plan, list)
        assert len(plan) > 0

        # Should contain both thought and action steps
        has_thought = any(step.get("kind") == "thought" for step in plan)
        has_action = any(step.get("kind") == "action" for step in plan)

        # At minimum should have thought steps
        assert has_thought

    @pytest.mark.asyncio
    async def test_planner_with_google_real_model(
        self, mock_repos: Dict[str, AsyncMock], mock_capabilities: MagicMock
    ) -> None:
        """Test planner with real Google model for planning."""
        if not test_settings.ai_provider.google.api_key:
            pytest.skip("Google API key not configured")

        # Create real Google model
        model = await async_create_model_for_role("planner")

        planner = StructuredPlanner(model=model, role="planner")

        # Test planning with a technical objective
        objective = "Implement a machine learning pipeline for data preprocessing"
        role = "ml_engineer"

        plan = await planner.plan(objective=objective, role=role)

        # Verify plan structure and content
        assert isinstance(plan, list)
        assert len(plan) > 0

        # Validate step structure
        for step in plan:
            assert isinstance(step, dict)
            assert "type" in step
            if step["type"] == "action":
                assert "capability" in step
                assert step["capability"] in ["read_file", "write_file", "web_search"]

    @pytest.mark.asyncio
    async def test_planner_fallback_without_model(self, mock_repos, mock_capabilities):
        """Test planner fallback behavior when no model is available."""
        planner = StructuredPlanner(model=None, role="planner")

        objective = "Simple test objective"
        role = "test_role"

        plan = await planner.plan(objective=objective, role=role)

        # Should return a single thought step for summarization
        assert isinstance(plan, list)
        assert len(plan) == 1

        step = plan[0]
        assert step["kind"] == "thought"
        assert step["thought"] == "summarize"
        assert "text" in step["args"]
        assert "role" in step["args"]

    @pytest.mark.asyncio
    async def test_planner_with_different_roles(
        self, mock_repos: Dict[str, AsyncMock], mock_capabilities: MagicMock
    ) -> None:
        """Test planner behavior with different roles."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        model = await async_create_model_for_role("planner")
        planner = StructuredPlanner(model=model)

        objective = "Build a web application"

        # Test different roles
        roles = ["frontend_developer", "backend_developer", "fullstack_developer", "devops_engineer"]

        for role in roles:
            plan = await planner.plan(objective=objective, role=role)

            assert isinstance(plan, list)
            assert len(plan) > 0

            # Plans should be role-appropriate (structure may differ)
            for step in plan:
                assert isinstance(step, dict)
                assert "kind" in step

    @pytest.mark.asyncio
    async def test_planner_with_complex_objectives(
        self, mock_repos: Dict[str, AsyncMock], mock_capabilities: MagicMock
    ) -> None:
        """Test planner with complex, multi-step objectives."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        model = await async_create_model_for_role("planner")
        planner = StructuredPlanner(model=model, role="planner")

        complex_objectives = [
            "Create a complete e-commerce platform with user authentication, product catalog, shopping cart, and payment integration",
            "Design and implement a microservices architecture for a social media application with real-time messaging",
            "Build a data analytics dashboard that connects to multiple databases, processes real-time streams, and provides interactive visualizations",
        ]

        for objective in complex_objectives:
            plan = await planner.plan(objective=objective, role="system_architect")

            # Complex objectives should generate more detailed plans
            assert isinstance(plan, list)
            assert len(plan) > 0

            # Should have structured steps
            action_steps = [step for step in plan if step.get("kind") == "action"]
            thought_steps = [step for step in plan if step.get("kind") == "thought"]

            # Should have at least some thought steps for planning
            assert len(thought_steps) > 0

    @pytest.mark.asyncio
    async def test_planner_error_handling(self, mock_repos: Dict[str, AsyncMock], mock_capabilities: MagicMock) -> None:
        """Test planner error handling with real models."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        # Test with invalid model that will fail
        planner = StructuredPlanner(model="invalid_model", role="planner")

        objective = "Test objective"
        role = "test_role"

        # Should handle gracefully and fall back to thought step
        plan = await planner.plan(objective=objective, role=role)

        assert isinstance(plan, list)
        assert len(plan) >= 1

    @pytest.mark.asyncio
    async def test_planner_concurrent_requests(
        self, mock_repos: Dict[str, AsyncMock], mock_capabilities: MagicMock
    ) -> None:
        """Test planner handling concurrent planning requests."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        model = await async_create_model_for_role("planner")
        planner = StructuredPlanner(model=model, role="planner")

        objectives = [
            ("Build a REST API", "api_developer"),
            ("Create a database schema", "database_designer"),
            ("Write unit tests", "qa_engineer"),
            ("Deploy to production", "devops_engineer"),
        ]

        # Run planning concurrently
        tasks = [planner.plan(objective=obj, role=role) for obj, role in objectives]

        plans = await asyncio.gather(*tasks, return_exceptions=True)

        # All plans should be successful
        for i, plan in enumerate(plans):
            assert not isinstance(plan, Exception), f"Plan {i} failed: {plan}"
            assert isinstance(plan, list)
            assert len(plan) > 0

    @pytest.mark.asyncio
    async def test_planner_model_creation_from_role(
        self, mock_repos: Dict[str, AsyncMock], mock_capabilities: MagicMock
    ) -> None:
        """Test planner creating model from role configuration."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        # Create planner without model, let it create from role
        planner = StructuredPlanner(model=None, role="planner")

        objective = "Test objective for role-based model creation"
        role = "planner"

        plan = await planner.plan(objective=objective, role=role)

        assert isinstance(plan, list)
        assert len(plan) > 0

        # Verify model was created and cached
        assert planner._model is not None

    @pytest.mark.asyncio
    async def test_planner_with_tenant_isolation(
        self, mock_repos: Dict[str, AsyncMock], mock_capabilities: MagicMock
    ) -> None:
        """Test planner with tenant-specific configuration."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        # Test with different tenant IDs
        tenant_ids = ["tenant1", "tenant2", "tenant3"]

        for tenant_id in tenant_ids:
            planner = StructuredPlanner(model=None, role="planner", tenant_id=tenant_id)

            objective = f"Test objective for {tenant_id}"
            role = "planner"

            plan = await planner.plan(objective=objective, role=role)

            assert isinstance(plan, list)
            assert len(plan) > 0
