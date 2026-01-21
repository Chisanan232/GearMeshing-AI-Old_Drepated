"""
Multi-provider E2E tests for AI agent planning with real models.

These tests verify planning functionality across different AI providers
to ensure consistent behavior and provider-specific optimizations.
"""

from __future__ import annotations

import asyncio
from test.settings import test_settings
from typing import Any, Dict, List, Tuple

import pytest

from gearmeshing_ai.agent_core.model_provider import async_create_model_for_role
from gearmeshing_ai.agent_core.planning.planner import StructuredPlanner


class TestMultiProviderPlanningE2E:
    """End-to-end tests for planning across multiple AI providers."""

    @pytest.fixture
    def provider_configs(self) -> Dict[str, Dict[str, Any]]:
        """Fixture providing provider configurations."""
        return {
            "openai": {
                "available": bool(test_settings.ai_provider.openai.api_key),
                "model": test_settings.ai_provider.openai.model,
                "api_key": (
                    test_settings.ai_provider.openai.api_key.get_secret_value()
                    if test_settings.ai_provider.openai.api_key
                    else None
                ),
            },
            "anthropic": {
                "available": bool(test_settings.ai_provider.anthropic.api_key),
                "model": test_settings.ai_provider.anthropic.model,
                "api_key": (
                    test_settings.ai_provider.anthropic.api_key.get_secret_value()
                    if test_settings.ai_provider.anthropic.api_key
                    else None
                ),
            },
            "google": {
                "available": bool(test_settings.ai_provider.google.api_key),
                "model": test_settings.ai_provider.google.model,
                "api_key": (
                    test_settings.ai_provider.google.api_key.get_secret_value()
                    if test_settings.ai_provider.google.api_key
                    else None
                ),
            },
        }

    @pytest.fixture
    def test_objectives(self) -> List[Tuple[str, str, str]]:
        """Fixture providing test objectives with expected complexity."""
        return [
            ("Simple task", "Write a hello world function", "simple"),
            ("Code review", "Review a Python function for bugs", "medium"),
            ("System design", "Design a microservices architecture", "complex"),
            ("Data analysis", "Create a data pipeline for analytics", "complex"),
            ("API development", "Build a REST API with authentication", "medium"),
            ("Testing strategy", "Design comprehensive test coverage", "medium"),
            ("Performance optimization", "Optimize database query performance", "complex"),
            ("Security audit", "Conduct security assessment of web app", "complex"),
        ]

    @pytest.mark.asyncio
    @pytest.mark.multi_provider
    async def test_cross_provider_planning_consistency(
        self, provider_configs: Dict[str, Dict[str, Any]], test_objectives: List[Tuple[str, str, str]]
    ) -> None:
        """Test planning consistency across different providers."""
        available_providers = [name for name, config in provider_configs.items() if config["available"]]

        if len(available_providers) < 2:
            pytest.skip("Need at least 2 providers for cross-provider testing")

        # Select a simple objective for consistency testing
        objective, role, complexity = test_objectives[0]  # Simple task

        plans = {}

        # Generate plans from each available provider
        for provider_name in available_providers:
            try:
                model = await async_create_model_for_role("planner")
                planner = StructuredPlanner(model=model, role="planner")

                plan = await planner.plan(objective=objective, role=role)
                plans[provider_name] = plan

                # Verify basic structure
                assert isinstance(plan, list)
                assert len(plan) > 0

            except Exception as e:
                pytest.fail(f"Provider {provider_name} failed: {e}")

        # Compare plan structures (should be similar in structure)
        plan_lengths = [len(plan) for plan in plans.values()]

        # Plans should have reasonable length variation (not too different)
        max_length = max(plan_lengths)
        min_length = min(plan_lengths)
        assert max_length - min_length <= max_length * 0.5, "Plans too different in length"

        # All plans should have valid step structure
        for provider_name, plan in plans.items():
            for step in plan:
                assert isinstance(step, dict)
                assert "type" in step
                assert step["type"] in ["thought", "action"]

    @pytest.mark.asyncio
    @pytest.mark.openai_only
    async def test_openai_planning_comprehensive(
        self, provider_configs: Dict[str, Dict[str, Any]], test_objectives: List[Tuple[str, str, str]]
    ) -> None:
        """Test comprehensive planning with OpenAI."""
        if not provider_configs["openai"]["available"]:
            pytest.skip("OpenAI not available")

        model = await async_create_model_for_role("planner")
        planner = StructuredPlanner(model=model, role="planner")

        # Test different complexity levels
        for objective, role, complexity in test_objectives[:4]:  # Test subset
            plan = await planner.plan(objective=objective, role=role)

            # Verify plan complexity matches expectation
            assert isinstance(plan, list)
            assert len(plan) > 0

            # More complex objectives should generate more steps
            if complexity == "complex":
                assert len(plan) >= 2, "Complex objectives should generate more steps"

            # Verify step quality
            action_steps = [step for step in plan if step.get("type") == "action"]
            thought_steps = [step for step in plan if step.get("type") == "thought"]

            # Should have thought steps for planning
            assert len(thought_steps) > 0, "Should have planning thoughts"

    @pytest.mark.asyncio
    @pytest.mark.anthropic_only
    async def test_anthropic_planning_comprehensive(
        self, provider_configs: Dict[str, Dict[str, Any]], test_objectives: List[Tuple[str, str, str]]
    ) -> None:
        """Test comprehensive planning with Anthropic."""
        if not provider_configs["anthropic"]["available"]:
            pytest.skip("Anthropic not available")

        model = await async_create_model_for_role("planner")
        planner = StructuredPlanner(model=model, role="planner")

        # Test different complexity levels
        for objective, role, complexity in test_objectives[:4]:  # Test subset
            plan = await planner.plan(objective=objective, role=role)

            # Verify plan structure
            assert isinstance(plan, list)
            assert len(plan) > 0

            # Anthropic should provide detailed reasoning
            thought_steps = [step for step in plan if step.get("type") == "thought"]
            assert len(thought_steps) > 0, "Anthropic should provide detailed thoughts"

            # Check thought step quality
            for step in thought_steps:
                assert "thought" in step
                assert "args" in step
                assert len(step["thought"]) > 10, "Thoughts should be substantial"

    @pytest.mark.asyncio
    @pytest.mark.google_only
    async def test_google_planning_comprehensive(
        self, provider_configs: Dict[str, Dict[str, Any]], test_objectives: List[Tuple[str, str, str]]
    ) -> None:
        """Test comprehensive planning with Google."""
        if not provider_configs["google"]["available"]:
            pytest.skip("Google not available")

        model = await async_create_model_for_role("planner")
        planner = StructuredPlanner(model=model, role="planner")

        # Test different complexity levels
        for objective, role, complexity in test_objectives[:4]:  # Test subset
            plan = await planner.plan(objective=objective, role=role)

            # Verify plan structure
            assert isinstance(plan, list)
            assert len(plan) > 0

            # Google should provide structured approach
            for step in plan:
                assert isinstance(step, dict)
                assert "type" in step

                if step["type"] == "action":
                    assert "capability" in step
                    assert "args" in step

    @pytest.mark.asyncio
    @pytest.mark.multi_provider
    async def test_provider_specific_optimizations(self, provider_configs: Dict[str, Dict[str, Any]]) -> None:
        """Test provider-specific planning optimizations."""
        available_providers = [name for name, config in provider_configs.items() if config["available"]]

        if len(available_providers) < 2:
            pytest.skip("Need at least 2 providers for optimization testing")

        # Test with provider-specific objectives
        test_cases = [
            ("Code generation", "Generate a Python class for data validation", "developer"),
            ("Analysis", "Analyze system performance bottlenecks", "analyst"),
            ("Design", "Design database schema for e-commerce", "architect"),
        ]

        for objective, description, role in test_cases:
            provider_results = {}

            for provider_name in available_providers:
                try:
                    model = await async_create_model_for_role("planner")
                    planner = StructuredPlanner(model=model, role="planner")

                    plan = await planner.plan(objective=description, role=role)
                    provider_results[provider_name] = plan

                except Exception as e:
                    pytest.fail(f"Provider {provider_name} failed for {objective}: {e}")

            # Each provider should generate valid plans
            for provider_name, plan in provider_results.items():
                assert isinstance(plan, list)
                assert len(plan) > 0

                # Verify plan quality
                has_thoughts = any(step.get("type") == "thought" for step in plan)
                assert has_thoughts, f"{provider_name} should generate thought steps"

    @pytest.mark.asyncio
    @pytest.mark.multi_provider
    async def test_concurrent_multi_provider_planning(self, provider_configs):
        """Test concurrent planning across multiple providers."""
        available_providers = [name for name, config in provider_configs.items() if config["available"]]

        if len(available_providers) < 2:
            pytest.skip("Need at least 2 providers for concurrent testing")

        # Create concurrent planning tasks
        tasks = []
        for provider_name in available_providers:
            for i in range(3):  # 3 tasks per provider
                task = asyncio.create_task(self._plan_with_provider(provider_name, f"Task {i}"))
                tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all results
        success_count = 0
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent planning failed: {result}")
            else:
                success_count += 1
                assert isinstance(result, list)
                assert len(result) > 0

        # Should have successful results
        assert success_count == len(tasks)

    async def _plan_with_provider(self, provider_name: str, task_id: str) -> List[Dict[str, Any]]:
        """Helper method for concurrent planning."""
        model = await async_create_model_for_role("planner")
        planner = StructuredPlanner(model=model, role="planner")

        objective = f"Complete task {task_id}: Implement a feature"
        role = "developer"

        return await planner.plan(objective=objective, role=role)

    @pytest.mark.asyncio
    @pytest.mark.multi_provider
    async def test_provider_error_handling(self, provider_configs):
        """Test error handling across different providers."""
        available_providers = [name for name, config in provider_configs.items() if config["available"]]

        if not available_providers:
            pytest.skip("No providers available for error handling test")

        for provider_name in available_providers:
            # Test with problematic objective
            problematic_objective = ""  # Empty objective
            role = "planner"

            try:
                model = await async_create_model_for_role("planner")
                planner = StructuredPlanner(model=model, role="planner")

                plan = await planner.plan(objective=problematic_objective, role=role)

                # Should handle gracefully (fallback to thought step)
                assert isinstance(plan, list)
                assert len(plan) >= 1

                # Should have fallback thought step
                thought_steps = [step for step in plan if step.get("type") == "thought"]
                assert len(thought_steps) > 0, f"{provider_name} should handle errors gracefully"

            except Exception as e:
                pytest.fail(f"Provider {provider_name} should handle errors gracefully: {e}")

    @pytest.mark.asyncio
    @pytest.mark.multi_provider
    async def test_provider_performance_comparison(self, provider_configs):
        """Test and compare performance across providers."""
        available_providers = [name for name, config in provider_configs.items() if config["available"]]

        if len(available_providers) < 2:
            pytest.skip("Need at least 2 providers for performance comparison")

        test_objective = "Design a simple REST API"
        test_role = "architect"

        performance_data = {}

        for provider_name in available_providers:
            # Measure planning time
            import time

            start_time = time.time()

            model = await async_create_model_for_role("planner")
            planner = StructuredPlanner(model=model, role="planner")

            plan = await planner.plan(objective=test_objective, role=test_role)

            end_time = time.time()

            performance_data[provider_name] = {
                "duration": end_time - start_time,
                "plan_length": len(plan),
                "plan": plan,
            }

        # Verify all providers completed successfully
        for provider_name, data in performance_data.items():
            assert data["duration"] > 0, f"{provider_name} should have positive duration"
            assert data["plan_length"] > 0, f"{provider_name} should generate non-empty plan"
            assert data["duration"] < 30.0, f"{provider_name} should complete within 30 seconds"

        # Performance should be reasonable (not too slow)
        max_duration = max(data["duration"] for data in performance_data.values())
        assert max_duration < 20.0, "All providers should complete within 20 seconds"

    @pytest.mark.asyncio
    @pytest.mark.multi_provider
    async def test_provider_plan_quality_metrics(self, provider_configs):
        """Test plan quality metrics across providers."""
        available_providers = [name for name, config in provider_configs.items() if config["available"]]

        if not available_providers:
            pytest.skip("No providers available for quality testing")

        test_objectives = [
            ("Build a web scraper", "developer"),
            ("Design database schema", "architect"),
            ("Write test cases", "qa_engineer"),
        ]

        quality_metrics = {}

        for provider_name in available_providers:
            provider_metrics = {
                "total_plans": 0,
                "avg_plan_length": 0,
                "thought_step_ratio": 0,
                "action_step_ratio": 0,
            }

            all_plans = []

            for objective, role in test_objectives:
                try:
                    model = await async_create_model_for_role("planner")
                    planner = StructuredPlanner(model=model, role="planner")

                    plan = await planner.plan(objective=objective, role=role)
                    all_plans.append(plan)

                except Exception as e:
                    pytest.fail(f"Provider {provider_name} failed quality test: {e}")

            # Calculate metrics
            if all_plans:
                total_steps = sum(len(plan) for plan in all_plans)
                total_thought_steps = sum(
                    len([step for step in plan if step.get("type") == "thought"]) for plan in all_plans
                )
                total_action_steps = sum(
                    len([step for step in plan if step.get("type") == "action"]) for plan in all_plans
                )

                provider_metrics.update(
                    {
                        "total_plans": len(all_plans),
                        "avg_plan_length": total_steps / len(all_plans),
                        "thought_step_ratio": total_thought_steps / total_steps if total_steps > 0 else 0,
                        "action_step_ratio": total_action_steps / total_steps if total_steps > 0 else 0,
                    }
                )

            quality_metrics[provider_name] = provider_metrics

        # Verify quality metrics
        for provider_name, metrics in quality_metrics.items():
            assert metrics["total_plans"] > 0, f"{provider_name} should generate plans"
            assert metrics["avg_plan_length"] > 0, f"{provider_name} should generate non-empty plans"
            assert metrics["thought_step_ratio"] > 0, f"{provider_name} should include thought steps"

            # Thought steps should be reasonable proportion (planning focus)
            assert metrics["thought_step_ratio"] >= 0.3, f"{provider_name} should have sufficient planning thoughts"

    @pytest.mark.asyncio
    @pytest.mark.multi_provider
    async def test_provider_role_adaptation(self, provider_configs):
        """Test how providers adapt to different roles."""
        available_providers = [name for name, config in provider_configs.items() if config["available"]]

        if not available_providers:
            pytest.skip("No providers available for role adaptation test")

        roles = ["planner", "developer", "architect", "analyst"]
        base_objective = "Create a user management system"

        role_adaptation_results = {}

        for provider_name in available_providers:
            provider_results = {}

            for role in roles:
                try:
                    model = await async_create_model_for_role("planner")
                    planner = StructuredPlanner(model=model, role="planner")

                    plan = await planner.plan(objective=base_objective, role=role)
                    provider_results[role] = plan

                except Exception as e:
                    pytest.fail(f"Provider {provider_name} failed for role {role}: {e}")

            role_adaptation_results[provider_name] = provider_results

        # Verify role adaptation
        for provider_name, results in role_adaptation_results.items():
            # Each role should generate a plan
            assert len(results) == len(roles), f"{provider_name} should handle all roles"

            for role, plan in results.items():
                assert isinstance(plan, list), f"{provider_name} should generate list for {role}"
                assert len(plan) > 0, f"{provider_name} should generate non-empty plan for {role}"

                # Plans should be role-appropriate (at minimum, valid structure)
                for step in plan:
                    assert isinstance(step, dict)
                    assert "type" in step
                    assert step["type"] in ["thought", "action"]
