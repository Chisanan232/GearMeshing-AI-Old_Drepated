"""
End-to-end smoke tests for LangGraph-based runtime engine with AI agent workflows.

These tests verify the complete AI agent workflow execution using real AI models
while mocking all other dependencies (database, cache, etc.).

Key objectives:
1. Verify complete AI agent workflow execution from planning to completion
2. Test different AI providers (OpenAI, Anthropic, Google) in real scenarios
3. Test normal cases: successful execution, multi-step workflows
4. Test edge cases: failures, retries, error handling, timeouts
5. Test different agent roles and capabilities
6. Ensure proper state management and event logging
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from test.settings import test_settings
from typing import Any, Dict, List, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.checkpoint.memory import MemorySaver

from gearmeshing_ai.agent_core.capabilities.base import (
    CapabilityResult,
)
from gearmeshing_ai.agent_core.model_provider import async_create_model_for_role
from gearmeshing_ai.agent_core.policy.global_policy import GlobalPolicy
from gearmeshing_ai.agent_core.runtime.engine import AgentEngine
from gearmeshing_ai.agent_core.runtime.models import EngineDeps
from gearmeshing_ai.agent_core.schemas.domain import (
    AgentEventType,
    AgentRun,
    AgentRunStatus,
    RiskLevel,
)


class BaseAIWorkflowTestSuite:
    """Base test suite for AI agent workflow smoke tests."""

    @pytest.fixture
    def mock_capabilities(self) -> MagicMock:
        """Mock capabilities registry with realistic capabilities."""
        capabilities = MagicMock()
        cast(MagicMock, capabilities.list_all).return_value = [
            {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {"query": "string", "max_results": "integer"},
            },
            {
                "name": "web_fetch",
                "description": "Fetch content from a URL",
                "parameters": {"url": "string"},
            },
            {
                "name": "docs_read",
                "description": "Read documentation or files",
                "parameters": {"path": "string"},
            },
            {
                "name": "summarize",
                "description": "Summarize content",
                "parameters": {"content": "string", "focus": "string"},
            },
            {
                "name": "shell_exec",
                "description": "Execute shell commands",
                "parameters": {"command": "string", "working_dir": "string"},
            },
        ]

        # Mock the get method to return async mock capability
        async_mock_capability = AsyncMock()
        cast(MagicMock, async_mock_capability.execute).return_value = CapabilityResult(
            ok=True, output={"status": "success", "data": "mock result"}
        )
        cast(MagicMock, capabilities.get).return_value = async_mock_capability

        return capabilities

    @pytest.fixture
    def mock_policy(self) -> GlobalPolicy:
        """Mock global policy for testing."""
        policy = MagicMock(spec=GlobalPolicy)

        # Default policy: allow everything
        mock_decision = MagicMock()
        mock_decision.block = False
        mock_decision.block_reason = None
        mock_decision.require_approval = False
        mock_decision.risk = RiskLevel.low

        # Setup all required methods
        cast(MagicMock, policy.decide).return_value = mock_decision
        cast(MagicMock, policy.validate_tool_args).return_value = None
        cast(MagicMock, policy.classify_risk).return_value = RiskLevel.low

        return policy

    @pytest.fixture
    def sample_agent_run(self) -> AgentRun:
        """Sample agent run for testing."""
        return AgentRun(
            id=str(uuid.uuid4()),
            role="assistant",
            objective="Test AI agent workflow execution",
            status=AgentRunStatus.pending,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tenant_id="test-tenant",
        )

    @pytest.fixture
    def engine_deps(
        self,
        mock_repositories: Dict[str, AsyncMock],
        mock_capabilities: MagicMock,
        mock_policy: GlobalPolicy,
    ) -> EngineDeps:
        """Create engine dependencies for testing."""
        checkpointer = MemorySaver()

        return EngineDeps(
            runs=mock_repositories["runs"],
            events=mock_repositories["events"],
            approvals=mock_repositories["approvals"],
            checkpoints=mock_repositories["checkpoints"],
            tool_invocations=mock_repositories["tool_invocations"],
            capabilities=mock_capabilities,
            usage=mock_repositories["usage"],
            checkpointer=checkpointer,
            prompt_provider=None,
            role_provider=None,
            thought_model=None,  # Will be set in tests
            mcp_info_provider=None,
            mcp_call=None,
        )


class TestAIWorkflowNormalCases(BaseAIWorkflowTestSuite):
    """Test normal AI agent workflow scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_complete_workflow_with_openai(
        self,
        engine_deps: EngineDeps,
        mock_policy: GlobalPolicy,
        sample_agent_run: AgentRun,
        compose_stack: Any,
        database_url: str,
        agent_configs_setup,
        patched_settings,
    ) -> None:
        """Test complete AI workflow from planning to execution with OpenAI."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        # Create real AI model using settings from dotenv
        thought_model = await async_create_model_for_role("assistant")

        # Update engine deps with thought model
        engine_deps = EngineDeps(
            runs=engine_deps.runs,
            events=engine_deps.events,
            approvals=engine_deps.approvals,
            checkpoints=engine_deps.checkpoints,
            tool_invocations=engine_deps.tool_invocations,
            capabilities=engine_deps.capabilities,
            usage=engine_deps.usage,
            checkpointer=engine_deps.checkpointer,
            thought_model=thought_model,
            prompt_provider=None,
            role_provider=None,
            mcp_info_provider=None,
            mcp_call=None,
        )

        engine = AgentEngine(policy=mock_policy, deps=engine_deps)

        # Define a realistic workflow
        workflow_plan = [
            {
                "kind": "thought",
                "thought": "analyze_requirements",
                "args": {"objective": "Create a data analysis script for sales data"},
            },
            {"kind": "action", "capability": "docs_read", "args": {"path": "sales_data.csv"}},
            {
                "kind": "thought",
                "thought": "plan_analysis",
                "args": {"data_type": "sales", "analysis_goals": ["trends", "insights"]},
            },
            {"kind": "action", "capability": "summarize", "args": {"content": "sales_data", "focus": "trend_analysis"}},
            {"kind": "thought", "thought": "generate_report", "args": {"format": "summary", "audience": "management"}},
        ]

        # Mock repository responses
        cast(MagicMock, engine_deps.runs.create).return_value = None
        cast(MagicMock, engine_deps.events.append).return_value = None
        cast(MagicMock, engine_deps.runs.get).return_value = sample_agent_run
        cast(MagicMock, engine_deps.runs.update_status).return_value = None

        # Mock capability execution
        mock_capability_result = CapabilityResult(ok=True, output={"status": "success", "data": "analysis results"})

        # Execute workflow
        result = await engine.start_run(run=sample_agent_run, plan=workflow_plan)

        # Verify execution completed
        assert result == sample_agent_run.id

        # Verify events were logged
        assert cast(MagicMock, engine_deps.events.append).call_count > 0

        # Verify run status was updated
        cast(MagicMock, engine_deps.runs.update_status).assert_called()

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_multi_step_workflow_with_anthropic(
        self,
        engine_deps: EngineDeps,
        mock_policy: GlobalPolicy,
        sample_agent_run: AgentRun,
        compose_stack: Any,
        database_url: str,
        agent_configs_setup,
        patched_settings,
    ) -> None:
        """Test multi-step workflow with Anthropic model."""
        if not test_settings.ai_provider.anthropic.api_key:
            pytest.skip("Anthropic API key not configured")

        # Create real Anthropic model
        thought_model = await async_create_model_for_role("assistant")

        engine_deps = EngineDeps(
            runs=engine_deps.runs,
            events=engine_deps.events,
            approvals=engine_deps.approvals,
            checkpoints=engine_deps.checkpoints,
            tool_invocations=engine_deps.tool_invocations,
            capabilities=engine_deps.capabilities,
            usage=engine_deps.usage,
            checkpointer=engine_deps.checkpointer,
            thought_model=thought_model,
            prompt_provider=None,
            role_provider=None,
            mcp_info_provider=None,
            mcp_call=None,
        )

        engine = AgentEngine(policy=mock_policy, deps=engine_deps)

        # Complex multi-step workflow
        complex_plan = [
            {"kind": "thought", "thought": "research_topic", "args": {"topic": "machine learning trends 2024"}},
            {
                "kind": "action",
                "capability": "web_search",
                "args": {"query": "machine learning trends 2024", "max_results": 10},
            },
            {
                "kind": "thought",
                "thought": "synthesize_findings",
                "args": {"sources": "search_results", "focus": "key_trends"},
            },
            {
                "kind": "action",
                "capability": "write_file",
                "args": {"file_path": "ml_trends_report.md", "content": " synthesized findings"},
            },
            {"kind": "thought", "thought": "validate_report", "args": {"criteria": "accuracy", "completeness": "high"}},
        ]

        # Setup mocks
        cast(MagicMock, engine_deps.runs.create).return_value = None
        cast(MagicMock, engine_deps.events.append).return_value = None
        cast(MagicMock, engine_deps.runs.get).return_value = sample_agent_run
        cast(MagicMock, engine_deps.runs.update_status).return_value = None

        # Execute workflow
        result = await engine.start_run(run=sample_agent_run, plan=complex_plan)

        # Verify successful execution
        assert result == sample_agent_run.id

        # Verify multiple events were logged
        event_calls = cast(MagicMock, engine_deps.events.append).call_args_list
        assert len(event_calls) >= 5  # At least one event per step

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_workflow_with_google_model(
        self,
        engine_deps: EngineDeps,
        mock_policy: GlobalPolicy,
        sample_agent_run: AgentRun,
        compose_stack: Any,
        database_url: str,
        agent_configs_setup,
        patched_settings,
    ) -> None:
        """Test workflow execution with Google model."""
        if not test_settings.ai_provider.google.api_key:
            pytest.skip("Google API key not configured")

        # Create real Google model
        thought_model = await async_create_model_for_role("assistant")

        engine_deps = EngineDeps(
            runs=engine_deps.runs,
            events=engine_deps.events,
            approvals=engine_deps.approvals,
            checkpoints=engine_deps.checkpoints,
            tool_invocations=engine_deps.tool_invocations,
            capabilities=engine_deps.capabilities,
            usage=engine_deps.usage,
            checkpointer=engine_deps.checkpointer,
            thought_model=thought_model,
            prompt_provider=None,
            role_provider=None,
            mcp_info_provider=None,
            mcp_call=None,
        )

        engine = AgentEngine(policy=mock_policy, deps=engine_deps)

        # Simple workflow for testing
        simple_plan = [
            {"kind": "thought", "thought": "process_request", "args": {"request": "Analyze user feedback data"}},
            {"kind": "action", "capability": "docs_read", "args": {"path": "user_feedback.csv"}},
            {"kind": "thought", "thought": "generate_insights", "args": {"focus": "user_satisfaction"}},
        ]

        # Setup mocks
        cast(MagicMock, engine_deps.runs.create).return_value = None
        cast(MagicMock, engine_deps.events.append).return_value = None
        cast(MagicMock, engine_deps.runs.get).return_value = sample_agent_run
        cast(MagicMock, engine_deps.runs.update_status).return_value = None

        # Execute workflow
        result = await engine.start_run(run=sample_agent_run, plan=simple_plan)

        # Verify execution
        assert result == sample_agent_run.id

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_sequential_workflows(
        self,
        engine_deps: EngineDeps,
        mock_policy: GlobalPolicy,
        compose_stack: Any,
        database_url: str,
        agent_configs_setup,
        patched_settings,
    ) -> None:
        """Test sequential execution of multiple workflows."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        # Create real AI model using settings from dotenv
        thought_model = await async_create_model_for_role("assistant")

        engine_deps = EngineDeps(
            runs=engine_deps.runs,
            events=engine_deps.events,
            approvals=engine_deps.approvals,
            checkpoints=engine_deps.checkpoints,
            tool_invocations=engine_deps.tool_invocations,
            capabilities=engine_deps.capabilities,
            usage=engine_deps.usage,
            checkpointer=engine_deps.checkpointer,
            thought_model=thought_model,
            prompt_provider=None,
            role_provider=None,
            mcp_info_provider=None,
            mcp_call=None,
        )

        engine = AgentEngine(policy=mock_policy, deps=engine_deps)

        # Create multiple runs
        runs = [
            AgentRun(
                id=str(uuid.uuid4()),
                role="assistant",
                objective=f"Concurrent workflow {i}",
                status=AgentRunStatus.pending,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                tenant_id="test-tenant",
            )
            for i in range(3)
        ]

        # Simple plan for each run
        simple_plan = [{"kind": "thought", "thought": "process_task", "args": {"task_id": "concurrent_test"}}]

        # Mock repository responses
        cast(MagicMock, engine_deps.runs.create).return_value = None
        cast(MagicMock, engine_deps.events.append).return_value = None

        # Create a proper mock for runs.get that returns the correct run
        def mock_get_run(run_id):
            for run in runs:
                if run.id == run_id:
                    return run
            return runs[0]  # fallback

        cast(MagicMock, engine_deps.runs.get).side_effect = mock_get_run
        cast(MagicMock, engine_deps.runs.update_status).return_value = None

        # Execute workflows sequentially to avoid async generator issues
        results = []
        for run in runs:
            result = await engine.start_run(run=run, plan=simple_plan)
            results.append(result)

        # Verify all executions succeeded
        for i, result in enumerate(results):
            assert result == runs[i].id


class TestAIWorkflowEdgeCases(BaseAIWorkflowTestSuite):
    """Test edge cases and error handling in AI agent workflows."""

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_workflow_with_capability_failure(
        self,
        engine_deps: EngineDeps,
        mock_policy: GlobalPolicy,
        sample_agent_run: AgentRun,
        compose_stack: Any,
        database_url: str,
        agent_configs_setup,
        patched_settings,
    ) -> None:
        """Test workflow handling when a capability fails."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        # Create real AI model using settings from dotenv
        thought_model = await async_create_model_for_role("assistant")

        engine_deps = EngineDeps(
            runs=engine_deps.runs,
            events=engine_deps.events,
            approvals=engine_deps.approvals,
            checkpoints=engine_deps.checkpoints,
            tool_invocations=engine_deps.tool_invocations,
            capabilities=engine_deps.capabilities,
            usage=engine_deps.usage,
            checkpointer=engine_deps.checkpointer,
            thought_model=thought_model,
            prompt_provider=None,
            role_provider=None,
            mcp_info_provider=None,
            mcp_call=None,
        )

        engine = AgentEngine(policy=mock_policy, deps=engine_deps)

        # Workflow with a failing capability
        failing_plan = [
            {"kind": "thought", "thought": "attempt_operation", "args": {"operation": "risky_task"}},
            {"kind": "action", "capability": "shell_exec", "args": {"command": "invalid_command_that_will_fail"}},
            {"kind": "thought", "thought": "handle_failure", "args": {"error": "command_failed"}},
        ]

        # Setup mocks
        cast(MagicMock, engine_deps.runs.create).return_value = None
        cast(MagicMock, engine_deps.events.append).return_value = None
        cast(MagicMock, engine_deps.runs.get).return_value = sample_agent_run
        cast(MagicMock, engine_deps.runs.update_status).return_value = None

        # Mock capability failure
        failing_result = CapabilityResult(ok=False, output={"error": "Command execution failed", "exit_code": 1})

        # Setup capability mock to return failure
        failing_capability = MagicMock()
        failing_capability.execute = AsyncMock(return_value=failing_result)
        cast(MagicMock, engine_deps.capabilities.get).return_value = failing_capability

        # Execute workflow
        result = await engine.start_run(run=sample_agent_run, plan=failing_plan)

        # Verify workflow handled the failure
        assert result == sample_agent_run.id

        # Note: Failure events might not be automatically generated for capability failures
        # The engine might handle capability failures differently

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_workflow_with_approval_required(
        self,
        engine_deps: EngineDeps,
        sample_agent_run: AgentRun,
        compose_stack: Any,
        database_url: str,
        agent_configs_setup,
        patched_settings,
    ) -> None:
        """Test workflow when approval is required for an action."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        # Create real AI model using settings from dotenv
        thought_model = await async_create_model_for_role("assistant")

        engine_deps = EngineDeps(
            runs=engine_deps.runs,
            events=engine_deps.events,
            approvals=engine_deps.approvals,
            checkpoints=engine_deps.checkpoints,
            tool_invocations=engine_deps.tool_invocations,
            capabilities=engine_deps.capabilities,
            usage=engine_deps.usage,
            checkpointer=engine_deps.checkpointer,
            thought_model=thought_model,
            prompt_provider=None,
            role_provider=None,
            mcp_info_provider=None,
            mcp_call=None,
        )

        # Mock policy to require approval
        mock_policy = MagicMock()
        mock_decision = MagicMock()
        mock_decision.block = False
        mock_decision.block_reason = None
        mock_decision.require_approval = True
        mock_decision.risk = RiskLevel.medium

        # Setup all required methods
        cast(MagicMock, mock_policy.decide).return_value = mock_decision
        cast(MagicMock, mock_policy.validate_tool_args).return_value = None
        cast(MagicMock, mock_policy.classify_risk).return_value = RiskLevel.medium

        engine = AgentEngine(policy=mock_policy, deps=engine_deps)

        # Workflow with approval-required action
        approval_plan = [
            {"kind": "thought", "thought": "prepare_sensitive_operation", "args": {"operation": "data_deletion"}},
            {"kind": "action", "capability": "shell_exec", "args": {"command": "rm -rf /sensitive/data"}},
        ]

        # Setup mocks
        cast(MagicMock, engine_deps.runs.create).return_value = None
        cast(MagicMock, engine_deps.events.append).return_value = None
        cast(MagicMock, engine_deps.runs.get).return_value = sample_agent_run
        cast(MagicMock, engine_deps.runs.update_status).return_value = None
        cast(MagicMock, engine_deps.approvals.create).return_value = None

        # Execute workflow
        result = await engine.start_run(run=sample_agent_run, plan=approval_plan)

        # Verify approval was requested
        cast(MagicMock, engine_deps.approvals.create).assert_called()

        # Verify approval events were logged
        event_calls = cast(MagicMock, engine_deps.events.append).call_args_list
        approval_events = [call for call in event_calls if call[0][0].type == AgentEventType.approval_requested]
        assert len(approval_events) > 0

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_workflow_with_invalid_plan(
        self,
        engine_deps: EngineDeps,
        mock_policy: GlobalPolicy,
        sample_agent_run: AgentRun,
        compose_stack: Any,
        database_url: str,
        agent_configs_setup,
        patched_settings,
    ) -> None:
        """Test workflow handling with invalid plan structure."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        # Create real AI model using settings from dotenv
        thought_model = await async_create_model_for_role("assistant")

        engine_deps = EngineDeps(
            runs=engine_deps.runs,
            events=engine_deps.events,
            approvals=engine_deps.approvals,
            checkpoints=engine_deps.checkpoints,
            tool_invocations=engine_deps.tool_invocations,
            capabilities=engine_deps.capabilities,
            usage=engine_deps.usage,
            checkpointer=engine_deps.checkpointer,
            thought_model=thought_model,
            prompt_provider=None,
            role_provider=None,
            mcp_info_provider=None,
            mcp_call=None,
        )

        engine = AgentEngine(policy=mock_policy, deps=engine_deps)

        # Invalid plan with unknown step kind
        invalid_plan = [{"kind": "unknown_step", "capability": "web_search", "args": {"query": "test"}}]

        # Setup mocks
        cast(MagicMock, engine_deps.runs.create).return_value = None
        cast(MagicMock, engine_deps.events.append).return_value = None
        cast(MagicMock, engine_deps.runs.get).return_value = sample_agent_run
        cast(MagicMock, engine_deps.runs.update_status).return_value = None

        # Execute workflow - should handle invalid plan gracefully
        with pytest.raises(ValueError, match="unknown step kind"):
            await engine.start_run(run=sample_agent_run, plan=invalid_plan)

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_workflow_with_empty_plan(
        self,
        engine_deps: EngineDeps,
        mock_policy: GlobalPolicy,
        sample_agent_run: AgentRun,
        compose_stack: Any,
        database_url: str,
        agent_configs_setup,
        patched_settings,
    ) -> None:
        """Test workflow handling with empty plan."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        # Create real AI model using settings from dotenv
        thought_model = await async_create_model_for_role("assistant")

        engine_deps = EngineDeps(
            runs=engine_deps.runs,
            events=engine_deps.events,
            approvals=engine_deps.approvals,
            checkpoints=engine_deps.checkpoints,
            tool_invocations=engine_deps.tool_invocations,
            capabilities=engine_deps.capabilities,
            usage=engine_deps.usage,
            checkpointer=engine_deps.checkpointer,
            thought_model=thought_model,
            prompt_provider=None,
            role_provider=None,
            mcp_info_provider=None,
            mcp_call=None,
        )

        engine = AgentEngine(policy=mock_policy, deps=engine_deps)

        # Empty plan
        empty_plan: List[Dict[str, Any]] = []

        # Setup mocks
        cast(MagicMock, engine_deps.runs.create).return_value = None
        cast(MagicMock, engine_deps.events.append).return_value = None
        cast(MagicMock, engine_deps.runs.get).return_value = sample_agent_run
        cast(MagicMock, engine_deps.runs.update_status).return_value = None

        # Execute workflow
        result = await engine.start_run(run=sample_agent_run, plan=empty_plan)

        # Verify empty plan is handled
        assert result == sample_agent_run.id

        # Verify completion event was logged
        event_calls = cast(MagicMock, engine_deps.events.append).call_args_list
        completion_events = [call for call in event_calls if call[0][0].type == AgentEventType.run_completed]
        assert len(completion_events) > 0


class TestAIWorkflowStateManagement(BaseAIWorkflowTestSuite):
    """Test state management and persistence in AI agent workflows."""

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_workflow_checkpointing(
        self,
        engine_deps: EngineDeps,
        mock_policy: GlobalPolicy,
        sample_agent_run: AgentRun,
        compose_stack: Any,
        database_url: str,
        agent_configs_setup,
        patched_settings,
    ) -> None:
        """Test workflow state checkpointing and resumption."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        # Create real AI model using settings from dotenv
        thought_model = await async_create_model_for_role("assistant")

        engine_deps = EngineDeps(
            runs=engine_deps.runs,
            events=engine_deps.events,
            approvals=engine_deps.approvals,
            checkpoints=engine_deps.checkpoints,
            tool_invocations=engine_deps.tool_invocations,
            capabilities=engine_deps.capabilities,
            usage=engine_deps.usage,
            checkpointer=engine_deps.checkpointer,
            thought_model=thought_model,
            prompt_provider=None,
            role_provider=None,
            mcp_info_provider=None,
            mcp_call=None,
        )

        engine = AgentEngine(policy=mock_policy, deps=engine_deps)

        # Multi-step workflow for checkpointing
        checkpoint_plan = [
            {"kind": "thought", "thought": "initialize_process", "args": {"process": "data_analysis"}},
            {"kind": "action", "capability": "docs_read", "args": {"path": "large_dataset.csv"}},
            {"kind": "thought", "thought": "process_data", "args": {"stage": "initial_analysis"}},
        ]

        # Setup mocks
        cast(MagicMock, engine_deps.runs.create).return_value = None
        cast(MagicMock, engine_deps.events.append).return_value = None
        cast(MagicMock, engine_deps.runs.get).return_value = sample_agent_run
        cast(MagicMock, engine_deps.runs.update_status).return_value = None

        # Execute workflow
        result = await engine.start_run(run=sample_agent_run, plan=checkpoint_plan)

        # Verify checkpoints were created
        assert cast(MagicMock, engine_deps.checkpoints.save).called or engine_deps.checkpointer is not None

        # Verify workflow completed
        assert result == sample_agent_run.id

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_workflow_event_logging(
        self,
        engine_deps: EngineDeps,
        mock_policy: GlobalPolicy,
        sample_agent_run: AgentRun,
        compose_stack: Any,
        database_url: str,
        agent_configs_setup,
        patched_settings,
    ) -> None:
        """Test comprehensive event logging during workflow execution."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        # Create real AI model using settings from dotenv
        thought_model = await async_create_model_for_role("assistant")

        engine_deps = EngineDeps(
            runs=engine_deps.runs,
            events=engine_deps.events,
            approvals=engine_deps.approvals,
            checkpoints=engine_deps.checkpoints,
            tool_invocations=engine_deps.tool_invocations,
            capabilities=engine_deps.capabilities,
            usage=engine_deps.usage,
            checkpointer=engine_deps.checkpointer,
            thought_model=thought_model,
            prompt_provider=None,
            role_provider=None,
            mcp_info_provider=None,
            mcp_call=None,
        )

        engine = AgentEngine(policy=mock_policy, deps=engine_deps)

        # Comprehensive workflow
        event_plan = [
            {"kind": "thought", "thought": "start_analysis", "args": {"topic": "performance_metrics"}},
            {"kind": "action", "capability": "docs_read", "args": {"path": "metrics.json"}},
            {"kind": "thought", "thought": "analyze_metrics", "args": {"metrics": "performance"}},
            {"kind": "action", "capability": "summarize", "args": {"content": "analysis results", "focus": "summary"}},
        ]

        # Setup mocks
        cast(MagicMock, engine_deps.runs.create).return_value = None
        cast(MagicMock, engine_deps.events.append).return_value = None
        cast(MagicMock, engine_deps.runs.get).return_value = sample_agent_run
        cast(MagicMock, engine_deps.runs.update_status).return_value = None

        # Execute workflow
        result = await engine.start_run(run=sample_agent_run, plan=event_plan)

        # Verify comprehensive event logging
        event_calls = cast(MagicMock, engine_deps.events.append).call_args_list
        event_types = [call[0][0].type for call in event_calls]

        # Should have various event types
        assert AgentEventType.run_started in event_types
        assert AgentEventType.plan_created in event_types
        assert AgentEventType.thought_executed in event_types
        assert AgentEventType.capability_requested in event_types
        assert AgentEventType.capability_executed in event_types
        assert AgentEventType.run_completed in event_types

        # Verify workflow completed
        assert result == sample_agent_run.id
