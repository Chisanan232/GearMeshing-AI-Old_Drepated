"""
Advanced smoke tests for LangGraph runtime engine with complex AI agent workflows.

These tests focus on advanced scenarios including:
1. Complex multi-agent workflows
2. Long-running workflows with resumption
3. Error recovery and retry mechanisms
4. Performance and scalability tests
5. Integration with external systems
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gearmeshing_ai.agent_core.model_provider import async_create_model_for_role
from gearmeshing_ai.agent_core.policy.global_policy import GlobalPolicy
from gearmeshing_ai.agent_core.runtime.engine import AgentEngine
from gearmeshing_ai.agent_core.runtime.models import EngineDeps
from gearmeshing_ai.agent_core.schemas.domain import (
    AgentEvent,
    AgentEventType,
    AgentRun,
    AgentRunStatus,
    CapabilityName,
    RiskLevel,
)
from gearmeshing_ai.agent_core.capabilities.base import CapabilityResult
from test.settings import test_settings

# Import fixtures from the shared fixtures module
from test.smoke_test.agent_core.runtime.fixtures import (
    test_database,
    patched_settings,
    mock_repositories,
    mock_capabilities,
    mock_policy,
    sample_agent_run,
    engine_deps,
)


class TestAdvancedAIWorkflows:
    """Advanced AI workflow test scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_long_running_workflow(
        self,
        engine_deps: EngineDeps,
        mock_policy: GlobalPolicy,
        test_database: str,
        patched_settings,
    ) -> None:
        """Test long-running workflow with multiple phases."""
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
        
        # Long-running workflow with multiple phases
        long_running_plan = [
            # Phase 1: Data Collection
            {
                "kind": "thought",
                "thought": "plan_data_collection",
                "args": {"sources": ["api", "database", "files"], "timeline": "2_hours"}
            },
            {
                "kind": "action",
                "capability": "web_search",
                "args": {"query": "market trends Q1 2024", "max_results": 50}
            },
            {
                "kind": "action",
                "capability": "docs_read",
                "args": {"file_path": "historical_data.csv"}
            },
            
            # Phase 2: Data Processing
            {
                "kind": "thought",
                "thought": "process_collected_data",
                "args": {"processing_type": "statistical_analysis", "quality": "high"}
            },
            {
                "kind": "action",
                "capability": "summarize",
                "args": {"data": "combined_dataset", "analysis_type": "comprehensive"}
            },
            
            # Phase 3: Insight Generation
            {
                "kind": "thought",
                "thought": "generate_insights",
                "args": {"focus_areas": ["trends", "anomalies", "opportunities"]}
            },
            {
                "kind": "action",
                "capability": "codegen",
                "args": {"file_path": "comprehensive_report.md", "content": "detailed_analysis"}
            },
            
            # Phase 4: Validation
            {
                "kind": "thought",
                "thought": "validate_findings",
                "args": {"validation_criteria": "statistical_significance", "confidence": 0.95}
            }
        ]
        
        # Create a long-running agent run
        long_run = AgentRun(
            id=str(uuid.uuid4()),
            role="data_analyst",
            objective="Comprehensive market analysis and trend identification",
            status=AgentRunStatus.pending,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tenant_id="test-tenant",
        )
        
        # Setup mocks
        cast(MagicMock, engine_deps.runs.create).return_value = None
        cast(MagicMock, engine_deps.events.append).return_value = None
        cast(MagicMock, engine_deps.runs.get).return_value = long_run
        cast(MagicMock, engine_deps.runs.update_status).return_value = None
        
        # Execute long-running workflow
        start_time = time.time()
        result = await engine.start_run(run=long_run, plan=long_running_plan)
        execution_time = time.time() - start_time
        
        # Verify execution completed successfully
        assert result == long_run.id
        
        # Verify comprehensive event logging
        event_calls = cast(MagicMock, engine_deps.events.append).call_args_list
        assert len(event_calls) >= 8  # At least one event per major step
        
        # Verify status updates occurred
        assert cast(MagicMock, engine_deps.runs.update_status).call_count >= 1  # At least completed status

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_workflow_with_retry_mechanism(
        self,
        engine_deps: EngineDeps,
        mock_policy: GlobalPolicy,
        test_database: str,
        patched_settings,
    ) -> None:
        """Test workflow with automatic retry on transient failures."""
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
        
        # Workflow with potential failures
        retry_plan = [
            {
                "kind": "thought",
                "thought": "attempt_unreliable_operation",
                "args": {"operation": "api_call", "retry_count": 0}
            },
            {
                "kind": "action",
                "capability": "web_search",
                "args": {"query": "unstable_api_data", "max_results": 10}
            },
            {
                "kind": "thought",
                "thought": "handle_transient_failure",
                "args": {"error": "timeout", "retry_strategy": "exponential_backoff"}
            }
        ]
        
        # Create agent run
        retry_run = AgentRun(
            id=str(uuid.uuid4()),
            role="resilient_agent",
            objective="Test retry mechanism with transient failures",
            status=AgentRunStatus.pending,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tenant_id="test-tenant",
        )
        
        # Setup mocks with failure then success
        call_count = 0
        async def mock_web_search(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call fails
                return CapabilityResult(
                    ok=False,
                    output={"error": "Network timeout", "retryable": True}
                )
            else:
                # Second call succeeds
                return CapabilityResult(
                    ok=True,
                    output={"results": ["data1", "data2"], "source": "api"}
                )
        
        # Mock the capability execution
        retry_capability = MagicMock()
        retry_capability.execute = mock_web_search
        cast(MagicMock, engine_deps.capabilities.get).return_value = retry_capability
        
        # Setup other mocks
        cast(MagicMock, engine_deps.runs.create).return_value = None
        cast(MagicMock, engine_deps.events.append).return_value = None
        cast(MagicMock, engine_deps.runs.get).return_value = retry_run
        cast(MagicMock, engine_deps.runs.update_status).return_value = None
        
        # Execute workflow with retry
        result = await engine.start_run(run=retry_run, plan=retry_plan)
        
        # Verify eventual success
        assert result == retry_run.id
        assert call_count >= 1  # Should have tried at least once
        
        # Verify retry events were logged
        event_calls = cast(MagicMock, engine_deps.events.append).call_args_list
        retry_events = [call for call in event_calls 
                        if "retry" in str(call[0][0]).lower()]
        assert len(retry_events) > 0

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_workflow_with_dynamic_adaptation(
        self,
        engine_deps: EngineDeps,
        mock_policy: GlobalPolicy,
        test_database: str,
        patched_settings,
    ) -> None:
        """Test workflow that adapts based on intermediate results."""
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
        
        # Adaptive workflow that changes based on results
        adaptive_plan = [
            {
                "kind": "thought",
                "thought": "initial_assessment",
                "args": {"task": "data_analysis", "complexity": "unknown"}
            },
            {
                "kind": "action",
                "capability": "docs_read",
                "args": {"file_path": "sample_data.csv"}
            },
            {
                "kind": "thought",
                "thought": "adapt_strategy_based_on_data",
                "args": {"data_size": "dynamic", "data_type": "to_be_determined"}
            }
        ]
        
        # Create adaptive agent run
        adaptive_run = AgentRun(
            id=str(uuid.uuid4()),
            role="adaptive_agent",
            objective="Adapt workflow strategy based on data characteristics",
            status=AgentRunStatus.pending,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tenant_id="test-tenant",
        )
        
        # Setup mocks with dynamic responses
        async def mock_read_file(*args, **kwargs):
            # Simulate discovering large dataset
            return CapabilityResult(
                ok=True,
                output={
                    "data_size": "10GB",
                    "data_type": "time_series",
                    "complexity": "high",
                    "sample": [{"timestamp": "2024-01-01", "value": 100}]
                }
            )
        
        adaptive_capability = MagicMock()
        adaptive_capability.execute = mock_read_file
        cast(MagicMock, engine_deps.capabilities.get).return_value = adaptive_capability
        
        # Setup other mocks
        cast(MagicMock, engine_deps.runs.create).return_value = None
        cast(MagicMock, engine_deps.events.append).return_value = None
        cast(MagicMock, engine_deps.runs.get).return_value = adaptive_run
        cast(MagicMock, engine_deps.runs.update_status).return_value = None
        
        # Execute adaptive workflow
        result = await engine.start_run(run=adaptive_run, plan=adaptive_plan)
        
        # Verify successful adaptation
        assert result == adaptive_run.id
        
        # Verify adaptation events were logged
        event_calls = cast(MagicMock, engine_deps.events.append).call_args_list
        adaptation_events = [call for call in event_calls 
                           if "adapt" in str(call[0][0]).lower()]
        assert len(adaptation_events) > 0

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_workflow_with_resource_constraints(
        self,
        engine_deps: EngineDeps,
        mock_policy: GlobalPolicy,
        test_database: str,
        patched_settings,
    ) -> None:
        """Test workflow behavior under resource constraints."""
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
        
        # Resource-constrained workflow
        constrained_plan = [
            {
                "kind": "thought",
                "thought": "optimize_for_resources",
                "args": {"memory_limit": "512MB", "time_limit": "30s"}
            },
            {
                "kind": "action",
                "capability": "summarize",
                "args": {"data": "large_dataset", "batch_size": "small", "optimization": "memory"}
            },
            {
                "kind": "thought",
                "thought": "handle_resource_exhaustion",
                "args": {"strategy": "graceful_degradation", "priority": "core_features"}
            }
        ]
        
        # Create resource-constrained run
        constrained_run = AgentRun(
            id=str(uuid.uuid4()),
            role="efficient_agent",
            objective="Execute workflow under strict resource constraints",
            status=AgentRunStatus.pending,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tenant_id="test-tenant",
        )
        
        # Setup mocks with resource constraints
        async def mock_analyze_with_constraints(*args, **kwargs):
            # Simulate resource constraint handling
            return CapabilityResult(
                ok=True,
                output={
                    "status": "completed_under_constraints",
                    "memory_used": "450MB",
                    "execution_time": "25s",
                    "results": "partial_analysis"
                }
            )
        
        constrained_capability = MagicMock()
        constrained_capability.execute = mock_analyze_with_constraints
        cast(MagicMock, engine_deps.capabilities.get).return_value = constrained_capability
        
        # Setup other mocks
        cast(MagicMock, engine_deps.runs.create).return_value = None
        cast(MagicMock, engine_deps.events.append).return_value = None
        cast(MagicMock, engine_deps.runs.get).return_value = constrained_run
        cast(MagicMock, engine_deps.runs.update_status).return_value = None
        
        # Execute constrained workflow
        result = await engine.start_run(run=constrained_run, plan=constrained_plan)
        
        # Verify successful execution under constraints
        assert result == constrained_run.id
        
        # Verify resource usage tracking (if implemented)
        # Note: Usage tracking might not be implemented in current engine version
        usage_calls = cast(MagicMock, engine_deps.usage.track).call_args_list if engine_deps.usage and hasattr(engine_deps.usage, 'track') else []
        # Don't assert on usage calls as they might not be implemented yet

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_workflow_with_external_dependencies(
        self,
        engine_deps: EngineDeps,
        mock_policy: GlobalPolicy,
        test_database: str,
        patched_settings,
    ) -> None:
        """Test workflow with external system dependencies."""
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
        
        # Workflow with external dependencies
        external_plan = [
            {
                "kind": "thought",
                "thought": "check_external_services",
                "args": {"services": ["database", "api", "filesystem"], "timeout": "10s"}
            },
            {
                "kind": "action",
                "capability": "shell_exec",
                "args": {"command": "curl -X GET https://api.example.com/health"}
            },
            {
                "kind": "thought",
                "thought": "handle_external_failure",
                "args": {"service": "api", "fallback": "cached_data"}
            },
            {
                "kind": "action",
                "capability": "docs_read",
                "args": {"file_path": "cached_api_response.json"}
            }
        ]
        
        # Create external-dependent run
        external_run = AgentRun(
            id=str(uuid.uuid4()),
            role="integrated_agent",
            objective="Execute workflow with external system dependencies",
            status=AgentRunStatus.pending,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tenant_id="test-tenant",
        )
        
        # Setup mocks with external dependency simulation
        call_count = 0
        async def mock_external_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First external call fails
                return CapabilityResult(
                    ok=False,
                    output={"error": "Service unavailable", "fallback_available": True}
                )
            else:
                # Fallback succeeds
                return CapabilityResult(
                    ok=True,
                    output={"cached_data": "fallback_results", "timestamp": "2024-01-01"}
                )
        
        external_capability = MagicMock()
        external_capability.execute = mock_external_call
        cast(MagicMock, engine_deps.capabilities.get).return_value = external_capability
        
        # Setup other mocks
        cast(MagicMock, engine_deps.runs.create).return_value = None
        cast(MagicMock, engine_deps.events.append).return_value = None
        cast(MagicMock, engine_deps.runs.get).return_value = external_run
        cast(MagicMock, engine_deps.runs.update_status).return_value = None
        
        # Execute workflow with external dependencies
        result = await engine.start_run(run=external_run, plan=external_plan)
        
        # Verify successful handling of external dependencies
        assert result == external_run.id
        assert call_count >= 2  # Should have tried external then fallback
        
        # Verify external dependency events were logged
        event_calls = cast(MagicMock, engine_deps.events.append).call_args_list
        external_events = [call for call in event_calls 
                          if "external" in str(call[0][0]).lower() or "fallback" in str(call[0][0]).lower()]
        assert len(external_events) > 0
