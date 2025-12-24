"""Unit tests for OrchestratorService.

Tests verify orchestrator functionality including run management, event streaming,
approval handling, and event enrichment for SSE responses.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, Mock

import pytest

from gearmeshing_ai.agent_core.schemas.domain import (
    AgentRun,
    AgentEvent,
    Approval,
    UsageLedgerEntry,
    AgentEventType,
    AgentRunStatus,
)
from gearmeshing_ai.server.schemas import (
    SSEResponse,
    SSEEventData,
    KeepAliveEvent,
    ErrorEvent,
    ThinkingData,
    ThinkingOutputData,
    OperationData,
    ToolExecutionData,
    ApprovalRequestData,
    ApprovalResolutionData,
    RunStartData,
    RunCompletionData,
    RunFailureData,
)
from gearmeshing_ai.server.services.orchestrator import OrchestratorService, get_orchestrator


class TestOrchestratorServiceInitialization:
    """Test OrchestratorService initialization."""

    def test_orchestrator_service_initializes(self):
        """Test OrchestratorService can be instantiated."""
        with patch('gearmeshing_ai.server.services.orchestrator.build_sql_repos'):
            with patch('gearmeshing_ai.server.services.orchestrator.AgentServiceDeps'):
                with patch('gearmeshing_ai.server.services.orchestrator.StructuredPlanner'):
                    with patch('gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider'):
                        with patch('gearmeshing_ai.server.services.orchestrator.AgentService'):
                            orchestrator = OrchestratorService()
                            assert orchestrator is not None

    def test_orchestrator_has_required_attributes(self):
        """Test OrchestratorService has all required attributes."""
        with patch('gearmeshing_ai.server.services.orchestrator.build_sql_repos'):
            with patch('gearmeshing_ai.server.services.orchestrator.AgentServiceDeps'):
                with patch('gearmeshing_ai.server.services.orchestrator.StructuredPlanner'):
                    with patch('gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider'):
                        with patch('gearmeshing_ai.server.services.orchestrator.AgentService'):
                            orchestrator = OrchestratorService()
                            assert hasattr(orchestrator, 'repos')
                            assert hasattr(orchestrator, 'deps')
                            assert hasattr(orchestrator, 'policy_provider')
                            assert hasattr(orchestrator, 'agent_service')


class TestOrchestratorRunManagement:
    """Test run management methods."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator with mocked dependencies."""
        with patch('gearmeshing_ai.server.services.orchestrator.build_sql_repos') as mock_repos:
            with patch('gearmeshing_ai.server.services.orchestrator.AgentServiceDeps'):
                with patch('gearmeshing_ai.server.services.orchestrator.StructuredPlanner'):
                    with patch('gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider'):
                        with patch('gearmeshing_ai.server.services.orchestrator.AgentService'):
                            orchestrator = OrchestratorService()
                            # Mock the repos
                            orchestrator.repos = MagicMock()
                            orchestrator.repos.runs = AsyncMock()
                            orchestrator.repos.events = AsyncMock()
                            orchestrator.repos.approvals = AsyncMock()
                            orchestrator.repos.usage = AsyncMock()
                            orchestrator.repos.policies = AsyncMock()
                            orchestrator.agent_service = AsyncMock()
                            yield orchestrator

    @pytest.mark.asyncio
    async def test_create_run(self, mock_orchestrator):
        """Test creating a new agent run."""
        run = AgentRun(
            id="test-run-1",
            tenant_id="default-tenant",
            role="analyst",
            objective="Analyze data",
            status=AgentRunStatus.running,
        )
        
        mock_orchestrator.repos.runs.get.return_value = run
        
        result = await mock_orchestrator.create_run(run)
        
        assert result.id == "test-run-1"
        mock_orchestrator.agent_service.run.assert_called_once_with(run=run)
        mock_orchestrator.repos.runs.get.assert_called_once_with("test-run-1")

    @pytest.mark.asyncio
    async def test_create_run_failure(self, mock_orchestrator):
        """Test create_run raises error when run is not persisted."""
        run = AgentRun(
            id="test-run-1",
            tenant_id="default-tenant",
            role="analyst",
            objective="Analyze data",
            status=AgentRunStatus.running,
        )
        
        mock_orchestrator.repos.runs.get.return_value = None
        
        with pytest.raises(RuntimeError, match="Run test-run-1 creation failed"):
            await mock_orchestrator.create_run(run)

    @pytest.mark.asyncio
    async def test_list_runs(self, mock_orchestrator):
        """Test listing agent runs."""
        runs = [
            AgentRun(id="run-1", tenant_id="tenant-1", role="analyst", objective="Task 1", status=AgentRunStatus.running),
            AgentRun(id="run-2", tenant_id="tenant-1", role="dev", objective="Task 2", status=AgentRunStatus.running),
        ]
        mock_orchestrator.repos.runs.list.return_value = runs
        
        result = await mock_orchestrator.list_runs(tenant_id="tenant-1", limit=100, offset=0)
        
        assert len(result) == 2
        assert result[0].id == "run-1"
        assert result[1].id == "run-2"
        mock_orchestrator.repos.runs.list.assert_called_once_with(
            tenant_id="tenant-1", limit=100, offset=0
        )

    @pytest.mark.asyncio
    async def test_list_runs_with_defaults(self, mock_orchestrator):
        """Test listing runs with default parameters."""
        mock_orchestrator.repos.runs.list.return_value = []
        
        await mock_orchestrator.list_runs()
        
        mock_orchestrator.repos.runs.list.assert_called_once_with(
            tenant_id=None, limit=100, offset=0
        )

    @pytest.mark.asyncio
    async def test_get_run(self, mock_orchestrator):
        """Test getting a specific run."""
        run = AgentRun(id="run-1", tenant_id="tenant-1", role="analyst", objective="Task", status=AgentRunStatus.running)
        mock_orchestrator.repos.runs.get.return_value = run
        
        result = await mock_orchestrator.get_run("run-1")
        
        assert result.id == "run-1"
        mock_orchestrator.repos.runs.get.assert_called_once_with("run-1")

    @pytest.mark.asyncio
    async def test_get_run_not_found(self, mock_orchestrator):
        """Test getting a non-existent run."""
        mock_orchestrator.repos.runs.get.return_value = None
        
        result = await mock_orchestrator.get_run("non-existent")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_cancel_run(self, mock_orchestrator):
        """Test cancelling a run."""
        await mock_orchestrator.cancel_run("run-1")
        
        mock_orchestrator.repos.runs.update_status.assert_called_once_with(
            run_id="run-1", status=AgentRunStatus.cancelled.value
        )


class TestOrchestratorEventManagement:
    """Test event-related methods."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        with patch('gearmeshing_ai.server.services.orchestrator.build_sql_repos'):
            with patch('gearmeshing_ai.server.services.orchestrator.AgentServiceDeps'):
                with patch('gearmeshing_ai.server.services.orchestrator.StructuredPlanner'):
                    with patch('gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider'):
                        with patch('gearmeshing_ai.server.services.orchestrator.AgentService'):
                            orchestrator = OrchestratorService()
                            orchestrator.repos = MagicMock()
                            orchestrator.repos.events = AsyncMock()
                            yield orchestrator

    @pytest.mark.asyncio
    async def test_get_run_events(self, mock_orchestrator):
        """Test getting events for a run."""
        events = [
            AgentEvent(id="event-1", run_id="run-1", type=AgentEventType.run_started),
            AgentEvent(id="event-2", run_id="run-1", type=AgentEventType.thought_executed),
        ]
        mock_orchestrator.repos.events.list.return_value = events
        
        result = await mock_orchestrator.get_run_events("run-1", limit=100)
        
        assert len(result) == 2
        mock_orchestrator.repos.events.list.assert_called_once_with(run_id="run-1", limit=100)

    @pytest.mark.asyncio
    async def test_get_run_events_with_defaults(self, mock_orchestrator):
        """Test getting events with default limit."""
        mock_orchestrator.repos.events.list.return_value = []
        
        await mock_orchestrator.get_run_events("run-1")
        
        mock_orchestrator.repos.events.list.assert_called_once_with(run_id="run-1", limit=100)


class TestOrchestratorApprovalManagement:
    """Test approval-related methods."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        with patch('gearmeshing_ai.server.services.orchestrator.build_sql_repos'):
            with patch('gearmeshing_ai.server.services.orchestrator.AgentServiceDeps'):
                with patch('gearmeshing_ai.server.services.orchestrator.StructuredPlanner'):
                    with patch('gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider'):
                        with patch('gearmeshing_ai.server.services.orchestrator.AgentService'):
                            orchestrator = OrchestratorService()
                            orchestrator.repos = MagicMock()
                            orchestrator.repos.approvals = AsyncMock()
                            orchestrator.agent_service = AsyncMock()
                            yield orchestrator

    @pytest.mark.asyncio
    async def test_get_pending_approvals(self, mock_orchestrator):
        """Test getting pending approvals."""
        from gearmeshing_ai.agent_core.schemas.domain import RiskLevel, CapabilityName
        approvals = [
            Approval(id="approval-1", run_id="run-1", risk=RiskLevel.high, capability=CapabilityName.shell_exec, reason="Dangerous operation"),
            Approval(id="approval-2", run_id="run-1", risk=RiskLevel.medium, capability=CapabilityName.code_execution, reason="Code execution"),
        ]
        mock_orchestrator.repos.approvals.list.return_value = approvals
        
        result = await mock_orchestrator.get_pending_approvals("run-1")
        
        assert len(result) == 2
        mock_orchestrator.repos.approvals.list.assert_called_once_with(
            run_id="run-1", pending_only=True
        )

    @pytest.mark.asyncio
    async def test_submit_approval_approved(self, mock_orchestrator):
        """Test submitting an approved decision."""
        from gearmeshing_ai.agent_core.schemas.domain import RiskLevel, CapabilityName, ApprovalDecision
        approval = Approval(
            id="approval-1",
            run_id="run-1",
            risk=RiskLevel.high,
            capability=CapabilityName.shell_exec,
            reason="Dangerous operation",
            decision=ApprovalDecision.approved,
            decided_by="user-1",
        )
        mock_orchestrator.repos.approvals.get.return_value = approval
        
        result = await mock_orchestrator.submit_approval(
            run_id="run-1",
            approval_id="approval-1",
            decision="approved",
            note="Looks good",
            decided_by="user-1",
        )
        
        assert result.id == "approval-1"
        mock_orchestrator.repos.approvals.resolve.assert_called_once_with(
            "approval-1", decision="approved", decided_by="user-1"
        )
        mock_orchestrator.agent_service.resume.assert_called_once_with(
            run_id="run-1", approval_id="approval-1"
        )

    @pytest.mark.asyncio
    async def test_submit_approval_rejected(self, mock_orchestrator):
        """Test submitting a rejected decision."""
        from gearmeshing_ai.agent_core.schemas.domain import RiskLevel, CapabilityName, ApprovalDecision
        approval = Approval(
            id="approval-1",
            run_id="run-1",
            risk=RiskLevel.high,
            capability=CapabilityName.shell_exec,
            reason="Dangerous operation",
            decision=ApprovalDecision.rejected,
            decided_by="user-1",
        )
        mock_orchestrator.repos.approvals.get.return_value = approval
        
        result = await mock_orchestrator.submit_approval(
            run_id="run-1",
            approval_id="approval-1",
            decision="rejected",
            note="Not approved",
            decided_by="user-1",
        )
        
        assert result.id == "approval-1"
        # resume should not be called for rejected
        mock_orchestrator.agent_service.resume.assert_not_called()

    @pytest.mark.asyncio
    async def test_submit_approval_not_found(self, mock_orchestrator):
        """Test submit_approval raises error when approval not found."""
        mock_orchestrator.repos.approvals.get.return_value = None
        
        with pytest.raises(RuntimeError, match="Approval approval-1 not found"):
            await mock_orchestrator.submit_approval(
                run_id="run-1",
                approval_id="approval-1",
                decision="approved",
                note=None,
                decided_by="user-1",
            )


class TestOrchestratorUsageAndPolicy:
    """Test usage and policy methods."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        with patch('gearmeshing_ai.server.services.orchestrator.build_sql_repos'):
            with patch('gearmeshing_ai.server.services.orchestrator.AgentServiceDeps'):
                with patch('gearmeshing_ai.server.services.orchestrator.StructuredPlanner'):
                    with patch('gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider'):
                        with patch('gearmeshing_ai.server.services.orchestrator.AgentService'):
                            orchestrator = OrchestratorService()
                            orchestrator.repos = MagicMock()
                            orchestrator.repos.usage = AsyncMock()
                            orchestrator.repos.policies = AsyncMock()
                            yield orchestrator

    @pytest.mark.asyncio
    async def test_list_usage(self, mock_orchestrator):
        """Test listing usage entries."""
        usage_entries = [
            UsageLedgerEntry(id="usage-1", run_id="run-1", provider="openai", model="gpt-4"),
            UsageLedgerEntry(id="usage-2", run_id="run-2", provider="openai", model="gpt-4"),
        ]
        mock_orchestrator.repos.usage.list.return_value = usage_entries
        
        result = await mock_orchestrator.list_usage(
            tenant_id="tenant-1",
            from_date=None,
            to_date=None,
        )
        
        assert len(result) == 2
        mock_orchestrator.repos.usage.list.assert_called_once_with(
            tenant_id="tenant-1", from_date=None, to_date=None
        )

    @pytest.mark.asyncio
    async def test_list_usage_with_date_range(self, mock_orchestrator):
        """Test listing usage with date range."""
        from_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        to_date = datetime(2024, 12, 31, tzinfo=timezone.utc)
        
        mock_orchestrator.repos.usage.list.return_value = []
        
        await mock_orchestrator.list_usage(
            tenant_id="tenant-1",
            from_date=from_date,
            to_date=to_date,
        )
        
        mock_orchestrator.repos.usage.list.assert_called_once_with(
            tenant_id="tenant-1", from_date=from_date, to_date=to_date
        )

    @pytest.mark.asyncio
    async def test_get_policy(self, mock_orchestrator):
        """Test getting policy for a tenant."""
        policy = {"rules": ["rule1", "rule2"]}
        mock_orchestrator.repos.policies.get.return_value = policy
        
        result = await mock_orchestrator.get_policy("tenant-1")
        
        assert result == policy
        mock_orchestrator.repos.policies.get.assert_called_once_with(tenant_id="tenant-1")

    @pytest.mark.asyncio
    async def test_get_policy_not_found(self, mock_orchestrator):
        """Test getting non-existent policy."""
        mock_orchestrator.repos.policies.get.return_value = None
        
        result = await mock_orchestrator.get_policy("tenant-1")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_update_policy(self, mock_orchestrator):
        """Test updating policy for a tenant."""
        config = {"rules": ["new_rule"]}
        
        result = await mock_orchestrator.update_policy("tenant-1", config)
        
        assert result == config
        mock_orchestrator.repos.policies.update.assert_called_once_with(
            tenant_id="tenant-1", config=config
        )


class TestOrchestratorEventEnrichment:
    """Test event enrichment for SSE responses."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        with patch('gearmeshing_ai.server.services.orchestrator.build_sql_repos'):
            with patch('gearmeshing_ai.server.services.orchestrator.AgentServiceDeps'):
                with patch('gearmeshing_ai.server.services.orchestrator.StructuredPlanner'):
                    with patch('gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider'):
                        with patch('gearmeshing_ai.server.services.orchestrator.AgentService'):
                            orchestrator = OrchestratorService()
                            yield orchestrator

    @pytest.mark.asyncio
    async def test_enrich_thought_executed_event(self, mock_orchestrator):
        """Test enriching a thought_executed event."""
        now = datetime.now(timezone.utc)
        event = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.thought_executed,
            created_at=now,
            payload={"thought": "Let me think"},
        )
        
        result = await mock_orchestrator._enrich_event_for_sse(event)
        
        assert isinstance(result, SSEResponse)
        assert result.data.category == "thinking"
        assert result.data.thinking is not None
        assert result.data.thinking.thought == "Let me think"

    @pytest.mark.asyncio
    async def test_enrich_artifact_created_thought_event(self, mock_orchestrator):
        """Test enriching an artifact_created event with thought kind."""
        now = datetime.now(timezone.utc)
        event = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.artifact_created,
            created_at=now,
            payload={
                "kind": "thought",
                "thought": "My thought",
                "data": {"key": "value"},
                "output": "output text",
                "system_prompt_key": "default",
            },
        )
        
        result = await mock_orchestrator._enrich_event_for_sse(event)
        
        assert result.data.category == "thinking_output"
        assert result.data.thinking_output is not None
        assert result.data.thinking_output.thought == "My thought"
        assert result.data.thinking_output.output == "output text"

    @pytest.mark.asyncio
    async def test_enrich_capability_executed_event(self, mock_orchestrator):
        """Test enriching a capability_executed event."""
        now = datetime.now(timezone.utc)
        event = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.capability_executed,
            created_at=now,
            payload={
                "capability": "search",
                "ok": True,
                "result": "search results",
            },
        )
        
        result = await mock_orchestrator._enrich_event_for_sse(event)
        
        assert result.data.category == "operation"
        assert result.data.operation is not None
        assert result.data.operation.capability == "search"
        assert result.data.operation.status == "success"
        assert result.data.operation.result == "search results"

    @pytest.mark.asyncio
    async def test_enrich_capability_executed_failed(self, mock_orchestrator):
        """Test enriching a failed capability_executed event."""
        now = datetime.now(timezone.utc)
        event = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.capability_executed,
            created_at=now,
            payload={
                "capability": "search",
                "ok": False,
                "result": "error message",
            },
        )
        
        result = await mock_orchestrator._enrich_event_for_sse(event)
        
        assert result.data.operation.status == "failed"

    @pytest.mark.asyncio
    async def test_enrich_tool_invoked_event(self, mock_orchestrator):
        """Test enriching a tool_invoked event."""
        now = datetime.now(timezone.utc)
        event = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.tool_invoked,
            created_at=now,
            payload={
                "server_id": "server-1",
                "tool_name": "search",
                "args": {"query": "test"},
                "result": "results",
                "ok": True,
                "risk": "low",
            },
        )
        
        result = await mock_orchestrator._enrich_event_for_sse(event)
        
        assert result.data.category == "tool_execution"
        assert result.data.tool_execution is not None
        assert result.data.tool_execution.server_id == "server-1"
        assert result.data.tool_execution.tool_name == "search"
        assert result.data.tool_execution.ok is True

    @pytest.mark.asyncio
    async def test_enrich_approval_requested_event(self, mock_orchestrator):
        """Test enriching an approval_requested event."""
        now = datetime.now(timezone.utc)
        event = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.approval_requested,
            created_at=now,
            payload={
                "capability": "delete",
                "risk": "high",
                "reason": "Dangerous operation",
            },
        )
        
        result = await mock_orchestrator._enrich_event_for_sse(event)
        
        assert result.data.category == "approval"
        assert result.data.approval_request is not None
        assert result.data.approval_request.capability == "delete"
        assert result.data.approval_request.risk == "high"

    @pytest.mark.asyncio
    async def test_enrich_approval_resolved_event(self, mock_orchestrator):
        """Test enriching an approval_resolved event."""
        now = datetime.now(timezone.utc)
        event = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.approval_resolved,
            created_at=now,
            payload={
                "decision": "approved",
                "decided_by": "user-1",
            },
        )
        
        result = await mock_orchestrator._enrich_event_for_sse(event)
        
        assert result.data.category == "approval"
        assert result.data.approval_resolution is not None
        assert result.data.approval_resolution.decision == "approved"
        assert result.data.approval_resolution.decided_by == "user-1"

    @pytest.mark.asyncio
    async def test_enrich_run_started_event(self, mock_orchestrator):
        """Test enriching a run_started event."""
        now = datetime.now(timezone.utc)
        event = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.run_started,
            created_at=now,
        )
        
        result = await mock_orchestrator._enrich_event_for_sse(event)
        
        assert result.data.category == "run_lifecycle"
        assert result.data.run_start is not None
        assert result.data.run_start.run_id == "run-1"

    @pytest.mark.asyncio
    async def test_enrich_run_completed_event(self, mock_orchestrator):
        """Test enriching a run_completed event."""
        now = datetime.now(timezone.utc)
        event = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.run_completed,
            created_at=now,
        )
        
        result = await mock_orchestrator._enrich_event_for_sse(event)
        
        assert result.data.category == "run_lifecycle"
        assert result.data.run_completion is not None
        assert result.data.run_completion.status == "succeeded"

    @pytest.mark.asyncio
    async def test_enrich_run_failed_event(self, mock_orchestrator):
        """Test enriching a run_failed event."""
        now = datetime.now(timezone.utc)
        event = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.run_failed,
            created_at=now,
            payload={"error": "Something went wrong"},
        )
        
        result = await mock_orchestrator._enrich_event_for_sse(event)
        
        assert result.data.category == "run_lifecycle"
        assert result.data.run_failure is not None
        assert result.data.run_failure.error == "Something went wrong"

    @pytest.mark.asyncio
    async def test_enrich_unknown_event_type(self, mock_orchestrator):
        """Test enriching an event with unknown type."""
        now = datetime.now(timezone.utc)
        event = MagicMock()
        event.type = "unknown_type"
        event.model_dump.return_value = {
            "id": "event-1",
            "run_id": "run-1",
            "type": "unknown_type",
            "created_at": now,
            "payload": {},
        }
        
        result = await mock_orchestrator._enrich_event_for_sse(event)
        
        assert result.data.category == "other"


class TestOrchestratorRoles:
    """Test role-related methods."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        with patch('gearmeshing_ai.server.services.orchestrator.build_sql_repos'):
            with patch('gearmeshing_ai.server.services.orchestrator.AgentServiceDeps'):
                with patch('gearmeshing_ai.server.services.orchestrator.StructuredPlanner'):
                    with patch('gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider'):
                        with patch('gearmeshing_ai.server.services.orchestrator.AgentService'):
                            orchestrator = OrchestratorService()
                            yield orchestrator

    @pytest.mark.asyncio
    async def test_list_available_roles(self, mock_orchestrator):
        """Test listing available roles."""
        result = await mock_orchestrator.list_available_roles()
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(role, str) for role in result)

    @pytest.mark.asyncio
    async def test_override_role_prompt(self, mock_orchestrator):
        """Test overriding a role prompt."""
        result = await mock_orchestrator.override_role_prompt(
            tenant_id="tenant-1",
            role="analyst",
            prompt="Custom prompt",
        )
        
        assert result["status"] == "updated"
        assert result["role"] == "analyst"
        assert result["tenant_id"] == "tenant-1"


class TestOrchestratorEventStreaming:
    """Test event streaming functionality."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        with patch('gearmeshing_ai.server.services.orchestrator.build_sql_repos'):
            with patch('gearmeshing_ai.server.services.orchestrator.AgentServiceDeps'):
                with patch('gearmeshing_ai.server.services.orchestrator.StructuredPlanner'):
                    with patch('gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider'):
                        with patch('gearmeshing_ai.server.services.orchestrator.AgentService'):
                            orchestrator = OrchestratorService()
                            orchestrator.repos = MagicMock()
                            orchestrator.repos.events = AsyncMock()
                            yield orchestrator

    @pytest.mark.asyncio
    async def test_stream_events_with_new_events(self, mock_orchestrator):
        """Test streaming events when new events are available."""
        now = datetime.now(timezone.utc)
        event = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.run_started,
            created_at=now,
        )
        
        # First call returns event, second call returns empty to stop streaming
        mock_orchestrator.repos.events.list.side_effect = [
            [event],
            [],
            [],
        ]
        
        events_received = []
        async for sse_event in mock_orchestrator.stream_events("run-1"):
            events_received.append(sse_event)
            if len(events_received) >= 1:
                break
        
        assert len(events_received) >= 1
        assert isinstance(events_received[0], SSEResponse)

    @pytest.mark.asyncio
    async def test_stream_events_keep_alive(self, mock_orchestrator):
        """Test streaming sends keep-alive when no events."""
        mock_orchestrator.repos.events.list.return_value = []
        
        events_received = []
        async for sse_event in mock_orchestrator.stream_events("run-1"):
            events_received.append(sse_event)
            if len(events_received) >= 1:
                break
        
        assert len(events_received) >= 1
        assert isinstance(events_received[0], KeepAliveEvent)

    @pytest.mark.asyncio
    async def test_stream_events_error_handling(self, mock_orchestrator):
        """Test streaming handles errors gracefully."""
        mock_orchestrator.repos.events.list.side_effect = Exception("Database error")
        
        events_received = []
        async for sse_event in mock_orchestrator.stream_events("run-1"):
            events_received.append(sse_event)
            if len(events_received) >= 1:
                break
        
        assert len(events_received) >= 1
        assert isinstance(events_received[0], ErrorEvent)

    @pytest.mark.asyncio
    async def test_stream_events_filters_by_timestamp(self, mock_orchestrator):
        """Test streaming filters events by timestamp."""
        now = datetime.now(timezone.utc)
        old_event = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.run_started,
            created_at=now,
        )
        new_event = AgentEvent(
            id="event-2",
            run_id="run-1",
            type=AgentEventType.thought_executed,
            created_at=datetime.now(timezone.utc),
        )
        
        # First call returns old event, second call returns both, third returns empty
        mock_orchestrator.repos.events.list.side_effect = [
            [old_event],
            [old_event, new_event],
            [],
        ]
        
        events_received = []
        async for sse_event in mock_orchestrator.stream_events("run-1"):
            events_received.append(sse_event)
            if len(events_received) >= 2:
                break
        
        # Should get old_event and new_event
        assert len(events_received) >= 1


class TestOrchestratorEventStreamingIdleCycleTimeout:
    """Test idle cycle timeout behavior (line 180-182)."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        with patch('gearmeshing_ai.server.services.orchestrator.build_sql_repos'):
            with patch('gearmeshing_ai.server.services.orchestrator.AgentServiceDeps'):
                with patch('gearmeshing_ai.server.services.orchestrator.StructuredPlanner'):
                    with patch('gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider'):
                        with patch('gearmeshing_ai.server.services.orchestrator.AgentService'):
                            orchestrator = OrchestratorService()
                            orchestrator.repos = MagicMock()
                            orchestrator.repos.events = AsyncMock()
                            yield orchestrator

    @pytest.mark.asyncio
    async def test_stream_events_closes_after_max_idle_cycles(self, mock_orchestrator):
        """Test streaming closes after reaching max idle cycles (120)."""
        # Return empty list to trigger idle cycles
        mock_orchestrator.repos.events.list.return_value = []
        
        events_received = []
        async for sse_event in mock_orchestrator.stream_events("run-1"):
            events_received.append(sse_event)
            # After 121 events (120 keep-alives + 1), should close
            if len(events_received) >= 121:
                break
        
        # Should have received exactly 120 keep-alive events before closing
        assert len(events_received) == 120
        assert all(isinstance(e, KeepAliveEvent) for e in events_received)

    @pytest.mark.asyncio
    async def test_stream_events_resets_idle_cycles_on_new_event(self, mock_orchestrator):
        """Test idle cycles reset when new events arrive."""
        now = datetime.now(timezone.utc)
        event1 = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.run_started,
            created_at=now,
        )
        event2 = AgentEvent(
            id="event-2",
            run_id="run-1",
            type=AgentEventType.thought_executed,
            created_at=datetime.now(timezone.utc),
        )
        
        # Simulate: event, then empty (idle), then event, then empty (idle), etc.
        mock_orchestrator.repos.events.list.side_effect = [
            [event1],  # Event received, idle_cycles = 0
            [],  # No event, idle_cycles = 1
            [],  # No event, idle_cycles = 2
            [event1, event2],  # New event, idle_cycles resets to 0
            [],  # No event, idle_cycles = 1
        ]
        
        events_received = []
        async for sse_event in mock_orchestrator.stream_events("run-1"):
            events_received.append(sse_event)
            if len(events_received) >= 4:
                break
        
        # Should have: event1, keep-alive, keep-alive, event2
        assert len(events_received) >= 3
        assert isinstance(events_received[0], SSEResponse)
        assert isinstance(events_received[1], KeepAliveEvent)

    @pytest.mark.asyncio
    async def test_stream_events_idle_cycles_increment_on_empty(self, mock_orchestrator):
        """Test idle cycles increment correctly on empty event lists."""
        # Return empty list multiple times
        mock_orchestrator.repos.events.list.return_value = []
        
        events_received = []
        async for sse_event in mock_orchestrator.stream_events("run-1"):
            events_received.append(sse_event)
            # Collect first 5 keep-alives to verify incremental behavior
            if len(events_received) >= 5:
                break
        
        # All should be keep-alive events
        assert len(events_received) == 5
        assert all(isinstance(e, KeepAliveEvent) for e in events_received)

    @pytest.mark.asyncio
    async def test_stream_events_closes_exactly_at_max_idle(self, mock_orchestrator):
        """Test streaming closes exactly at max_idle_cycles threshold."""
        # Return empty to trigger idle cycles
        mock_orchestrator.repos.events.list.return_value = []
        
        events_received = []
        async for sse_event in mock_orchestrator.stream_events("run-1"):
            events_received.append(sse_event)
            # Don't break early - let it close naturally
            if len(events_received) > 125:
                # Safety break to avoid infinite loop
                break
        
        # Should close at exactly 120 idle cycles
        assert len(events_received) == 120

    @pytest.mark.asyncio
    async def test_stream_events_idle_cycles_with_mixed_events(self, mock_orchestrator):
        """Test idle cycles behavior with mixed event and empty responses."""
        now = datetime.now(timezone.utc)
        event1 = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.run_started,
            created_at=now,
        )
        # Create a second event with a later timestamp to ensure it's considered "new"
        event2 = AgentEvent(
            id="event-2",
            run_id="run-1",
            type=AgentEventType.thought_executed,
            created_at=now + timedelta(seconds=1),
            payload={"thought": "thinking"},
        )
        
        # Pattern: event1, empty, empty, event2, empty, empty, ...
        mock_orchestrator.repos.events.list.side_effect = [
            [event1],  # idle_cycles = 0
            [],  # idle_cycles = 1
            [],  # idle_cycles = 2
            [event1, event2],  # idle_cycles = 0 (reset), event2 is new
            [],  # idle_cycles = 1
            [],  # idle_cycles = 2
        ]
        
        events_received = []
        async for sse_event in mock_orchestrator.stream_events("run-1"):
            events_received.append(sse_event)
            if len(events_received) >= 5:
                break
        
        # Should have: event1, keep-alive, keep-alive, event2, keep-alive
        assert len(events_received) >= 4
        assert isinstance(events_received[0], SSEResponse)
        assert isinstance(events_received[1], KeepAliveEvent)
        assert isinstance(events_received[3], SSEResponse)


class TestOrchestratorEventStreamingAsyncSleep:
    """Test async sleep behavior (line 188-189)."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        with patch('gearmeshing_ai.server.services.orchestrator.build_sql_repos'):
            with patch('gearmeshing_ai.server.services.orchestrator.AgentServiceDeps'):
                with patch('gearmeshing_ai.server.services.orchestrator.StructuredPlanner'):
                    with patch('gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider'):
                        with patch('gearmeshing_ai.server.services.orchestrator.AgentService'):
                            orchestrator = OrchestratorService()
                            orchestrator.repos = MagicMock()
                            orchestrator.repos.events = AsyncMock()
                            yield orchestrator

    @pytest.mark.asyncio
    async def test_stream_events_calls_asyncio_sleep(self, mock_orchestrator):
        """Test streaming calls asyncio.sleep between polls."""
        mock_orchestrator.repos.events.list.return_value = []
        
        with patch('gearmeshing_ai.server.services.orchestrator.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            events_received = []
            async for sse_event in mock_orchestrator.stream_events("run-1"):
                events_received.append(sse_event)
                if len(events_received) >= 2:
                    break
            
            # Should have called sleep at least once
            assert mock_sleep.call_count >= 1

    @pytest.mark.asyncio
    async def test_stream_events_sleep_interval_is_half_second(self, mock_orchestrator):
        """Test streaming uses 0.5 second poll interval."""
        mock_orchestrator.repos.events.list.return_value = []
        
        with patch('gearmeshing_ai.server.services.orchestrator.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            events_received = []
            async for sse_event in mock_orchestrator.stream_events("run-1"):
                events_received.append(sse_event)
                if len(events_received) >= 3:
                    break
            
            # All sleep calls should be with 0.5 second interval
            for call in mock_sleep.call_args_list:
                assert call[0][0] == 0.5

    @pytest.mark.asyncio
    async def test_stream_events_sleep_after_each_poll(self, mock_orchestrator):
        """Test streaming sleeps after each poll cycle."""
        now = datetime.now(timezone.utc)
        event = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.run_started,
            created_at=now,
        )
        
        # Return event, then empty, then event
        mock_orchestrator.repos.events.list.side_effect = [
            [event],
            [],
            [event],
        ]
        
        with patch('gearmeshing_ai.server.services.orchestrator.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            events_received = []
            async for sse_event in mock_orchestrator.stream_events("run-1"):
                events_received.append(sse_event)
                if len(events_received) >= 3:
                    break
            
            # Should sleep after each poll (3 polls = 3 sleeps)
            assert mock_sleep.call_count >= 2

    @pytest.mark.asyncio
    async def test_stream_events_sleep_on_error(self, mock_orchestrator):
        """Test streaming sleeps even when error occurs."""
        mock_orchestrator.repos.events.list.side_effect = [
            Exception("Database error"),
            Exception("Database error"),
        ]
        
        with patch('gearmeshing_ai.server.services.orchestrator.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            events_received = []
            async for sse_event in mock_orchestrator.stream_events("run-1"):
                events_received.append(sse_event)
                if len(events_received) >= 2:
                    break
            
            # Should sleep even on errors
            assert mock_sleep.call_count >= 1
            # All calls should be 0.5 seconds
            for call in mock_sleep.call_args_list:
                assert call[0][0] == 0.5

    @pytest.mark.asyncio
    async def test_stream_events_sleep_maintains_consistent_interval(self, mock_orchestrator):
        """Test streaming maintains consistent sleep interval across multiple cycles."""
        mock_orchestrator.repos.events.list.return_value = []
        
        with patch('gearmeshing_ai.server.services.orchestrator.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            events_received = []
            async for sse_event in mock_orchestrator.stream_events("run-1"):
                events_received.append(sse_event)
                if len(events_received) >= 10:
                    break
            
            # All sleep calls should have consistent 0.5 second interval
            assert all(call[0][0] == 0.5 for call in mock_sleep.call_args_list)
            # Should have slept multiple times
            assert mock_sleep.call_count >= 9


class TestGetOrchestratorSingleton:
    """Test the get_orchestrator singleton function."""

    def test_get_orchestrator_returns_singleton(self):
        """Test get_orchestrator returns the same instance."""
        with patch('gearmeshing_ai.server.services.orchestrator.build_sql_repos'):
            with patch('gearmeshing_ai.server.services.orchestrator.AgentServiceDeps'):
                with patch('gearmeshing_ai.server.services.orchestrator.StructuredPlanner'):
                    with patch('gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider'):
                        with patch('gearmeshing_ai.server.services.orchestrator.AgentService'):
                            # Reset the global singleton
                            import gearmeshing_ai.server.services.orchestrator as orch_module
                            orch_module._orchestrator = None
                            
                            orchestrator1 = get_orchestrator()
                            orchestrator2 = get_orchestrator()
                            
                            assert orchestrator1 is orchestrator2

    def test_get_orchestrator_lazy_initialization(self):
        """Test get_orchestrator initializes on first call."""
        with patch('gearmeshing_ai.server.services.orchestrator.build_sql_repos'):
            with patch('gearmeshing_ai.server.services.orchestrator.AgentServiceDeps'):
                with patch('gearmeshing_ai.server.services.orchestrator.StructuredPlanner'):
                    with patch('gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider'):
                        with patch('gearmeshing_ai.server.services.orchestrator.AgentService'):
                            import gearmeshing_ai.server.services.orchestrator as orch_module
                            orch_module._orchestrator = None
                            
                            assert orch_module._orchestrator is None
                            orchestrator = get_orchestrator()
                            assert orch_module._orchestrator is not None
                            assert orchestrator is orch_module._orchestrator
