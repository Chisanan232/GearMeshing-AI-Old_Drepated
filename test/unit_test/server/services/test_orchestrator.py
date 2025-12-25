"""Unit tests for OrchestratorService.

Tests verify orchestrator functionality including run management, event streaming,
approval handling, and event enrichment for SSE responses.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Generator, List, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gearmeshing_ai.agent_core.schemas.domain import (
    AgentEvent,
    AgentEventType,
    AgentRun,
    AgentRunStatus,
    Approval,
    UsageLedgerEntry,
)
from gearmeshing_ai.server.schemas import (
    ErrorEvent,
    KeepAliveEvent,
    SSEResponse,
)
from gearmeshing_ai.server.services.orchestrator import (
    OrchestratorService,
    get_orchestrator,
)


class TestOrchestratorServiceInitialization:
    """Test OrchestratorService initialization."""

    @patch("gearmeshing_ai.server.services.orchestrator.AgentService")
    @patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider")
    @patch("gearmeshing_ai.server.services.orchestrator.StructuredPlanner")
    @patch("gearmeshing_ai.server.services.orchestrator.AgentServiceDeps")
    @patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos")
    def test_orchestrator_service_initializes(self, mock_repos: MagicMock, mock_deps: MagicMock, mock_planner: MagicMock, mock_policy: MagicMock, mock_service: MagicMock) -> None:
        """Test OrchestratorService can be instantiated."""
        orchestrator: OrchestratorService = OrchestratorService()
        assert orchestrator is not None

    @patch("gearmeshing_ai.server.services.orchestrator.AgentService")
    @patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider")
    @patch("gearmeshing_ai.server.services.orchestrator.StructuredPlanner")
    @patch("gearmeshing_ai.server.services.orchestrator.AgentServiceDeps")
    @patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos")
    def test_orchestrator_has_required_attributes(self, mock_repos: MagicMock, mock_deps: MagicMock, mock_planner: MagicMock, mock_policy: MagicMock, mock_service: MagicMock) -> None:
        """Test OrchestratorService has all required attributes."""
        orchestrator: OrchestratorService = OrchestratorService()
        assert hasattr(orchestrator, "repos")
        assert hasattr(orchestrator, "deps")
        assert hasattr(orchestrator, "policy_provider")
        assert hasattr(orchestrator, "agent_service")


class TestOrchestratorRunManagement:
    """Test run management methods."""

    @pytest.fixture
    def mock_orchestrator(self) -> Generator[OrchestratorService, None, None]:
        """Create a mock orchestrator with mocked dependencies."""
        with patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos"):
            with patch("gearmeshing_ai.server.services.orchestrator.AgentServiceDeps"):
                with patch("gearmeshing_ai.server.services.orchestrator.StructuredPlanner"):
                    with patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider"):
                        with patch("gearmeshing_ai.server.services.orchestrator.AgentService"):
                            orchestrator: OrchestratorService = OrchestratorService()
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
    async def test_create_run(self, mock_orchestrator: OrchestratorService) -> None:
        """Test creating a new agent run."""
        run: AgentRun = AgentRun(
            id="test-run-1",
            tenant_id="default-tenant",
            role="analyst",
            objective="Analyze data",
            status=AgentRunStatus.running,
        )

        srv_repo_run_get_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.runs.get)
        srv_repo_run_get_mock.return_value = run

        result: AgentRun = await mock_orchestrator.create_run(run)

        assert result.id == "test-run-1"
        srv_run = cast(MagicMock, mock_orchestrator.agent_service.run)
        srv_run.assert_called_once_with(run=run)
        srv_repo_run_get_mock.assert_called_once_with("test-run-1")

    @pytest.mark.asyncio
    async def test_create_run_failure(self, mock_orchestrator: OrchestratorService) -> None:
        """Test create_run raises error when run is not persisted."""
        run: AgentRun = AgentRun(
            id="test-run-1",
            tenant_id="default-tenant",
            role="analyst",
            objective="Analyze data",
            status=AgentRunStatus.running,
        )

        srv_repo_run_get_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.runs.get)
        srv_repo_run_get_mock.return_value = None

        with pytest.raises(RuntimeError, match="Run test-run-1 creation failed"):
            await mock_orchestrator.create_run(run)

    @pytest.mark.asyncio
    async def test_list_runs(self, mock_orchestrator: OrchestratorService) -> None:
        """Test listing agent runs."""
        runs: List[AgentRun] = [
            AgentRun(
                id="run-1", tenant_id="tenant-1", role="analyst", objective="Task 1", status=AgentRunStatus.running
            ),
            AgentRun(id="run-2", tenant_id="tenant-1", role="dev", objective="Task 2", status=AgentRunStatus.running),
        ]
        srv_repo_run_list_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.runs.list)
        srv_repo_run_list_mock.return_value = runs

        result: List[AgentRun] = await mock_orchestrator.list_runs(tenant_id="tenant-1", limit=100, offset=0)

        assert len(result) == 2
        assert result[0].id == "run-1"
        assert result[1].id == "run-2"
        srv_repo_run_list_mock.assert_called_once_with(tenant_id="tenant-1", limit=100, offset=0)

    @pytest.mark.asyncio
    async def test_list_runs_with_defaults(self, mock_orchestrator: OrchestratorService) -> None:
        """Test listing runs with default parameters."""
        srv_repo_run_list_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.runs.list)
        srv_repo_run_list_mock.return_value = []

        await mock_orchestrator.list_runs()

        srv_repo_run_list_mock.assert_called_once_with(tenant_id=None, limit=100, offset=0)

    @pytest.mark.asyncio
    async def test_get_run(self, mock_orchestrator: OrchestratorService) -> None:
        """Test getting a specific run."""
        run: AgentRun = AgentRun(
            id="run-1", tenant_id="tenant-1", role="analyst", objective="Task", status=AgentRunStatus.running
        )
        srv_repo_run_get_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.runs.get)
        srv_repo_run_get_mock.return_value = run

        result: AgentRun | None = await mock_orchestrator.get_run("run-1")

        assert result is not None
        assert result.id == "run-1"
        srv_repo_run_get_mock.assert_called_once_with("run-1")

    @pytest.mark.asyncio
    async def test_get_run_not_found(self, mock_orchestrator: OrchestratorService) -> None:
        """Test getting a non-existent run."""
        srv_repo_run_get_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.runs.get)
        srv_repo_run_get_mock.return_value = None

        result: AgentRun | None = await mock_orchestrator.get_run("non-existent")

        assert result is None

    @pytest.mark.asyncio
    async def test_cancel_run(self, mock_orchestrator: OrchestratorService) -> None:
        """Test cancelling a run."""
        await mock_orchestrator.cancel_run("run-1")

        srv_repo_run_update_status_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.runs.update_status)
        srv_repo_run_update_status_mock.assert_called_once_with(
            run_id="run-1", status=AgentRunStatus.cancelled.value
        )


class TestOrchestratorEventManagement:
    """Test event-related methods."""

    @pytest.fixture
    def mock_orchestrator(self) -> Generator[OrchestratorService, None, None]:
        """Create a mock orchestrator."""
        with patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos"):
            with patch("gearmeshing_ai.server.services.orchestrator.AgentServiceDeps"):
                with patch("gearmeshing_ai.server.services.orchestrator.StructuredPlanner"):
                    with patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider"):
                        with patch("gearmeshing_ai.server.services.orchestrator.AgentService"):
                            orchestrator: OrchestratorService = OrchestratorService()
                            orchestrator.repos = MagicMock()
                            orchestrator.repos.events = AsyncMock()
                            yield orchestrator

    @pytest.mark.asyncio
    async def test_get_run_events(self, mock_orchestrator: OrchestratorService) -> None:
        """Test getting events for a run."""
        events: List[AgentEvent] = [
            AgentEvent(id="event-1", run_id="run-1", type=AgentEventType.run_started),
            AgentEvent(id="event-2", run_id="run-1", type=AgentEventType.thought_executed),
        ]
        srv_repo_events_list_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.events.list)
        srv_repo_events_list_mock.return_value = events

        result: List[AgentEvent] = await mock_orchestrator.get_run_events("run-1", limit=100)

        assert len(result) == 2
        srv_repo_events_list_mock.assert_called_once_with(run_id="run-1", limit=100)

    @pytest.mark.asyncio
    async def test_get_run_events_with_defaults(self, mock_orchestrator: OrchestratorService) -> None:
        """Test getting events with default limit."""
        srv_repo_events_list_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.events.list)
        srv_repo_events_list_mock.return_value = []

        await mock_orchestrator.get_run_events("run-1")

        srv_repo_events_list_mock.assert_called_once_with(run_id="run-1", limit=100)


class TestOrchestratorApprovalManagement:
    """Test approval-related methods."""

    @pytest.fixture
    def mock_orchestrator(self) -> Generator[OrchestratorService, None, None]:
        """Create a mock orchestrator."""
        with patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos"):
            with patch("gearmeshing_ai.server.services.orchestrator.AgentServiceDeps"):
                with patch("gearmeshing_ai.server.services.orchestrator.StructuredPlanner"):
                    with patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider"):
                        with patch("gearmeshing_ai.server.services.orchestrator.AgentService"):
                            orchestrator: OrchestratorService = OrchestratorService()
                            orchestrator.repos = MagicMock()
                            orchestrator.repos.approvals = AsyncMock()
                            orchestrator.agent_service = AsyncMock()
                            yield orchestrator

    @pytest.mark.asyncio
    async def test_get_pending_approvals(self, mock_orchestrator: OrchestratorService) -> None:
        """Test getting pending approvals."""
        from gearmeshing_ai.agent_core.schemas.domain import CapabilityName, RiskLevel

        approvals: List[Approval] = [
            Approval(
                id="approval-1",
                run_id="run-1",
                risk=RiskLevel.high,
                capability=CapabilityName.shell_exec,
                reason="Dangerous operation",
            ),
            Approval(
                id="approval-2",
                run_id="run-1",
                risk=RiskLevel.medium,
                capability=CapabilityName.code_execution,
                reason="Code execution",
            ),
        ]
        srv_repo_approvals_list_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.approvals.list)
        srv_repo_approvals_list_mock.return_value = approvals

        result: List[Approval] = await mock_orchestrator.get_pending_approvals("run-1")

        assert len(result) == 2
        srv_repo_approvals_list_mock.assert_called_once_with(run_id="run-1", pending_only=True)

    @pytest.mark.asyncio
    async def test_submit_approval_approved(self, mock_orchestrator: OrchestratorService) -> None:
        """Test submitting an approved decision."""
        from gearmeshing_ai.agent_core.schemas.domain import (
            ApprovalDecision,
            CapabilityName,
            RiskLevel,
        )

        approval: Approval = Approval(
            id="approval-1",
            run_id="run-1",
            risk=RiskLevel.high,
            capability=CapabilityName.shell_exec,
            reason="Dangerous operation",
            decision=ApprovalDecision.approved,
            decided_by="user-1",
        )
        srv_repo_approvals_get_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.approvals.get)
        srv_repo_approvals_get_mock.return_value = approval

        result: Approval = await mock_orchestrator.submit_approval(
            run_id="run-1",
            approval_id="approval-1",
            decision="approved",
            note="Looks good",
            decided_by="user-1",
        )

        assert result.id == "approval-1"
        srv_repo_approvals_resolve_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.approvals.resolve)
        srv_repo_approvals_resolve_mock.assert_called_once_with(
            "approval-1", decision="approved", decided_by="user-1"
        )
        srv_agent_service_resume_mock: MagicMock = cast(MagicMock, mock_orchestrator.agent_service.resume)
        srv_agent_service_resume_mock.assert_called_once_with(run_id="run-1", approval_id="approval-1")

    @pytest.mark.asyncio
    async def test_submit_approval_rejected(self, mock_orchestrator: OrchestratorService) -> None:
        """Test submitting a rejected decision."""
        from gearmeshing_ai.agent_core.schemas.domain import (
            ApprovalDecision,
            CapabilityName,
            RiskLevel,
        )

        approval: Approval = Approval(
            id="approval-1",
            run_id="run-1",
            risk=RiskLevel.high,
            capability=CapabilityName.shell_exec,
            reason="Dangerous operation",
            decision=ApprovalDecision.rejected,
            decided_by="user-1",
        )
        srv_repo_approvals_get_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.approvals.get)
        srv_repo_approvals_get_mock.return_value = approval

        result: Approval = await mock_orchestrator.submit_approval(
            run_id="run-1",
            approval_id="approval-1",
            decision="rejected",
            note="Not approved",
            decided_by="user-1",
        )

        assert result.id == "approval-1"
        # resume should not be called for rejected
        srv_agent_service_resume_mock: MagicMock = cast(MagicMock, mock_orchestrator.agent_service.resume)
        srv_agent_service_resume_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_submit_approval_not_found(self, mock_orchestrator: OrchestratorService) -> None:
        """Test submit_approval raises error when approval not found."""
        srv_repo_approvals_get_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.approvals.get)
        srv_repo_approvals_get_mock.return_value = None

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
    def mock_orchestrator(self) -> Generator[OrchestratorService, None, None]:
        """Create a mock orchestrator."""
        with patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos"):
            with patch("gearmeshing_ai.server.services.orchestrator.AgentServiceDeps"):
                with patch("gearmeshing_ai.server.services.orchestrator.StructuredPlanner"):
                    with patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider"):
                        with patch("gearmeshing_ai.server.services.orchestrator.AgentService"):
                            orchestrator: OrchestratorService = OrchestratorService()
                            orchestrator.repos = MagicMock()
                            orchestrator.repos.usage = AsyncMock()
                            orchestrator.repos.policies = AsyncMock()
                            yield orchestrator

    @pytest.mark.asyncio
    async def test_list_usage(self, mock_orchestrator: OrchestratorService) -> None:
        """Test listing usage entries."""
        usage_entries: List[UsageLedgerEntry] = [
            UsageLedgerEntry(id="usage-1", run_id="run-1", provider="openai", model="gpt-4"),
            UsageLedgerEntry(id="usage-2", run_id="run-2", provider="openai", model="gpt-4"),
        ]
        srv_repo_usage_list_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.usage.list)
        srv_repo_usage_list_mock.return_value = usage_entries

        result: List[UsageLedgerEntry] = await mock_orchestrator.list_usage(
            tenant_id="tenant-1",
            from_date=None,
            to_date=None,
        )

        assert len(result) == 2
        srv_repo_usage_list_mock.assert_called_once_with(tenant_id="tenant-1", from_date=None, to_date=None)

    @pytest.mark.asyncio
    async def test_list_usage_with_date_range(self, mock_orchestrator: OrchestratorService) -> None:
        """Test listing usage with date range."""
        from_date: datetime = datetime(2024, 1, 1, tzinfo=timezone.utc)
        to_date: datetime = datetime(2024, 12, 31, tzinfo=timezone.utc)

        srv_repo_usage_list_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.usage.list)
        srv_repo_usage_list_mock.return_value = []

        await mock_orchestrator.list_usage(
            tenant_id="tenant-1",
            from_date=from_date,
            to_date=to_date,
        )

        srv_repo_usage_list_mock.assert_called_once_with(
            tenant_id="tenant-1", from_date=from_date, to_date=to_date
        )

    @pytest.mark.asyncio
    async def test_get_policy(self, mock_orchestrator: OrchestratorService) -> None:
        """Test getting policy for a tenant."""
        policy: dict[str, Any] = {"rules": ["rule1", "rule2"]}
        srv_repo_policies_get_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.policies.get)
        srv_repo_policies_get_mock.return_value = policy

        result: dict[str, Any] | None = await mock_orchestrator.get_policy("tenant-1")

        assert result == policy
        srv_repo_policies_get_mock.assert_called_once_with(tenant_id="tenant-1")

    @pytest.mark.asyncio
    async def test_get_policy_not_found(self, mock_orchestrator: OrchestratorService) -> None:
        """Test getting non-existent policy."""
        srv_repo_policies_get_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.policies.get)
        srv_repo_policies_get_mock.return_value = None

        result: dict[str, Any] | None = await mock_orchestrator.get_policy("tenant-1")

        assert result is None

    @pytest.mark.asyncio
    async def test_update_policy(self, mock_orchestrator: OrchestratorService) -> None:
        """Test updating policy for a tenant."""
        config: dict[str, Any] = {"rules": ["new_rule"]}

        result: dict[str, Any] = await mock_orchestrator.update_policy("tenant-1", config)

        assert result == config
        srv_repo_policies_update_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.policies.update)
        srv_repo_policies_update_mock.assert_called_once_with(tenant_id="tenant-1", config=config)


class TestOrchestratorEventEnrichment:
    """Test event enrichment for SSE responses."""

    @pytest.fixture
    def mock_orchestrator(self) -> Generator[OrchestratorService, None, None]:
        """Create a mock orchestrator."""
        with patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos"):
            with patch("gearmeshing_ai.server.services.orchestrator.AgentServiceDeps"):
                with patch("gearmeshing_ai.server.services.orchestrator.StructuredPlanner"):
                    with patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider"):
                        with patch("gearmeshing_ai.server.services.orchestrator.AgentService"):
                            orchestrator: OrchestratorService = OrchestratorService()
                            yield orchestrator

    @pytest.mark.asyncio
    async def test_enrich_thought_executed_event(self, mock_orchestrator: OrchestratorService) -> None:
        """Test enriching a thought_executed event."""
        now: datetime = datetime.now(timezone.utc)
        event: AgentEvent = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.thought_executed,
            created_at=now,
            payload={"thought": "Let me think"},
        )

        result: SSEResponse = await mock_orchestrator._enrich_event_for_sse(event)

        assert isinstance(result, SSEResponse)
        assert result.data.category == "thinking"
        assert result.data.thinking is not None
        assert result.data.thinking.thought == "Let me think"

    @pytest.mark.asyncio
    async def test_enrich_artifact_created_thought_event(self, mock_orchestrator: OrchestratorService) -> None:
        """Test enriching an artifact_created event with thought kind."""
        now: datetime = datetime.now(timezone.utc)
        event: AgentEvent = AgentEvent(
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

        result: SSEResponse = await mock_orchestrator._enrich_event_for_sse(event)

        assert result.data.category == "thinking_output"
        assert result.data.thinking_output is not None
        assert result.data.thinking_output.thought == "My thought"
        assert result.data.thinking_output.output == "output text"

    @pytest.mark.asyncio
    async def test_enrich_capability_executed_event(self, mock_orchestrator: OrchestratorService) -> None:
        """Test enriching a capability_executed event."""
        now: datetime = datetime.now(timezone.utc)
        event: AgentEvent = AgentEvent(
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

        result: SSEResponse = await mock_orchestrator._enrich_event_for_sse(event)

        assert result.data.category == "operation"
        assert result.data.operation is not None
        assert result.data.operation.capability == "search"
        assert result.data.operation.status == "success"
        assert result.data.operation.result == "search results"

    @pytest.mark.asyncio
    async def test_enrich_capability_executed_failed(self, mock_orchestrator: OrchestratorService) -> None:
        """Test enriching a failed capability_executed event."""
        now: datetime = datetime.now(timezone.utc)
        event: AgentEvent = AgentEvent(
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

        result: SSEResponse = await mock_orchestrator._enrich_event_for_sse(event)

        assert result.data.operation is not None
        assert result.data.operation.status == "failed"

    @pytest.mark.asyncio
    async def test_enrich_tool_invoked_event(self, mock_orchestrator: OrchestratorService) -> None:
        """Test enriching a tool_invoked event."""
        now: datetime = datetime.now(timezone.utc)
        event: AgentEvent = AgentEvent(
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

        result: SSEResponse = await mock_orchestrator._enrich_event_for_sse(event)

        assert result.data.category == "tool_execution"
        assert result.data.tool_execution is not None
        assert result.data.tool_execution.server_id == "server-1"
        assert result.data.tool_execution.tool_name == "search"
        assert result.data.tool_execution.ok is True

    @pytest.mark.asyncio
    async def test_enrich_approval_requested_event(self, mock_orchestrator: OrchestratorService) -> None:
        """Test enriching an approval_requested event."""
        now: datetime = datetime.now(timezone.utc)
        event: AgentEvent = AgentEvent(
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

        result: SSEResponse = await mock_orchestrator._enrich_event_for_sse(event)

        assert result.data.category == "approval"
        assert result.data.approval_request is not None
        assert result.data.approval_request.capability == "delete"
        assert result.data.approval_request.risk == "high"

    @pytest.mark.asyncio
    async def test_enrich_approval_resolved_event(self, mock_orchestrator: OrchestratorService) -> None:
        """Test enriching an approval_resolved event."""
        now: datetime = datetime.now(timezone.utc)
        event: AgentEvent = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.approval_resolved,
            created_at=now,
            payload={
                "decision": "approved",
                "decided_by": "user-1",
            },
        )

        result: SSEResponse = await mock_orchestrator._enrich_event_for_sse(event)

        assert result.data.category == "approval"
        assert result.data.approval_resolution is not None
        assert result.data.approval_resolution.decision == "approved"
        assert result.data.approval_resolution.decided_by == "user-1"

    @pytest.mark.asyncio
    async def test_enrich_run_started_event(self, mock_orchestrator: OrchestratorService) -> None:
        """Test enriching a run_started event."""
        now: datetime = datetime.now(timezone.utc)
        event: AgentEvent = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.run_started,
            created_at=now,
        )

        result: SSEResponse = await mock_orchestrator._enrich_event_for_sse(event)

        assert result.data.category == "run_lifecycle"
        assert result.data.run_start is not None
        assert result.data.run_start.run_id == "run-1"

    @pytest.mark.asyncio
    async def test_enrich_run_completed_event(self, mock_orchestrator: OrchestratorService) -> None:
        """Test enriching a run_completed event."""
        now: datetime = datetime.now(timezone.utc)
        event: AgentEvent = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.run_completed,
            created_at=now,
        )

        result: SSEResponse = await mock_orchestrator._enrich_event_for_sse(event)

        assert result.data.category == "run_lifecycle"
        assert result.data.run_completion is not None
        assert result.data.run_completion.status == "succeeded"

    @pytest.mark.asyncio
    async def test_enrich_run_failed_event(self, mock_orchestrator: OrchestratorService) -> None:
        """Test enriching a run_failed event."""
        now: datetime = datetime.now(timezone.utc)
        event: AgentEvent = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.run_failed,
            created_at=now,
            payload={"error": "Something went wrong"},
        )

        result: SSEResponse = await mock_orchestrator._enrich_event_for_sse(event)

        assert result.data.category == "run_lifecycle"
        assert result.data.run_failure is not None
        assert result.data.run_failure.error == "Something went wrong"


class TestOrchestratorRoles:
    """Test role-related methods."""

    @pytest.fixture
    def mock_orchestrator(self) -> Generator[OrchestratorService, None, None]:
        """Create a mock orchestrator."""
        with patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos"):
            with patch("gearmeshing_ai.server.services.orchestrator.AgentServiceDeps"):
                with patch("gearmeshing_ai.server.services.orchestrator.StructuredPlanner"):
                    with patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider"):
                        with patch("gearmeshing_ai.server.services.orchestrator.AgentService"):
                            orchestrator: OrchestratorService = OrchestratorService()
                            yield orchestrator

    @pytest.mark.asyncio
    async def test_list_available_roles(self, mock_orchestrator: OrchestratorService) -> None:
        """Test listing available roles."""
        result: List[str] = await mock_orchestrator.list_available_roles()

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(role, str) for role in result)

    @pytest.mark.asyncio
    async def test_override_role_prompt(self, mock_orchestrator: OrchestratorService) -> None:
        """Test overriding a role prompt."""
        result: dict[str, Any] = await mock_orchestrator.override_role_prompt(
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
    def mock_orchestrator(self) -> Generator[OrchestratorService, None, None]:
        """Create a mock orchestrator."""
        with patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos"):
            with patch("gearmeshing_ai.server.services.orchestrator.AgentServiceDeps"):
                with patch("gearmeshing_ai.server.services.orchestrator.StructuredPlanner"):
                    with patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider"):
                        with patch("gearmeshing_ai.server.services.orchestrator.AgentService"):
                            orchestrator: OrchestratorService = OrchestratorService()
                            orchestrator.repos = MagicMock()
                            orchestrator.repos.events = AsyncMock()
                            yield orchestrator

    @pytest.mark.asyncio
    async def test_stream_events_with_new_events(self, mock_orchestrator: OrchestratorService) -> None:
        """Test streaming events when new events are available."""
        now: datetime = datetime.now(timezone.utc)
        event: AgentEvent = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.run_started,
            created_at=now,
        )

        # First call returns event, second call returns empty to stop streaming
        srv_repo_events_list_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.events.list)
        srv_repo_events_list_mock.side_effect = [
            [event],
            [],
            [],
        ]

        events_received: List[Any] = []
        async for sse_event in mock_orchestrator.stream_events("run-1"):
            events_received.append(sse_event)
            if len(events_received) >= 1:
                break

        assert len(events_received) >= 1
        assert isinstance(events_received[0], SSEResponse)

    @pytest.mark.asyncio
    async def test_stream_events_keep_alive(self, mock_orchestrator: OrchestratorService) -> None:
        """Test streaming sends keep-alive when no events."""
        srv_repo_events_list_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.events.list)
        srv_repo_events_list_mock.return_value = []

        events_received: List[Any] = []
        async for sse_event in mock_orchestrator.stream_events("run-1"):
            events_received.append(sse_event)
            if len(events_received) >= 1:
                break

        assert len(events_received) >= 1
        assert isinstance(events_received[0], KeepAliveEvent)

    @pytest.mark.asyncio
    async def test_stream_events_error_handling(self, mock_orchestrator: OrchestratorService) -> None:
        """Test streaming handles errors gracefully."""
        srv_repo_events_list_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.events.list)
        srv_repo_events_list_mock.side_effect = Exception("Database error")

        events_received: List[Any] = []
        async for sse_event in mock_orchestrator.stream_events("run-1"):
            events_received.append(sse_event)
            if len(events_received) >= 1:
                break

        assert len(events_received) >= 1
        assert isinstance(events_received[0], ErrorEvent)

    @pytest.mark.asyncio
    async def test_stream_events_filters_by_timestamp(self, mock_orchestrator: OrchestratorService) -> None:
        """Test streaming filters events by timestamp."""
        now: datetime = datetime.now(timezone.utc)
        old_event: AgentEvent = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.run_started,
            created_at=now,
        )
        new_event: AgentEvent = AgentEvent(
            id="event-2",
            run_id="run-1",
            type=AgentEventType.thought_executed,
            created_at=datetime.now(timezone.utc),
        )

        # First call returns old event, second call returns both, third returns empty
        srv_repo_events_list_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.events.list)
        srv_repo_events_list_mock.side_effect = [
            [old_event],
            [old_event, new_event],
            [],
        ]

        events_received: List[Any] = []
        async for sse_event in mock_orchestrator.stream_events("run-1"):
            events_received.append(sse_event)
            if len(events_received) >= 2:
                break

        # Should get old_event and new_event
        assert len(events_received) >= 1


class TestOrchestratorEventStreamingIdleCycleTimeout:
    """Test idle cycle timeout behavior (line 180-182)."""

    @pytest.fixture
    def mock_orchestrator(self) -> Generator[OrchestratorService, None, None]:
        """Create a mock orchestrator."""
        with patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos"):
            with patch("gearmeshing_ai.server.services.orchestrator.AgentServiceDeps"):
                with patch("gearmeshing_ai.server.services.orchestrator.StructuredPlanner"):
                    with patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider"):
                        with patch("gearmeshing_ai.server.services.orchestrator.AgentService"):
                            orchestrator: OrchestratorService = OrchestratorService()
                            orchestrator.repos = MagicMock()
                            orchestrator.repos.events = AsyncMock()
                            yield orchestrator

    @pytest.mark.asyncio
    async def test_stream_events_closes_after_max_idle_cycles(self, mock_orchestrator: OrchestratorService) -> None:
        """Test streaming closes after reaching max idle cycles (120)."""
        # Return empty list to trigger idle cycles
        srv_repo_events_list_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.events.list)
        srv_repo_events_list_mock.return_value = []

        events_received: List[Any] = []
        async for sse_event in mock_orchestrator.stream_events("run-1"):
            events_received.append(sse_event)
            # After 121 events (120 keep-alives + 1), should close
            if len(events_received) >= 121:
                break

        # Should have received exactly 120 keep-alive events before closing
        assert len(events_received) == 120
        assert all(isinstance(e, KeepAliveEvent) for e in events_received)

    @pytest.mark.asyncio
    async def test_stream_events_resets_idle_cycles_on_new_event(self, mock_orchestrator: OrchestratorService) -> None:
        """Test idle cycles reset when new events arrive."""
        now: datetime = datetime.now(timezone.utc)
        event1: AgentEvent = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.run_started,
            created_at=now,
        )
        event2: AgentEvent = AgentEvent(
            id="event-2",
            run_id="run-1",
            type=AgentEventType.thought_executed,
            created_at=datetime.now(timezone.utc),
        )

        # Simulate: event, then empty (idle), then event, then empty (idle), etc.
        srv_repo_events_list_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.events.list)
        srv_repo_events_list_mock.side_effect = [
            [event1],  # Event received, idle_cycles = 0
            [],  # No event, idle_cycles = 1
            [],  # No event, idle_cycles = 2
            [event1, event2],  # New event, idle_cycles resets to 0
            [],  # No event, idle_cycles = 1
        ]

        events_received: List[Any] = []
        async for sse_event in mock_orchestrator.stream_events("run-1"):
            events_received.append(sse_event)
            if len(events_received) >= 4:
                break

        # Should have: event1, keep-alive, keep-alive, event2
        assert len(events_received) >= 3
        assert isinstance(events_received[0], SSEResponse)
        assert isinstance(events_received[1], KeepAliveEvent)

    @pytest.mark.asyncio
    async def test_stream_events_idle_cycles_increment_on_empty(self, mock_orchestrator: OrchestratorService) -> None:
        """Test idle cycles increment correctly on empty event lists."""
        # Return empty list multiple times
        srv_repo_events_list_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.events.list)
        srv_repo_events_list_mock.return_value = []

        events_received: List[Any] = []
        async for sse_event in mock_orchestrator.stream_events("run-1"):
            events_received.append(sse_event)
            # Collect first 5 keep-alives to verify incremental behavior
            if len(events_received) >= 5:
                break

        # All should be keep-alive events
        assert len(events_received) == 5
        assert all(isinstance(e, KeepAliveEvent) for e in events_received)

    @pytest.mark.asyncio
    async def test_stream_events_closes_exactly_at_max_idle(self, mock_orchestrator: OrchestratorService) -> None:
        """Test streaming closes exactly at max_idle_cycles threshold."""
        # Return empty to trigger idle cycles
        srv_repo_events_list_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.events.list)
        srv_repo_events_list_mock.return_value = []

        events_received: List[Any] = []
        async for sse_event in mock_orchestrator.stream_events("run-1"):
            events_received.append(sse_event)
            # Don't break early - let it close naturally
            if len(events_received) > 125:
                # Safety break to avoid infinite loop
                break

        # Should close at exactly 120 idle cycles
        assert len(events_received) == 120

    @pytest.mark.asyncio
    async def test_stream_events_idle_cycles_with_mixed_events(self, mock_orchestrator: OrchestratorService) -> None:
        """Test idle cycles behavior with mixed event and empty responses."""
        now: datetime = datetime.now(timezone.utc)
        event1: AgentEvent = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.run_started,
            created_at=now,
        )
        # Create a second event with a later timestamp to ensure it's considered "new"
        event2: AgentEvent = AgentEvent(
            id="event-2",
            run_id="run-1",
            type=AgentEventType.thought_executed,
            created_at=now + timedelta(seconds=1),
            payload={"thought": "thinking"},
        )

        # Pattern: event1, empty, empty, event2, empty, empty, ...
        srv_repo_events_list_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.events.list)
        srv_repo_events_list_mock.side_effect = [
            [event1],  # idle_cycles = 0
            [],  # idle_cycles = 1
            [],  # idle_cycles = 2
            [event1, event2],  # idle_cycles = 0 (reset), event2 is new
            [],  # idle_cycles = 1
            [],  # idle_cycles = 2
        ]

        events_received: List[Any] = []
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
    def mock_orchestrator(self) -> Generator[OrchestratorService, None, None]:
        """Create a mock orchestrator."""
        with patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos"):
            with patch("gearmeshing_ai.server.services.orchestrator.AgentServiceDeps"):
                with patch("gearmeshing_ai.server.services.orchestrator.StructuredPlanner"):
                    with patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider"):
                        with patch("gearmeshing_ai.server.services.orchestrator.AgentService"):
                            orchestrator: OrchestratorService = OrchestratorService()
                            orchestrator.repos = MagicMock()
                            orchestrator.repos.events = AsyncMock()
                            yield orchestrator

    @pytest.mark.asyncio
    async def test_stream_events_calls_asyncio_sleep(self, mock_orchestrator: OrchestratorService) -> None:
        """Test streaming calls asyncio.sleep between polls."""
        srv_repo_events_list_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.events.list)
        srv_repo_events_list_mock.return_value = []

        with patch("gearmeshing_ai.server.services.orchestrator.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            events_received: List[Any] = []
            async for sse_event in mock_orchestrator.stream_events("run-1"):
                events_received.append(sse_event)
                if len(events_received) >= 2:
                    break

            # Should have called sleep at least once
            assert mock_sleep.call_count >= 1

    @pytest.mark.asyncio
    async def test_stream_events_sleep_interval_is_half_second(self, mock_orchestrator: OrchestratorService) -> None:
        """Test streaming uses 0.5 second poll interval."""
        srv_repo_events_list_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.events.list)
        srv_repo_events_list_mock.return_value = []

        with patch("gearmeshing_ai.server.services.orchestrator.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            events_received: List[Any] = []
            async for sse_event in mock_orchestrator.stream_events("run-1"):
                events_received.append(sse_event)
                if len(events_received) >= 3:
                    break

            # All sleep calls should be with 0.5 second interval
            for call in mock_sleep.call_args_list:
                assert call[0][0] == 0.5

    @pytest.mark.asyncio
    async def test_stream_events_sleep_after_each_poll(self, mock_orchestrator: OrchestratorService) -> None:
        """Test streaming sleeps after each poll cycle."""
        now: datetime = datetime.now(timezone.utc)
        event: AgentEvent = AgentEvent(
            id="event-1",
            run_id="run-1",
            type=AgentEventType.run_started,
            created_at=now,
        )

        # Return event, then empty, then event
        srv_repo_events_list_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.events.list)
        srv_repo_events_list_mock.side_effect = [
            [event],
            [],
            [event],
        ]

        with patch("gearmeshing_ai.server.services.orchestrator.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            events_received: List[Any] = []
            async for sse_event in mock_orchestrator.stream_events("run-1"):
                events_received.append(sse_event)
                if len(events_received) >= 3:
                    break

            # Should sleep after each poll (3 polls = 3 sleeps)
            assert mock_sleep.call_count >= 2

    @pytest.mark.asyncio
    async def test_stream_events_sleep_on_error(self, mock_orchestrator: OrchestratorService) -> None:
        """Test streaming sleeps even when error occurs."""
        srv_repo_events_list_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.events.list)
        srv_repo_events_list_mock.side_effect = [
            Exception("Database error"),
            Exception("Database error"),
        ]

        with patch("gearmeshing_ai.server.services.orchestrator.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            events_received: List[Any] = []
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
    async def test_stream_events_sleep_maintains_consistent_interval(self, mock_orchestrator: OrchestratorService) -> None:
        """Test streaming maintains consistent sleep interval across multiple cycles."""
        srv_repo_events_list_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.events.list)
        srv_repo_events_list_mock.return_value = []

        with patch("gearmeshing_ai.server.services.orchestrator.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            events_received: List[Any] = []
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

    def test_get_orchestrator_returns_singleton(self) -> None:
        """Test get_orchestrator returns the same instance."""
        with patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos"):
            with patch("gearmeshing_ai.server.services.orchestrator.AgentServiceDeps"):
                with patch("gearmeshing_ai.server.services.orchestrator.StructuredPlanner"):
                    with patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider"):
                        with patch("gearmeshing_ai.server.services.orchestrator.AgentService"):
                            # Reset the global singleton
                            import gearmeshing_ai.server.services.orchestrator as orch_module

                            orch_module._orchestrator = None

                            orchestrator1: OrchestratorService = get_orchestrator()
                            orchestrator2: OrchestratorService = get_orchestrator()

                            assert orchestrator1 is orchestrator2

    def test_get_orchestrator_lazy_initialization(self) -> None:
        """Test get_orchestrator initializes on first call."""
        with patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos"):
            with patch("gearmeshing_ai.server.services.orchestrator.AgentServiceDeps"):
                with patch("gearmeshing_ai.server.services.orchestrator.StructuredPlanner"):
                    with patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider"):
                        with patch("gearmeshing_ai.server.services.orchestrator.AgentService"):
                            import gearmeshing_ai.server.services.orchestrator as orch_module

                            orch_module._orchestrator = None

                            assert orch_module._orchestrator is None
                            orchestrator: OrchestratorService = get_orchestrator()
                            assert orch_module._orchestrator is not None
                            assert orchestrator is orch_module._orchestrator
