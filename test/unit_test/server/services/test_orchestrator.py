"""Unit tests for OrchestratorService.

Tests verify orchestrator functionality including run management, event streaming,
approval handling, and event enrichment for SSE responses.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gearmeshing_ai.agent_core.policy.models import PolicyConfig
from gearmeshing_ai.agent_core.repos.interfaces import EventRepository
from gearmeshing_ai.agent_core.schemas.domain import (
    AgentEvent,
    AgentEventType,
    AgentRun,
    AgentRunStatus,
    Approval,
    UsageLedgerEntry,
)
from gearmeshing_ai.server.schemas import (
    KeepAliveEvent,
    SSEResponse,
)
from gearmeshing_ai.server.services.orchestrator import (
    BroadcastingEventRepository,
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
    def test_orchestrator_service_initializes(
        self,
        mock_repos: MagicMock,
        mock_deps: MagicMock,
        mock_planner: MagicMock,
        mock_policy: MagicMock,
        mock_service: MagicMock,
    ) -> None:
        """Test OrchestratorService can be instantiated."""
        orchestrator: OrchestratorService = OrchestratorService()
        assert orchestrator is not None

    @patch("gearmeshing_ai.server.services.orchestrator.AgentService")
    @patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider")
    @patch("gearmeshing_ai.server.services.orchestrator.StructuredPlanner")
    @patch("gearmeshing_ai.server.services.orchestrator.AgentServiceDeps")
    @patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos")
    def test_orchestrator_has_required_attributes(
        self,
        mock_repos: MagicMock,
        mock_deps: MagicMock,
        mock_planner: MagicMock,
        mock_policy: MagicMock,
        mock_service: MagicMock,
    ) -> None:
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
        # AgentService.run is NOT called in create_run anymore (it's async background)
        # srv_run = cast(MagicMock, mock_orchestrator.agent_service.run)
        # srv_run.assert_called_once_with(run=run)
        # It calls repos.runs.create
        cast(MagicMock, mock_orchestrator.repos.runs.create).assert_called_once_with(run)

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
        srv_repo_run_update_status_mock.assert_called_once_with(run_id="run-1", status=AgentRunStatus.cancelled.value)

    @pytest.mark.asyncio
    async def test_execute_workflow(self, mock_orchestrator: OrchestratorService) -> None:
        """Test executing workflow success path."""
        run: AgentRun = AgentRun(
            id="run-1", tenant_id="tenant-1", role="analyst", objective="Task", status=AgentRunStatus.pending
        )
        srv_repo_run_get_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.runs.get)
        srv_repo_run_get_mock.return_value = run

        await mock_orchestrator.execute_workflow("run-1")

        # Verify status update to running
        srv_repo_run_update_status_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.runs.update_status)
        # update_status is called with positional run_id and keyword status
        srv_repo_run_update_status_mock.assert_called_once_with("run-1", status=AgentRunStatus.running.value)

        # Verify service run called
        srv_agent_service_run_mock: MagicMock = cast(MagicMock, mock_orchestrator.agent_service.run)
        srv_agent_service_run_mock.assert_called_once()
        assert srv_agent_service_run_mock.call_args[1]["run"].id == "run-1"

    @pytest.mark.asyncio
    async def test_execute_workflow_failure(self, mock_orchestrator: OrchestratorService) -> None:
        """Test executing workflow failure handling."""
        run: AgentRun = AgentRun(
            id="run-1", tenant_id="tenant-1", role="analyst", objective="Task", status=AgentRunStatus.pending
        )
        srv_repo_run_get_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.runs.get)
        srv_repo_run_get_mock.return_value = run

        # Simulate failure in agent service
        srv_agent_service_run_mock: MagicMock = cast(MagicMock, mock_orchestrator.agent_service.run)
        srv_agent_service_run_mock.side_effect = Exception("Workflow failed")

        await mock_orchestrator.execute_workflow("run-1")

        # Verify status update to running (start) then failed
        srv_repo_run_update_status_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.runs.update_status)
        assert srv_repo_run_update_status_mock.call_count == 2

        # First call: running
        assert srv_repo_run_update_status_mock.call_args_list[0][1]["status"] == AgentRunStatus.running.value
        # Second call: failed
        assert srv_repo_run_update_status_mock.call_args_list[1][1]["status"] == AgentRunStatus.failed.value

        # Verify event log
        srv_repo_events_append_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.events.append)
        srv_repo_events_append_mock.assert_called_once()
        event = srv_repo_events_append_mock.call_args[0][0]
        assert event.run_id == "run-1"
        assert event.type == AgentEventType.run_failed
        assert event.payload["error"] == "Workflow failed"

    @pytest.mark.asyncio
    async def test_execute_resume_failure(self, mock_orchestrator: OrchestratorService) -> None:
        """Test execute_resume failure handling."""
        # Setup
        run_id = "run-1"
        approval_id = "app-1"
        error_msg = "Resume failed"

        srv_agent_service_resume_mock: MagicMock = cast(MagicMock, mock_orchestrator.agent_service.resume)
        srv_agent_service_resume_mock.side_effect = Exception(error_msg)

        # Action
        await mock_orchestrator.execute_resume(run_id, approval_id)

        # Verify status update to failed
        srv_repo_run_update_status_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.runs.update_status)
        srv_repo_run_update_status_mock.assert_called_once_with(run_id, status=AgentRunStatus.failed.value)

        # Verify event log
        srv_repo_events_append_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.events.append)
        srv_repo_events_append_mock.assert_called_once()
        event = srv_repo_events_append_mock.call_args[0][0]
        assert event.run_id == run_id
        assert event.type == AgentEventType.run_failed
        assert event.payload["error"] == error_msg


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

        with patch.object(mock_orchestrator, "execute_resume", new_callable=AsyncMock) as mock_exec_resume:
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

            mock_exec_resume.assert_called_once_with(run_id="run-1", approval_id="approval-1")

    @pytest.mark.asyncio
    async def test_execute_resume(self, mock_orchestrator: OrchestratorService) -> None:
        """Test execute_resume calls agent service."""
        await mock_orchestrator.execute_resume(run_id="run-1", approval_id="approval-1")

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

        with patch.object(mock_orchestrator, "execute_resume", new_callable=AsyncMock) as mock_exec_resume:
            result: Approval = await mock_orchestrator.submit_approval(
                run_id="run-1",
                approval_id="approval-1",
                decision="rejected",
                note="Not approved",
                decided_by="user-1",
            )

            assert result.id == "approval-1"
            # resume should not be called for rejected
            mock_exec_resume.assert_not_called()

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

    @pytest.mark.asyncio
    async def test_submit_approval_tracks_background_task(self, mock_orchestrator: OrchestratorService) -> None:
        """Test that submit_approval adds the resume task to background_tasks."""
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
            reason="Dangerous",
            decision=ApprovalDecision.approved,
        )
        cast(MagicMock, mock_orchestrator.repos.approvals.get).return_value = approval

        # Mock execute_resume to stay pending so we can check the set
        event = asyncio.Event()

        async def delayed_resume(*args, **kwargs):
            await event.wait()

        with patch.object(mock_orchestrator, "execute_resume", side_effect=delayed_resume):
            # Action
            await mock_orchestrator.submit_approval(
                run_id="run-1",
                approval_id="approval-1",
                decision="approved",
                note=None,
                decided_by="user",
            )

            # Verify task is tracked
            assert len(mock_orchestrator.background_tasks) == 1

            # Cleanup: let the task finish
            event.set()
            # Wait for task to complete and callback to run
            await asyncio.sleep(0)
            await asyncio.sleep(0)


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

        srv_repo_usage_list_mock.assert_called_once_with(tenant_id="tenant-1", from_date=from_date, to_date=to_date)

    @pytest.mark.asyncio
    async def test_get_policy(self, mock_orchestrator: OrchestratorService) -> None:
        """Test getting policy for a tenant."""
        policy_config: PolicyConfig = PolicyConfig(autonomy_profile="strict", version="policy-v1")
        srv_repo_policies_get_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.policies.get)
        srv_repo_policies_get_mock.return_value = policy_config

        result: PolicyConfig | None = await mock_orchestrator.get_policy("tenant-1")

        assert result is not None
        assert result.autonomy_profile == "strict"
        srv_repo_policies_get_mock.assert_called_once_with(tenant_id="tenant-1")

    @pytest.mark.asyncio
    async def test_get_policy_not_found(self, mock_orchestrator: OrchestratorService) -> None:
        """Test getting non-existent policy."""
        srv_repo_policies_get_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.policies.get)
        srv_repo_policies_get_mock.return_value = None

        result: PolicyConfig | None = await mock_orchestrator.get_policy("tenant-1")

        assert result is None

    @pytest.mark.asyncio
    async def test_update_policy(self, mock_orchestrator: OrchestratorService) -> None:
        """Test updating policy for a tenant."""
        config: PolicyConfig = PolicyConfig(autonomy_profile="strict")

        result: PolicyConfig = await mock_orchestrator.update_policy("tenant-1", config)

        assert result.autonomy_profile == "strict"
        srv_repo_policies_update_mock: MagicMock = cast(MagicMock, mock_orchestrator.repos.policies.update)
        srv_repo_policies_update_mock.assert_called_once()


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


class TestOrchestratorEventStreaming:
    """Test event streaming functionality with broadcasting queue."""

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
    async def test_stream_history_and_realtime(self, mock_orchestrator: OrchestratorService) -> None:
        """Test streaming yields history first, then realtime events."""
        now = datetime.now(timezone.utc)

        # Historical event
        hist_event = AgentEvent(id="hist-1", run_id="run-1", type=AgentEventType.run_started, created_at=now)
        # Real-time event
        rt_event = AgentEvent(id="rt-1", run_id="run-1", type=AgentEventType.thought_executed, created_at=now)
        # Terminal event
        term_event = AgentEvent(id="term-1", run_id="run-1", type=AgentEventType.run_completed, created_at=now)

        # Mock history fetch
        cast(MagicMock, mock_orchestrator.repos.events.list).return_value = [hist_event]

        # Use a generator to push events to queue after a delay, simulating async arrival
        async def event_generator():
            stream_gen = mock_orchestrator.stream_events("run-1")

            # Start consuming
            # First yield should be from history
            e1 = await anext(stream_gen)
            yield e1

            # Now push realtime events to the queue
            queue = mock_orchestrator.event_listeners["run-1"][0]
            await queue.put(rt_event)
            await queue.put(term_event)

            # Consume realtime
            e2 = await anext(stream_gen)
            yield e2
            e3 = await anext(stream_gen)
            yield e3

            # Should stop after terminal
            try:
                await anext(stream_gen)
            except StopAsyncIteration:
                pass

        received = [e async for e in event_generator()]

        assert len(received) == 3
        # Check history
        assert received[0].data.id == "hist-1"
        # Check realtime
        assert received[1].data.id == "rt-1"
        # Check terminal
        assert received[2].data.id == "term-1"

    @pytest.mark.asyncio
    async def test_stream_keep_alive(self, mock_orchestrator: OrchestratorService) -> None:
        """Test keep-alive is sent on timeout."""
        cast(MagicMock, mock_orchestrator.repos.events.list).return_value = []

        # We need to mock wait_for to raise TimeoutError once
        with patch("asyncio.wait_for", side_effect=[asyncio.TimeoutError, asyncio.CancelledError]) as mock_wait:
            events = []
            try:
                async for event in mock_orchestrator.stream_events("run-1"):
                    events.append(event)
            except asyncio.CancelledError:
                pass

            # Should receive at least one keep-alive
            assert any(isinstance(e, KeepAliveEvent) for e in events)

    @pytest.mark.asyncio
    async def test_stream_events_propagates_cancellation(self, mock_orchestrator: OrchestratorService) -> None:
        """Test that stream_events propagates CancelledError."""
        cast(MagicMock, mock_orchestrator.repos.events.list).return_value = []

        # Raise CancelledError immediately when waiting for queue
        with patch("asyncio.wait_for", side_effect=asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                async for _ in mock_orchestrator.stream_events("run-1"):
                    pass

    @pytest.mark.asyncio
    async def test_broadcasting_repository(self) -> None:
        """Test BroadcastingEventRepository appends to DB and Queue."""
        inner_repo = AsyncMock(spec=EventRepository)
        listeners: Dict[str, Any] = {}
        repo = BroadcastingEventRepository(inner_repo, listeners)

        # Setup listener
        q: asyncio.Queue = asyncio.Queue()
        listeners["run-1"] = [q]

        event = AgentEvent(id="1", run_id="run-1", type=AgentEventType.run_started)

        # Action
        await repo.append(event)

        # Verify DB append
        inner_repo.append.assert_called_once_with(event)

        # Verify Queue append
        assert q.qsize() == 1
        received = await q.get()
        assert received == event

    @pytest.mark.asyncio
    async def test_broadcasting_repository_no_listeners(self) -> None:
        """Test broadcasting repo works without listeners."""
        inner_repo = AsyncMock(spec=EventRepository)
        listeners: Dict[str, Any] = {}
        repo = BroadcastingEventRepository(inner_repo, listeners)

        event = AgentEvent(id="1", run_id="run-1", type=AgentEventType.run_started)

        # Action
        await repo.append(event)

        # Verify DB append
        inner_repo.append.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_broadcasting_repository_list(self) -> None:
        """Test BroadcastingEventRepository delegates list to inner repo."""
        inner_repo = AsyncMock(spec=EventRepository)
        listeners: Dict[str, Any] = {}
        repo = BroadcastingEventRepository(inner_repo, listeners)

        expected_events = [AgentEvent(id="1", run_id="run-1", type=AgentEventType.run_started)]
        inner_repo.list.return_value = expected_events

        # Action
        result = await repo.list("run-1", limit=50)

        # Verify
        inner_repo.list.assert_called_once_with("run-1", 50)
        assert result == expected_events

    @pytest.mark.asyncio
    async def test_stream_events_deduplication(self, mock_orchestrator: OrchestratorService) -> None:
        """Test stream_events deduplicates events present in both history and real-time queue."""
        now = datetime.now(timezone.utc)
        run_id = "run-dup-test"

        # Event present in history
        event_1 = AgentEvent(id="evt-1", run_id=run_id, type=AgentEventType.thought_executed, created_at=now)

        # Same event arriving via queue (race condition simulation)
        event_1_dup = AgentEvent(id="evt-1", run_id=run_id, type=AgentEventType.thought_executed, created_at=now)

        # New event
        event_2 = AgentEvent(id="evt-2", run_id=run_id, type=AgentEventType.run_completed, created_at=now)

        # Mock history fetch
        cast(MagicMock, mock_orchestrator.repos.events.list).return_value = [event_1]

        # Use generator to push to queue
        async def event_generator():
            stream_gen = mock_orchestrator.stream_events(run_id)

            # 1. Fetch history (should yield event_1)
            e1 = await anext(stream_gen)
            yield e1

            # 2. Push duplicate and new event to queue
            queue = mock_orchestrator.event_listeners[run_id][0]
            await queue.put(event_1_dup)
            await queue.put(event_2)

            # 3. Consume next events
            # Should skip event_1_dup and yield event_2
            e2 = await anext(stream_gen)
            yield e2

            # Should be done (event_2 is terminal run_completed)
            try:
                await anext(stream_gen)
            except StopAsyncIteration:
                pass

        received = [e async for e in event_generator()]

        assert len(received) == 2
        assert received[0].data.id == "evt-1"
        assert received[1].data.id == "evt-2"


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
