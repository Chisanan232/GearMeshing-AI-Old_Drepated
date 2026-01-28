"""End-to-end tests for SQL repository implementations.

This module provides comprehensive testing of all SQL repository implementations
to verify correct persistence, retrieval, and manipulation of domain objects.
Tests cover all repositories: Run, Event, Approval, Checkpoint, ToolInvocation,
Usage, and Policy repositories.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import JSON
from sqlalchemy.dialects.sqlite.base import SQLiteTypeCompiler

# Patch SQLite type compiler to handle JSONB (needed for in-memory testing)
original_process = SQLiteTypeCompiler.process


def patched_process(self, type_, **kw):
    from sqlalchemy.dialects.postgresql import JSONB

    if isinstance(type_, JSONB):
        return self.process(JSON(), **kw)
    return original_process(self, type_, **kw)


SQLiteTypeCompiler.process = patched_process  # type: ignore[method-assign]

from gearmeshing_ai.agent_core.policy.models import PolicyConfig
from gearmeshing_ai.agent_core.repos.sql import (
    SqlRepoBundle,
    build_sql_repos,
    create_all,
    create_engine,
    create_sessionmaker,
)
from gearmeshing_ai.core.models.domain import (
    AgentEvent,
    AgentEventType,
    AgentRun,
    AgentRunStatus,
    Approval,
    ApprovalDecision,
    AutonomyProfile,
    Checkpoint,
    RiskLevel,
    ToolInvocation,
    UsageLedgerEntry,
)
from gearmeshing_ai.info_provider import CapabilityName


@pytest.fixture
async def db_engine():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await create_all(engine)
    yield engine
    await engine.dispose()


@pytest.fixture
async def repos(db_engine) -> SqlRepoBundle:
    """Create repository bundle with in-memory database."""
    session_factory = create_sessionmaker(db_engine)
    return build_sql_repos(session_factory=session_factory)


class TestRunRepository:
    """Test suite for SqlRunRepository."""

    @pytest.mark.asyncio
    async def test_create_and_get_run(self, repos: SqlRepoBundle) -> None:
        """Test creating and retrieving a run."""
        run = AgentRun(
            id="run-1",
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            role="analyst",
            autonomy_profile=AutonomyProfile.balanced,
            objective="Analyze market trends",
            done_when="Market analysis complete",
            prompt_provider_version="v1.0",
            status=AgentRunStatus.running,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        await repos.runs.create(run)
        retrieved = await repos.runs.get("run-1")

        assert retrieved is not None
        assert retrieved.id == "run-1"
        assert retrieved.tenant_id == "tenant-1"
        assert retrieved.objective == "Analyze market trends"
        assert retrieved.status == AgentRunStatus.running
        assert retrieved.autonomy_profile == AutonomyProfile.balanced

    @pytest.mark.asyncio
    async def test_get_nonexistent_run(self, repos: SqlRepoBundle) -> None:
        """Test retrieving a non-existent run returns None."""
        result = await repos.runs.get("nonexistent-run")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_run_status(self, repos: SqlRepoBundle) -> None:
        """Test updating run status."""
        run = AgentRun(
            id="run-2",
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            role="analyst",
            autonomy_profile=AutonomyProfile.strict,
            objective="Test objective",
            done_when="Done",
            prompt_provider_version="v1.0",
            status=AgentRunStatus.running,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        await repos.runs.create(run)
        await repos.runs.update_status("run-2", status=AgentRunStatus.succeeded.value)

        updated = await repos.runs.get("run-2")
        assert updated is not None
        assert updated.status == AgentRunStatus.succeeded

    @pytest.mark.asyncio
    async def test_update_nonexistent_run_status(self, repos: SqlRepoBundle) -> None:
        """Test updating status of non-existent run does not raise error."""
        await repos.runs.update_status("nonexistent", status=AgentRunStatus.succeeded.value)

    @pytest.mark.asyncio
    async def test_list_runs(self, repos: SqlRepoBundle) -> None:
        """Test listing runs."""
        for i in range(3):
            run = AgentRun(
                id=f"run-{i}",
                tenant_id="tenant-1",
                workspace_id="workspace-1",
                role="analyst",
                autonomy_profile=AutonomyProfile.balanced,
                objective=f"Objective {i}",
                done_when="Done",
                prompt_provider_version="v1.0",
                status=AgentRunStatus.running,
                created_at=datetime.now(timezone.utc) + timedelta(seconds=i),
                updated_at=datetime.now(timezone.utc),
            )
            await repos.runs.create(run)

        runs = await repos.runs.list(tenant_id="tenant-1")
        assert len(runs) == 3
        assert all(r.tenant_id == "tenant-1" for r in runs)

    @pytest.mark.asyncio
    async def test_list_runs_with_pagination(self, repos: SqlRepoBundle) -> None:
        """Test listing runs with limit and offset."""
        for i in range(5):
            run = AgentRun(
                id=f"run-{i}",
                tenant_id="tenant-1",
                workspace_id="workspace-1",
                role="analyst",
                autonomy_profile=AutonomyProfile.balanced,
                objective=f"Objective {i}",
                done_when="Done",
                prompt_provider_version="v1.0",
                status=AgentRunStatus.running,
                created_at=datetime.now(timezone.utc) + timedelta(seconds=i),
                updated_at=datetime.now(timezone.utc),
            )
            await repos.runs.create(run)

        page1 = await repos.runs.list(tenant_id="tenant-1", limit=2, offset=0)
        page2 = await repos.runs.list(tenant_id="tenant-1", limit=2, offset=2)

        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].id != page2[0].id

    @pytest.mark.asyncio
    async def test_list_runs_by_tenant(self, repos: SqlRepoBundle) -> None:
        """Test filtering runs by tenant."""
        for tenant_id in ["tenant-1", "tenant-2"]:
            for i in range(2):
                run = AgentRun(
                    id=f"{tenant_id}-run-{i}",
                    tenant_id=tenant_id,
                    workspace_id="workspace-1",
                    role="analyst",
                    autonomy_profile=AutonomyProfile.balanced,
                    objective=f"Objective {i}",
                    done_when="Done",
                    prompt_provider_version="v1.0",
                    status=AgentRunStatus.running,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
                await repos.runs.create(run)

        tenant1_runs = await repos.runs.list(tenant_id="tenant-1")
        tenant2_runs = await repos.runs.list(tenant_id="tenant-2")

        assert len(tenant1_runs) == 2
        assert len(tenant2_runs) == 2
        assert all(r.tenant_id == "tenant-1" for r in tenant1_runs)
        assert all(r.tenant_id == "tenant-2" for r in tenant2_runs)


class TestEventRepository:
    """Test suite for SqlEventRepository."""

    @pytest.mark.asyncio
    async def test_append_and_list_events(self, repos: SqlRepoBundle) -> None:
        """Test appending and listing events."""
        run_id = "run-1"

        for i in range(3):
            event = AgentEvent(
                id=f"event-{i}",
                run_id=run_id,
                type=AgentEventType.thought_executed,
                created_at=datetime.now(timezone.utc) + timedelta(seconds=i),
                correlation_id=f"corr-{i}",
                payload={"thought": f"Thought {i}"},
            )
            await repos.events.append(event)

        events = await repos.events.list(run_id)
        assert len(events) == 3
        assert all(e.run_id == run_id for e in events)
        assert events[0].type == AgentEventType.thought_executed

    @pytest.mark.asyncio
    async def test_list_events_empty(self, repos: SqlRepoBundle) -> None:
        """Test listing events for non-existent run returns empty list."""
        events = await repos.events.list("nonexistent-run")
        assert events == []

    @pytest.mark.asyncio
    async def test_list_events_with_limit(self, repos: SqlRepoBundle) -> None:
        """Test listing events with limit."""
        run_id = "run-1"

        for i in range(5):
            event = AgentEvent(
                id=f"event-{i}",
                run_id=run_id,
                type=AgentEventType.thought_executed,
                created_at=datetime.now(timezone.utc) + timedelta(seconds=i),
                correlation_id=f"corr-{i}",
                payload={"index": i},
            )
            await repos.events.append(event)

        events = await repos.events.list(run_id, limit=3)
        assert len(events) == 3

    @pytest.mark.asyncio
    async def test_events_ordered_by_creation(self, repos: SqlRepoBundle) -> None:
        """Test events are returned in creation order."""
        run_id = "run-1"
        base_time = datetime.now(timezone.utc)

        for i in range(3):
            event = AgentEvent(
                id=f"event-{i}",
                run_id=run_id,
                type=AgentEventType.thought_executed,
                created_at=base_time + timedelta(seconds=i),
                correlation_id=f"corr-{i}",
                payload={"index": i},
            )
            await repos.events.append(event)

        events = await repos.events.list(run_id)
        assert events[0].payload["index"] == 0
        assert events[1].payload["index"] == 1
        assert events[2].payload["index"] == 2


class TestApprovalRepository:
    """Test suite for SqlApprovalRepository."""

    @pytest.mark.asyncio
    async def test_create_and_get_approval(self, repos: SqlRepoBundle) -> None:
        """Test creating and retrieving an approval."""
        approval = Approval(
            id="approval-1",
            run_id="run-1",
            risk=RiskLevel.high,
            capability=CapabilityName.shell_exec,
            reason="High-risk operation",
            requested_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
            decision=None,
            decided_at=None,
            decided_by=None,
        )

        await repos.approvals.create(approval)
        retrieved = await repos.approvals.get("approval-1")

        assert retrieved is not None
        assert retrieved.id == "approval-1"
        assert retrieved.risk == RiskLevel.high
        assert retrieved.capability == CapabilityName.shell_exec
        assert retrieved.decision is None

    @pytest.mark.asyncio
    async def test_resolve_approval(self, repos: SqlRepoBundle) -> None:
        """Test resolving an approval."""
        approval = Approval(
            id="approval-2",
            run_id="run-1",
            risk=RiskLevel.medium,
            capability=CapabilityName.code_execution,
            reason="Code execution requested",
            requested_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
            decision=None,
            decided_at=None,
            decided_by=None,
        )

        await repos.approvals.create(approval)
        await repos.approvals.resolve("approval-2", decision=ApprovalDecision.approved.value, decided_by="user-1")

        resolved = await repos.approvals.get("approval-2")
        assert resolved is not None
        assert resolved.decision == ApprovalDecision.approved
        assert resolved.decided_by == "user-1"
        assert resolved.decided_at is not None

    @pytest.mark.asyncio
    async def test_list_pending_approvals(self, repos: SqlRepoBundle) -> None:
        """Test listing pending approvals."""
        run_id = "run-1"

        for i in range(3):
            approval = Approval(
                id=f"approval-{i}",
                run_id=run_id,
                risk=RiskLevel.high,
                capability=CapabilityName.shell_exec,
                reason=f"Reason {i}",
                requested_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
                decision=None,
                decided_at=None,
                decided_by=None,
            )
            await repos.approvals.create(approval)

        await repos.approvals.resolve("approval-1", decision=ApprovalDecision.approved.value, decided_by="user-1")

        pending = await repos.approvals.list(run_id, pending_only=True)
        assert len(pending) == 2
        assert all(a.decision is None for a in pending)

    @pytest.mark.asyncio
    async def test_list_all_approvals(self, repos: SqlRepoBundle) -> None:
        """Test listing all approvals including resolved."""
        run_id = "run-1"

        for i in range(3):
            approval = Approval(
                id=f"approval-{i}",
                run_id=run_id,
                risk=RiskLevel.high,
                capability=CapabilityName.shell_exec,
                reason=f"Reason {i}",
                requested_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
                decision=None,
                decided_at=None,
                decided_by=None,
            )
            await repos.approvals.create(approval)

        await repos.approvals.resolve("approval-1", decision=ApprovalDecision.approved.value, decided_by="user-1")

        all_approvals = await repos.approvals.list(run_id, pending_only=False)
        assert len(all_approvals) == 3


class TestCheckpointRepository:
    """Test suite for SqlCheckpointRepository."""

    @pytest.mark.asyncio
    async def test_save_and_get_latest_checkpoint(self, repos: SqlRepoBundle) -> None:
        """Test saving and retrieving latest checkpoint."""
        run_id = "run-1"

        checkpoint = Checkpoint(
            id="checkpoint-1",
            run_id=run_id,
            node="node_1",
            state={"key": "value"},
            created_at=datetime.now(timezone.utc),
        )

        await repos.checkpoints.save(checkpoint)
        retrieved = await repos.checkpoints.latest(run_id)

        assert retrieved is not None
        assert retrieved.id == "checkpoint-1"
        assert retrieved.node == "node_1"
        assert retrieved.state == {"key": "value"}

    @pytest.mark.asyncio
    async def test_latest_checkpoint_returns_most_recent(self, repos: SqlRepoBundle) -> None:
        """Test latest checkpoint returns most recent one."""
        run_id = "run-1"
        base_time = datetime.now(timezone.utc)

        for i in range(3):
            checkpoint = Checkpoint(
                id=f"checkpoint-{i}",
                run_id=run_id,
                node=f"node_{i}",
                state={"index": i},
                created_at=base_time + timedelta(seconds=i),
            )
            await repos.checkpoints.save(checkpoint)

        latest = await repos.checkpoints.latest(run_id)
        assert latest is not None
        assert latest.id == "checkpoint-2"
        assert latest.state["index"] == 2

    @pytest.mark.asyncio
    async def test_latest_checkpoint_nonexistent_run(self, repos: SqlRepoBundle) -> None:
        """Test latest checkpoint for non-existent run returns None."""
        result = await repos.checkpoints.latest("nonexistent-run")
        assert result is None


class TestToolInvocationRepository:
    """Test suite for SqlToolInvocationRepository."""

    @pytest.mark.asyncio
    async def test_append_tool_invocation(self, repos: SqlRepoBundle) -> None:
        """Test appending a tool invocation."""
        invocation = ToolInvocation(
            id="invocation-1",
            run_id="run-1",
            server_id="github-mcp",
            tool_name="create_issue",
            args={"title": "Bug fix", "body": "Fix memory leak"},
            ok=True,
            result={"issue_id": "123"},
            risk=RiskLevel.medium,
            created_at=datetime.now(timezone.utc),
        )

        await repos.tool_invocations.append(invocation)

    @pytest.mark.asyncio
    async def test_append_multiple_tool_invocations(self, repos: SqlRepoBundle) -> None:
        """Test appending multiple tool invocations."""
        for i in range(3):
            invocation = ToolInvocation(
                id=f"invocation-{i}",
                run_id="run-1",
                server_id="github-mcp",
                tool_name=f"tool_{i}",
                args={"index": i},
                ok=True,
                result={"success": True},
                risk=RiskLevel.low,
                created_at=datetime.now(timezone.utc) + timedelta(seconds=i),
            )
            await repos.tool_invocations.append(invocation)


class TestUsageRepository:
    """Test suite for SqlUsageRepository."""

    @pytest.mark.asyncio
    async def test_append_usage_entry(self, repos: SqlRepoBundle) -> None:
        """Test appending a usage ledger entry."""
        run = AgentRun(
            id="run-1",
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            role="analyst",
            autonomy_profile=AutonomyProfile.balanced,
            objective="Test",
            done_when="Done",
            prompt_provider_version="v1.0",
            status=AgentRunStatus.running,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        await repos.runs.create(run)

        usage = UsageLedgerEntry(
            id="usage-1",
            run_id="run-1",
            provider="openai",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.005,
            created_at=datetime.now(timezone.utc),
        )

        await repos.usage.append(usage)

    @pytest.mark.asyncio
    async def test_list_usage_by_tenant(self, repos: SqlRepoBundle) -> None:
        """Test listing usage entries by tenant."""
        for tenant_id in ["tenant-1", "tenant-2"]:
            run = AgentRun(
                id=f"{tenant_id}-run-1",
                tenant_id=tenant_id,
                workspace_id="workspace-1",
                role="analyst",
                autonomy_profile=AutonomyProfile.balanced,
                objective="Test",
                done_when="Done",
                prompt_provider_version="v1.0",
                status=AgentRunStatus.running,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            await repos.runs.create(run)

            for i in range(2):
                usage = UsageLedgerEntry(
                    id=f"{tenant_id}-usage-{i}",
                    run_id=f"{tenant_id}-run-1",
                    provider="openai",
                    model="gpt-4",
                    prompt_tokens=100 * (i + 1),
                    completion_tokens=50 * (i + 1),
                    total_tokens=150 * (i + 1),
                    cost_usd=0.005 * (i + 1),
                    created_at=datetime.now(timezone.utc) + timedelta(seconds=i),
                )
                await repos.usage.append(usage)

        tenant1_usage = await repos.usage.list("tenant-1")
        tenant2_usage = await repos.usage.list("tenant-2")

        assert len(tenant1_usage) == 2
        assert len(tenant2_usage) == 2

    @pytest.mark.asyncio
    async def test_list_usage_with_date_range(self, repos: SqlRepoBundle) -> None:
        """Test listing usage entries with date range filter."""
        run = AgentRun(
            id="run-1",
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            role="analyst",
            autonomy_profile=AutonomyProfile.balanced,
            objective="Test",
            done_when="Done",
            prompt_provider_version="v1.0",
            status=AgentRunStatus.running,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        await repos.runs.create(run)

        base_time = datetime.now(timezone.utc)
        for i in range(5):
            usage = UsageLedgerEntry(
                id=f"usage-{i}",
                run_id="run-1",
                provider="openai",
                model="gpt-4",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cost_usd=0.005,
                created_at=base_time + timedelta(hours=i),
            )
            await repos.usage.append(usage)

        from_date = base_time + timedelta(hours=1)
        to_date = base_time + timedelta(hours=3)

        filtered = await repos.usage.list("tenant-1", from_date=from_date, to_date=to_date)
        assert len(filtered) == 3


class TestPolicyRepository:
    """Test suite for SqlPolicyRepository."""

    @pytest.mark.asyncio
    async def test_create_and_get_policy(self, repos: SqlRepoBundle) -> None:
        """Test creating and retrieving a policy."""
        config = PolicyConfig(autonomy_profile="strict")

        await repos.policies.update("tenant-1", config)
        retrieved = await repos.policies.get("tenant-1")

        assert retrieved is not None
        assert isinstance(retrieved, PolicyConfig)
        assert retrieved.autonomy_profile == "strict"

    @pytest.mark.asyncio
    async def test_get_nonexistent_policy(self, repos: SqlRepoBundle) -> None:
        """Test retrieving non-existent policy returns None."""
        result = await repos.policies.get("nonexistent-tenant")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_existing_policy(self, repos: SqlRepoBundle) -> None:
        """Test updating an existing policy."""
        config1 = PolicyConfig(autonomy_profile="balanced")
        await repos.policies.update("tenant-1", config1)

        config2 = PolicyConfig(autonomy_profile="strict")
        await repos.policies.update("tenant-1", config2)

        retrieved = await repos.policies.get("tenant-1")
        assert retrieved is not None
        assert isinstance(retrieved, PolicyConfig)
        assert retrieved.autonomy_profile == "strict"

    @pytest.mark.asyncio
    async def test_policy_with_nested_config(self, repos: SqlRepoBundle) -> None:
        """Test policy with nested configuration."""
        config = PolicyConfig(
            autonomy_profile="strict",
            version="policy-v1",
        )

        await repos.policies.update("tenant-1", config)
        retrieved = await repos.policies.get("tenant-1")

        assert retrieved is not None
        assert isinstance(retrieved, PolicyConfig)
        assert retrieved.autonomy_profile == "strict"
        assert retrieved.version == "policy-v1"


class TestRepositoryIntegration:
    """Integration tests across multiple repositories."""

    @pytest.mark.asyncio
    async def test_complete_run_lifecycle(self, repos: SqlRepoBundle) -> None:
        """Test complete run lifecycle with multiple repository operations."""
        run_id = "run-1"
        tenant_id = "tenant-1"

        run = AgentRun(
            id=run_id,
            tenant_id=tenant_id,
            workspace_id="workspace-1",
            role="analyst",
            autonomy_profile=AutonomyProfile.balanced,
            objective="Analyze data",
            done_when="Analysis complete",
            prompt_provider_version="v1.0",
            status=AgentRunStatus.running,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        await repos.runs.create(run)

        event = AgentEvent(
            id="event-1",
            run_id=run_id,
            type=AgentEventType.thought_executed,
            created_at=datetime.now(timezone.utc),
            correlation_id="corr-1",
            payload={"thought": "Starting analysis"},
        )
        await repos.events.append(event)

        checkpoint = Checkpoint(
            id="checkpoint-1",
            run_id=run_id,
            node="node_1",
            state={"progress": 0.5},
            created_at=datetime.now(timezone.utc),
        )
        await repos.checkpoints.save(checkpoint)

        usage = UsageLedgerEntry(
            id="usage-1",
            run_id=run_id,
            provider="openai",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.005,
            created_at=datetime.now(timezone.utc),
        )
        await repos.usage.append(usage)

        await repos.runs.update_status(run_id, status=AgentRunStatus.succeeded.value)

        retrieved_run = await repos.runs.get(run_id)
        retrieved_events = await repos.events.list(run_id)
        retrieved_checkpoint = await repos.checkpoints.latest(run_id)
        retrieved_usage = await repos.usage.list(tenant_id)

        assert retrieved_run is not None
        assert retrieved_run.status == AgentRunStatus.succeeded
        assert len(retrieved_events) == 1
        assert retrieved_checkpoint is not None
        assert len(retrieved_usage) == 1

    @pytest.mark.asyncio
    async def test_approval_workflow(self, repos: SqlRepoBundle) -> None:
        """Test complete approval workflow."""
        run_id = "run-1"

        approval = Approval(
            id="approval-1",
            run_id=run_id,
            risk=RiskLevel.high,
            capability=CapabilityName.shell_exec,
            reason="High-risk shell execution",
            requested_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
            decision=None,
            decided_at=None,
            decided_by=None,
        )
        await repos.approvals.create(approval)

        pending = await repos.approvals.list(run_id, pending_only=True)
        assert len(pending) == 1

        await repos.approvals.resolve("approval-1", decision=ApprovalDecision.approved.value, decided_by="user-1")

        pending_after = await repos.approvals.list(run_id, pending_only=True)
        assert len(pending_after) == 0

        all_approvals = await repos.approvals.list(run_id, pending_only=False)
        assert len(all_approvals) == 1
        assert all_approvals[0].decision == ApprovalDecision.approved

    @pytest.mark.asyncio
    async def test_multi_tenant_isolation(self, repos: SqlRepoBundle) -> None:
        """Test data isolation between tenants."""
        for tenant_id in ["tenant-1", "tenant-2"]:
            run = AgentRun(
                id=f"{tenant_id}-run",
                tenant_id=tenant_id,
                workspace_id="workspace-1",
                role="analyst",
                autonomy_profile=AutonomyProfile.balanced,
                objective="Test",
                done_when="Done",
                prompt_provider_version="v1.0",
                status=AgentRunStatus.running,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            await repos.runs.create(run)

            usage = UsageLedgerEntry(
                id=f"{tenant_id}-usage",
                run_id=f"{tenant_id}-run",
                provider="openai",
                model="gpt-4",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cost_usd=0.005,
                created_at=datetime.now(timezone.utc),
            )
            await repos.usage.append(usage)

        tenant1_usage = await repos.usage.list("tenant-1")
        tenant2_usage = await repos.usage.list("tenant-2")

        assert len(tenant1_usage) == 1
        assert len(tenant2_usage) == 1
        assert tenant1_usage[0].run_id == "tenant-1-run"
        assert tenant2_usage[0].run_id == "tenant-2-run"
