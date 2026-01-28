"""End-to-end tests for SQL repositories using Testcontainers PostgreSQL.

This module provides comprehensive testing of all SQL repository implementations
with a real PostgreSQL database using Testcontainers, testing real-world scenarios
and edge cases that are closer to production usage.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from testcontainers.postgres import PostgresContainer

from gearmeshing_ai.core.models.domain.policy import PolicyConfig
from gearmeshing_ai.core.database import (
    create_all,
    create_engine,
    create_sessionmaker,
)
from gearmeshing_ai.agent_core.repos.sql import (
    SqlRepoBundle,
    build_sql_repos,
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


@pytest.fixture(scope="session")
def postgres_container():
    """Start a PostgreSQL container for the test session."""
    container = PostgresContainer("postgres:16-alpine")
    container.start()
    yield container
    container.stop()


@pytest.fixture
async def db_engine(postgres_container):
    """Create a database engine connected to the test PostgreSQL container."""
    db_url = postgres_container.get_connection_url()
    engine = create_engine(db_url)
    await create_all(engine)
    yield engine
    await engine.dispose()


@pytest.fixture
async def repos(db_engine) -> SqlRepoBundle:
    """Create repository bundle with test PostgreSQL database."""
    session_factory = create_sessionmaker(db_engine)
    return build_sql_repos(session_factory=session_factory)


class TestRunRepositoryEdgeCases:
    """Edge case tests for SqlRunRepository."""

    @pytest.mark.asyncio
    async def test_create_run_with_special_characters_in_fields(self, repos: SqlRepoBundle) -> None:
        """Test creating run with special characters in text fields."""
        run = AgentRun(
            id="run-special-chars",
            tenant_id="tenant-with-dashes-123",
            workspace_id="workspace_with_underscores",
            role="analyst/researcher",
            autonomy_profile=AutonomyProfile.balanced,
            objective="Analyze 'quoted' data & special chars: @#$%",
            done_when='Analysis with "quotes" complete',
            prompt_provider_version="v1.0-beta+build.123",
            status=AgentRunStatus.running,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        await repos.runs.create(run)
        retrieved = await repos.runs.get("run-special-chars")

        assert retrieved is not None
        assert retrieved.objective == "Analyze 'quoted' data & special chars: @#$%"
        assert retrieved.done_when == 'Analysis with "quotes" complete'

    @pytest.mark.asyncio
    async def test_create_run_with_very_long_objective(self, repos: SqlRepoBundle) -> None:
        """Test creating run with very long objective text."""
        long_objective = "A" * 5000
        run = AgentRun(
            id="run-long-text",
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            role="analyst",
            autonomy_profile=AutonomyProfile.balanced,
            objective=long_objective,
            done_when="Done",
            prompt_provider_version="v1.0",
            status=AgentRunStatus.running,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        await repos.runs.create(run)
        retrieved = await repos.runs.get("run-long-text")

        assert retrieved is not None
        assert len(retrieved.objective) == 5000
        assert retrieved.objective == long_objective

    @pytest.mark.asyncio
    async def test_update_status_multiple_times(self, repos: SqlRepoBundle) -> None:
        """Test updating run status multiple times in sequence."""
        run = AgentRun(
            id="run-multi-status",
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

        status_sequence = [
            AgentRunStatus.paused_for_approval,
            AgentRunStatus.running,
            AgentRunStatus.succeeded,
        ]

        for status in status_sequence:
            await repos.runs.update_status("run-multi-status", status=status.value)
            retrieved = await repos.runs.get("run-multi-status")
            assert retrieved
            assert retrieved.status == status

    @pytest.mark.asyncio
    async def test_list_runs_with_large_offset(self, repos: SqlRepoBundle) -> None:
        """Test listing runs with offset larger than result set."""
        for i in range(5):
            run = AgentRun(
                id=f"run-offset-{i}",
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

        results = await repos.runs.list(tenant_id="tenant-1", limit=10, offset=100)
        assert results == []

    @pytest.mark.asyncio
    async def test_list_runs_with_zero_limit(self, repos: SqlRepoBundle) -> None:
        """Test listing runs with zero limit returns empty list."""
        for i in range(3):
            run = AgentRun(
                id=f"run-zero-limit-{i}",
                tenant_id="tenant-1",
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

        results = await repos.runs.list(tenant_id="tenant-1", limit=0)
        assert results == []

    @pytest.mark.asyncio
    async def test_create_runs_with_same_id_fails_gracefully(self, repos: SqlRepoBundle) -> None:
        """Test creating duplicate run IDs (should fail at database level)."""
        run1 = AgentRun(
            id="run-duplicate",
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            role="analyst",
            autonomy_profile=AutonomyProfile.balanced,
            objective="First",
            done_when="Done",
            prompt_provider_version="v1.0",
            status=AgentRunStatus.running,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        await repos.runs.create(run1)

        run2 = AgentRun(
            id="run-duplicate",
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            role="analyst",
            autonomy_profile=AutonomyProfile.balanced,
            objective="Second",
            done_when="Done",
            prompt_provider_version="v1.0",
            status=AgentRunStatus.running,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        with pytest.raises(Exception):
            await repos.runs.create(run2)


class TestEventRepositoryEdgeCases:
    """Edge case tests for SqlEventRepository."""

    @pytest.mark.asyncio
    async def test_append_event_with_large_payload(self, repos: SqlRepoBundle) -> None:
        """Test appending event with large JSON payload."""
        large_payload = {"data": "X" * 10000, "nested": {"deep": {"structure": ["item"] * 1000}}}

        event = AgentEvent(
            id="event-large-payload",
            run_id="run-1",
            type=AgentEventType.thought_executed,
            created_at=datetime.now(timezone.utc),
            correlation_id="corr-1",
            payload=large_payload,
        )

        await repos.events.append(event)
        events = await repos.events.list("run-1")

        assert len(events) == 1
        assert events[0].payload == large_payload

    @pytest.mark.asyncio
    async def test_append_events_with_null_correlation_id(self, repos: SqlRepoBundle) -> None:
        """Test appending events with null correlation IDs."""
        run_id = "run-null-corr"
        event = AgentEvent(
            id="event-no-corr",
            run_id=run_id,
            type=AgentEventType.thought_executed,
            created_at=datetime.now(timezone.utc),
            correlation_id=None,
            payload={"test": "data"},
        )

        await repos.events.append(event)
        events = await repos.events.list(run_id)

        assert len(events) == 1
        assert events[0].correlation_id is None

    @pytest.mark.asyncio
    async def test_list_events_with_different_event_types(self, repos: SqlRepoBundle) -> None:
        """Test listing events with various event types."""
        run_id = "run-event-types"
        event_types = [
            AgentEventType.thought_executed,
            AgentEventType.artifact_created,
            AgentEventType.capability_executed,
            AgentEventType.tool_invoked,
            AgentEventType.approval_requested,
            AgentEventType.approval_resolved,
        ]

        for i, event_type in enumerate(event_types):
            event = AgentEvent(
                id=f"event-type-{i}",
                run_id=run_id,
                type=event_type,
                created_at=datetime.now(timezone.utc) + timedelta(seconds=i),
                correlation_id=f"corr-{i}",
                payload={"type": event_type.value},
            )
            await repos.events.append(event)

        events = await repos.events.list(run_id)
        assert len(events) == len(event_types)
        assert all(e.type in event_types for e in events)

    @pytest.mark.asyncio
    async def test_list_events_respects_limit(self, repos: SqlRepoBundle) -> None:
        """Test that list events respects the limit parameter."""
        run_id = "run-event-limit"
        for i in range(100):
            event = AgentEvent(
                id=f"event-limit-{i}",
                run_id=run_id,
                type=AgentEventType.thought_executed,
                created_at=datetime.now(timezone.utc) + timedelta(milliseconds=i),
                correlation_id=f"corr-{i}",
                payload={"index": i},
            )
            await repos.events.append(event)

        events_10 = await repos.events.list(run_id, limit=10)
        events_50 = await repos.events.list(run_id, limit=50)
        events_all = await repos.events.list(run_id, limit=1000)

        assert len(events_10) == 10
        assert len(events_50) == 50
        assert len(events_all) == 100


class TestApprovalRepositoryEdgeCases:
    """Edge case tests for SqlApprovalRepository."""

    @pytest.mark.asyncio
    async def test_approval_expiration_edge_case(self, repos: SqlRepoBundle) -> None:
        """Test approval with expiration in the past."""
        approval = Approval(
            id="approval-expired",
            run_id="run-1",
            risk=RiskLevel.high,
            capability=CapabilityName.shell_exec,
            reason="Test expired approval",
            requested_at=datetime.now(timezone.utc) - timedelta(hours=1),
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=30),
            decision=None,
            decided_at=None,
            decided_by=None,
        )

        await repos.approvals.create(approval)
        retrieved = await repos.approvals.get("approval-expired")

        assert retrieved is not None
        assert retrieved.expires_at
        assert retrieved.expires_at < datetime.now(timezone.utc)

    @pytest.mark.asyncio
    async def test_resolve_approval_with_long_reason(self, repos: SqlRepoBundle) -> None:
        """Test resolving approval with very long reason text."""
        approval = Approval(
            id="approval-long-reason",
            run_id="run-1",
            risk=RiskLevel.medium,
            capability=CapabilityName.code_execution,
            reason="X" * 2000,
            requested_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
            decision=None,
            decided_at=None,
            decided_by=None,
        )

        await repos.approvals.create(approval)
        await repos.approvals.resolve(
            "approval-long-reason",
            decision=ApprovalDecision.approved.value,
            decided_by="user-with-very-long-name-" + "X" * 100,
        )

        resolved = await repos.approvals.get("approval-long-reason")
        assert resolved is not None
        assert len(resolved.reason) == 2000
        assert resolved.decided_by
        assert len(resolved.decided_by) > 100

    @pytest.mark.asyncio
    async def test_list_approvals_with_mixed_decisions(self, repos: SqlRepoBundle) -> None:
        """Test listing approvals with mixed decision states."""
        run_id = "run-mixed-decisions"
        decisions = [
            (ApprovalDecision.approved, "user-1"),
            (ApprovalDecision.rejected, "user-2"),
            (None, None),
            (ApprovalDecision.approved, "user-3"),
            (None, None),
        ]

        for i, (decision, decided_by) in enumerate(decisions):
            approval = Approval(
                id=f"approval-mixed-{i}",
                run_id=run_id,
                risk=RiskLevel.high,
                capability=CapabilityName.shell_exec,
                reason=f"Reason {i}",
                requested_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
                decision=decision,
                decided_at=datetime.now(timezone.utc) if decision else None,
                decided_by=decided_by,
            )
            await repos.approvals.create(approval)

        pending = await repos.approvals.list(run_id, pending_only=True)
        all_approvals = await repos.approvals.list(run_id, pending_only=False)

        assert len(pending) == 2
        assert len(all_approvals) == 5
        assert all(a.decision is None for a in pending)

    @pytest.mark.asyncio
    async def test_resolve_already_resolved_approval(self, repos: SqlRepoBundle) -> None:
        """Test resolving an already resolved approval (should update)."""
        approval = Approval(
            id="approval-re-resolve",
            run_id="run-1",
            risk=RiskLevel.high,
            capability=CapabilityName.shell_exec,
            reason="Test",
            requested_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
            decision=None,
            decided_at=None,
            decided_by=None,
        )

        await repos.approvals.create(approval)
        await repos.approvals.resolve(
            "approval-re-resolve", decision=ApprovalDecision.approved.value, decided_by="user-1"
        )

        first_resolve = await repos.approvals.get("approval-re-resolve")
        assert first_resolve
        first_decided_at = first_resolve.decided_at

        await asyncio.sleep(0.1)

        await repos.approvals.resolve(
            "approval-re-resolve", decision=ApprovalDecision.rejected.value, decided_by="user-2"
        )

        second_resolve = await repos.approvals.get("approval-re-resolve")

        assert second_resolve
        assert second_resolve.decision
        assert second_resolve.decision == ApprovalDecision.rejected
        assert second_resolve.decided_by
        assert second_resolve.decided_by == "user-2"
        assert second_resolve.decided_at
        assert first_decided_at
        assert second_resolve.decided_at > first_decided_at


class TestCheckpointRepositoryEdgeCases:
    """Edge case tests for SqlCheckpointRepository."""

    @pytest.mark.asyncio
    async def test_checkpoint_with_large_state(self, repos: SqlRepoBundle) -> None:
        """Test checkpoint with large state object."""
        run_id = "run-large-checkpoint"
        large_state = {
            "graph_state": {
                "nodes": [{"id": i, "data": "X" * 100} for i in range(100)],
                "edges": [[i, i + 1] for i in range(99)],
            }
        }

        checkpoint = Checkpoint(
            id="checkpoint-large",
            run_id=run_id,
            node="node_final",
            state=large_state,
            created_at=datetime.now(timezone.utc),
        )

        await repos.checkpoints.save(checkpoint)
        retrieved = await repos.checkpoints.latest(run_id)

        assert retrieved is not None
        assert retrieved.state == large_state

    @pytest.mark.asyncio
    async def test_multiple_checkpoints_latest_returns_newest(self, repos: SqlRepoBundle) -> None:
        """Test that latest always returns the most recent checkpoint."""
        base_time = datetime.now(timezone.utc)

        for i in range(10):
            checkpoint = Checkpoint(
                id=f"checkpoint-{i}",
                run_id="run-1",
                node=f"node_{i}",
                state={"step": i},
                created_at=base_time + timedelta(seconds=i),
            )
            await repos.checkpoints.save(checkpoint)

        latest = await repos.checkpoints.latest("run-1")
        assert latest is not None
        assert latest.id == "checkpoint-9"
        assert latest.state["step"] == 9

    @pytest.mark.asyncio
    async def test_checkpoint_with_empty_state(self, repos: SqlRepoBundle) -> None:
        """Test checkpoint with empty state object."""
        run_id = "run-empty-checkpoint"

        checkpoint = Checkpoint(
            id="checkpoint-empty",
            run_id=run_id,
            node="node_start",
            state={},
            created_at=datetime.now(timezone.utc),
        )

        await repos.checkpoints.save(checkpoint)
        retrieved = await repos.checkpoints.latest(run_id)

        assert retrieved is not None
        assert retrieved.state == {}


class TestToolInvocationRepositoryEdgeCases:
    """Edge case tests for SqlToolInvocationRepository."""

    @pytest.mark.asyncio
    async def test_tool_invocation_with_large_args_and_result(self, repos: SqlRepoBundle) -> None:
        """Test tool invocation with large arguments and results."""
        invocation = ToolInvocation(
            id="invocation-large",
            run_id="run-1",
            server_id="github-mcp",
            tool_name="create_issue",
            args={"description": "X" * 5000, "labels": ["bug"] * 100},
            ok=True,
            result={"data": "Y" * 5000, "nested": {"items": list(range(1000))}},
            risk=RiskLevel.medium,
            created_at=datetime.now(timezone.utc),
        )

        await repos.tool_invocations.append(invocation)

    @pytest.mark.asyncio
    async def test_tool_invocation_with_empty_result(self, repos: SqlRepoBundle) -> None:
        """Test tool invocation with empty result object."""
        invocation = ToolInvocation(
            id="invocation-empty-result",
            run_id="run-1",
            server_id="github-mcp",
            tool_name="list_repos",
            args={"org": "test"},
            ok=False,
            result={},
            risk=RiskLevel.low,
            created_at=datetime.now(timezone.utc),
        )

        await repos.tool_invocations.append(invocation)

    @pytest.mark.asyncio
    async def test_tool_invocation_with_all_risk_levels(self, repos: SqlRepoBundle) -> None:
        """Test tool invocations with all risk levels."""
        risk_levels = [RiskLevel.low, RiskLevel.medium, RiskLevel.high]

        for i, risk in enumerate(risk_levels):
            invocation = ToolInvocation(
                id=f"invocation-risk-{i}",
                run_id="run-1",
                server_id="test-mcp",
                tool_name=f"tool_{i}",
                args={"index": i},
                ok=True,
                result={"success": True},
                risk=risk,
                created_at=datetime.now(timezone.utc),
            )
            await repos.tool_invocations.append(invocation)


class TestUsageRepositoryEdgeCases:
    """Edge case tests for SqlUsageRepository."""

    @pytest.mark.asyncio
    async def test_usage_with_zero_tokens(self, repos: SqlRepoBundle) -> None:
        """Test usage entry with zero token counts."""
        run = AgentRun(
            id="run-zero-tokens",
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
            id="usage-zero",
            run_id="run-zero-tokens",
            provider="openai",
            model="gpt-4",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            created_at=datetime.now(timezone.utc),
        )

        await repos.usage.append(usage)

    @pytest.mark.asyncio
    async def test_usage_with_very_high_token_counts(self, repos: SqlRepoBundle) -> None:
        """Test usage entry with very high token counts."""
        run = AgentRun(
            id="run-high-tokens",
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
            id="usage-high",
            run_id="run-high-tokens",
            provider="openai",
            model="gpt-4",
            prompt_tokens=1000000,
            completion_tokens=500000,
            total_tokens=1500000,
            cost_usd=99999.99,
            created_at=datetime.now(timezone.utc),
        )

        await repos.usage.append(usage)

    @pytest.mark.asyncio
    async def test_usage_list_with_exact_date_boundaries(self, repos: SqlRepoBundle) -> None:
        """Test usage list with exact date boundary matching."""
        tenant_id = "tenant-date-boundary"
        run = AgentRun(
            id="run-date-boundary",
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

        base_time = datetime.now(timezone.utc)

        for i in range(5):
            usage = UsageLedgerEntry(
                id=f"usage-boundary-{i}",
                run_id="run-date-boundary",
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

        filtered = await repos.usage.list(tenant_id, from_date=from_date, to_date=to_date)
        assert len(filtered) == 3

        filtered_exact_start = await repos.usage.list(tenant_id, from_date=from_date)
        assert len(filtered_exact_start) == 4

        filtered_exact_end = await repos.usage.list(tenant_id, to_date=to_date)
        assert len(filtered_exact_end) == 4

    @pytest.mark.asyncio
    async def test_usage_list_with_no_matching_tenant(self, repos: SqlRepoBundle) -> None:
        """Test usage list returns empty for non-existent tenant."""
        result = await repos.usage.list("nonexistent-tenant")
        assert result == []


class TestPolicyRepositoryEdgeCases:
    """Edge case tests for SqlPolicyRepository."""

    @pytest.mark.asyncio
    async def test_policy_with_complex_nested_structure(self, repos: SqlRepoBundle) -> None:
        """Test policy with deeply nested configuration."""
        config = PolicyConfig(
            autonomy_profile="strict",
            version="policy-v1",
        )

        await repos.policies.update("tenant-complex", config)
        retrieved = await repos.policies.get("tenant-complex")

        assert retrieved is not None
        assert isinstance(retrieved, PolicyConfig)
        assert retrieved.autonomy_profile == "strict"
        assert retrieved.version == "policy-v1"

    @pytest.mark.asyncio
    async def test_policy_update_overwrites_completely(self, repos: SqlRepoBundle) -> None:
        """Test that policy update completely overwrites previous config."""
        config1 = PolicyConfig(autonomy_profile="balanced")
        await repos.policies.update("tenant-overwrite", config1)

        config2 = PolicyConfig(autonomy_profile="strict")
        await repos.policies.update("tenant-overwrite", config2)

        retrieved = await repos.policies.get("tenant-overwrite")
        assert retrieved is not None
        assert isinstance(retrieved, PolicyConfig)
        assert retrieved.autonomy_profile == "strict"

    @pytest.mark.asyncio
    async def test_policy_with_special_tenant_ids(self, repos: SqlRepoBundle) -> None:
        """Test policy with special characters in tenant ID."""
        special_tenant_ids = [
            "tenant-with-dashes",
            "tenant_with_underscores",
            "tenant.with.dots",
            "tenant@example.com",
            "tenant/with/slashes",
        ]

        for tenant_id in special_tenant_ids:
            config = PolicyConfig(autonomy_profile="balanced")
            await repos.policies.update(tenant_id, config)

            retrieved = await repos.policies.get(tenant_id)
            assert retrieved is not None
            assert isinstance(retrieved, PolicyConfig)
            assert retrieved.autonomy_profile == "balanced"


class TestRepositoryConcurrency:
    """Concurrency and stress tests for repositories."""

    @pytest.mark.asyncio
    async def test_concurrent_run_creation(self, repos: SqlRepoBundle) -> None:
        """Test creating multiple runs concurrently."""

        async def create_run(i: int):
            run = AgentRun(
                id=f"run-concurrent-{i}",
                tenant_id="tenant-concurrent",
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

        await asyncio.gather(*[create_run(i) for i in range(20)])

        runs = await repos.runs.list(tenant_id="tenant-concurrent")
        assert len(runs) == 20

    @pytest.mark.asyncio
    async def test_concurrent_event_appending(self, repos: SqlRepoBundle) -> None:
        """Test appending events concurrently."""
        run = AgentRun(
            id="run-concurrent-events",
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

        async def append_event(i: int):
            event = AgentEvent(
                id=f"event-concurrent-{i}",
                run_id="run-concurrent-events",
                type=AgentEventType.thought_executed,
                created_at=datetime.now(timezone.utc) + timedelta(milliseconds=i),
                correlation_id=f"corr-{i}",
                payload={"index": i},
            )
            await repos.events.append(event)

        await asyncio.gather(*[append_event(i) for i in range(50)])

        events = await repos.events.list("run-concurrent-events", limit=100)
        assert len(events) == 50

    @pytest.mark.asyncio
    async def test_concurrent_approval_resolution(self, repos: SqlRepoBundle) -> None:
        """Test resolving approvals concurrently."""
        run_id = "run-concurrent-approvals"

        for i in range(10):
            approval = Approval(
                id=f"approval-concurrent-{i}",
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

        async def resolve_approval(i: int):
            await repos.approvals.resolve(
                f"approval-concurrent-{i}", decision=ApprovalDecision.approved.value, decided_by=f"user-{i}"
            )

        await asyncio.gather(*[resolve_approval(i) for i in range(10)])

        pending = await repos.approvals.list(run_id, pending_only=True)
        assert len(pending) == 0

        all_approvals = await repos.approvals.list(run_id, pending_only=False)
        assert len(all_approvals) == 10
        assert all(a.decision == ApprovalDecision.approved for a in all_approvals)

    @pytest.mark.asyncio
    async def test_concurrent_usage_appending(self, repos: SqlRepoBundle) -> None:
        """Test appending usage entries concurrently."""
        run = AgentRun(
            id="run-concurrent-usage",
            tenant_id="tenant-concurrent-usage",
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

        async def append_usage(i: int):
            usage = UsageLedgerEntry(
                id=f"usage-concurrent-{i}",
                run_id="run-concurrent-usage",
                provider="openai",
                model="gpt-4",
                prompt_tokens=100 * (i + 1),
                completion_tokens=50 * (i + 1),
                total_tokens=150 * (i + 1),
                cost_usd=0.005 * (i + 1),
                created_at=datetime.now(timezone.utc) + timedelta(milliseconds=i),
            )
            await repos.usage.append(usage)

        await asyncio.gather(*[append_usage(i) for i in range(30)])

        usage_list = await repos.usage.list("tenant-concurrent-usage")
        assert len(usage_list) == 30
