from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from gearmeshing_ai.agent_core.factory import build_default_registry
from gearmeshing_ai.agent_core.planning.planner import StructuredPlanner
from gearmeshing_ai.agent_core.policy.models import PolicyConfig
from gearmeshing_ai.agent_core.policy.provider import DatabasePolicyProvider
from gearmeshing_ai.agent_core.repos.sql import SqlRepoBundle, build_sql_repos
from gearmeshing_ai.agent_core.schemas.domain import (
    AgentEvent,
    AgentEventType,
    AgentRun,
    AgentRunStatus,
    Approval,
    UsageLedgerEntry,
)
from gearmeshing_ai.agent_core.service import AgentService, AgentServiceDeps
from gearmeshing_ai.core.logging_config import get_logger
from gearmeshing_ai.server.core.database import async_session_maker
from gearmeshing_ai.server.schemas import (
    ApprovalRequestData,
    ApprovalResolutionData,
    ErrorEvent,
    KeepAliveEvent,
    OperationData,
    RunCompletionData,
    RunFailureData,
    RunStartData,
    SSEEventData,
    SSEResponse,
    ThinkingData,
    ThinkingOutputData,
    ToolExecutionData,
)

logger = get_logger(__name__)


class OrchestratorService:
    """
    Service layer for managing agent runs and orchestrating execution.
    Wraps the core AgentService and Repositories to provide business logic for the API.
    """

    def __init__(self) -> None:
        # Build repositories using the global session factory
        self.repos: SqlRepoBundle = build_sql_repos(session_factory=async_session_maker)

        # Build runtime dependencies (shared across services)
        self.deps = AgentServiceDeps(
            engine_deps=self._build_engine_deps(),
            planner=StructuredPlanner(),
        )

        # Create default policy config
        default_policy = PolicyConfig()

        # Initialize DatabasePolicyProvider for tenant-specific policies
        # This loads policies from the persistence layer based on tenant_id
        self.policy_provider = DatabasePolicyProvider(
            policy_repository=self.repos.policies,
            default=default_policy,
        )

        # Initialize the core AgentService with policy provider
        # This enables dynamic, tenant-aware policy resolution per run
        self.agent_service = AgentService(
            policy_config=default_policy,
            deps=self.deps,
            policy_provider=self.policy_provider,
        )

    def _build_engine_deps(self):
        from gearmeshing_ai.agent_core.runtime import EngineDeps

        return EngineDeps(
            runs=self.repos.runs,
            events=self.repos.events,
            approvals=self.repos.approvals,
            checkpoints=self.repos.checkpoints,
            tool_invocations=self.repos.tool_invocations,
            usage=self.repos.usage,
            capabilities=build_default_registry(),
        )

    async def create_run(self, run: AgentRun) -> AgentRun:
        """Create and start a new agent run."""
        # 1. Create the run record (AgentService.run does this, but we might want to return the object immediately)
        # AgentService.run returns the ID.
        # Let's use AgentService.run which orchestrates the initial plan and start.
        # But AgentService.run expects the Run object to be passed in.

        # Persist initial state if AgentService doesn't do it before returning?
        # AgentService.run calls engine.start_run -> persists run.

        await self.agent_service.run(run=run)

        # Fetch back to return full object
        created = await self.repos.runs.get(run.id)
        if not created:
            raise RuntimeError(f"Run {run.id} creation failed")
        return created

    async def list_runs(self, tenant_id: str | None = None, limit: int = 100, offset: int = 0) -> List[AgentRun]:
        """List agent runs with optional filtering."""
        return await self.repos.runs.list(tenant_id=tenant_id, limit=limit, offset=offset)

    async def get_run(self, run_id: str) -> Optional[AgentRun]:
        return await self.repos.runs.get(run_id)

    async def get_run_events(self, run_id: str, limit: int = 100) -> List[AgentEvent]:
        return await self.repos.events.list(run_id=run_id, limit=limit)

    async def get_pending_approvals(self, run_id: str) -> List[Approval]:
        return await self.repos.approvals.list(run_id=run_id, pending_only=True)

    async def submit_approval(
        self, run_id: str, approval_id: str, decision: str, note: str | None, decided_by: str | None
    ) -> Approval:
        await self.repos.approvals.resolve(approval_id, decision=decision, decided_by=decided_by)
        # Resume the run if approved
        # In a real event-driven system, this might trigger an event.
        # Here we manually call resume.
        if decision == "approved":
            # Resume in background? Or await?
            # Await for now to ensure it starts.
            await self.agent_service.resume(run_id=run_id, approval_id=approval_id)

        approval = await self.repos.approvals.get(approval_id)
        if not approval:
            raise RuntimeError(f"Approval {approval_id} not found after resolution")
        return approval

    async def list_usage(
        self, tenant_id: str, from_date: Optional[datetime] = None, to_date: Optional[datetime] = None
    ) -> List[UsageLedgerEntry]:
        return await self.repos.usage.list(tenant_id=tenant_id, from_date=from_date, to_date=to_date)

    async def get_policy(self, tenant_id: str) -> Optional[PolicyConfig]:
        return await self.repos.policies.get(tenant_id=tenant_id)

    async def update_policy(self, tenant_id: str, config: PolicyConfig) -> PolicyConfig:
        await self.repos.policies.update(tenant_id=tenant_id, config=config)
        return config

    async def cancel_run(self, run_id: str) -> None:
        """Cancel an active run by updating its status to cancelled."""
        await self.repos.runs.update_status(run_id=run_id, status=AgentRunStatus.cancelled.value)

    async def stream_events(self, run_id: str) -> AsyncGenerator[Union[SSEResponse, KeepAliveEvent, ErrorEvent], None]:
        """
        Yields JSON-serializable events for a run as they happen (or via polling).
        Polls the event repository periodically and yields new events.

        Events are enriched with thinking details and operation results:
        - Thought events include the thinking process and model output
        - Action events include capability execution results and policy decisions
        - All events are persisted to database for audit trail

        Returns Pydantic models that are JSON-serializable.
        """
        last_event_timestamp: Optional[datetime] = None
        poll_interval = 0.5
        max_idle_cycles = 120  # 60 seconds of idle before closing
        idle_cycles = 0

        while True:
            try:
                # Fetch all events for this run
                events = await self.repos.events.list(run_id=run_id, limit=1000)

                # Filter to events we haven't seen yet (by timestamp)
                if last_event_timestamp is None:
                    new_events = events
                else:
                    new_events = [e for e in events if hasattr(e, "created_at") and e.created_at > last_event_timestamp]

                if new_events:
                    idle_cycles = 0
                    # Update last_event_timestamp
                    if hasattr(new_events[-1], "created_at"):
                        last_event_timestamp = new_events[-1].created_at

                    # Yield each new event with enriched data
                    for event in new_events:
                        sse_event = await self._enrich_event_for_sse(event)
                        yield sse_event
                else:
                    # No new events, send keep-alive
                    idle_cycles += 1
                    yield KeepAliveEvent()

                    # Close stream if idle for too long
                    if idle_cycles >= max_idle_cycles:
                        break

                await asyncio.sleep(poll_interval)

            except Exception as e:
                logger.error(f"Error streaming events for run {run_id}: {e}")
                yield ErrorEvent(error=str(e))
                await asyncio.sleep(poll_interval)

    async def _enrich_event_for_sse(self, event: AgentEvent) -> SSEResponse:
        """
        Enrich an event with additional context for SSE streaming.

        Returns a Pydantic SSEResponse model that is JSON-serializable.

        Includes:
        - Thinking details (model output, reasoning)
        - Operation results (tool outputs, capability results)
        - Policy decisions (approval status, risk classification)
        - Timestamps and correlation IDs
        """
        event_dict = event.model_dump() if hasattr(event, "model_dump") else event.__dict__

        # Extract base event fields
        event_id = event_dict.get("id")
        event_type = event.type
        created_at = event_dict.get("created_at", datetime.now(timezone.utc))
        run_id = event_dict.get("run_id")
        payload = event_dict.get("payload", {})
        category = "other"

        # Initialize enriched fields
        thinking = None
        thinking_output = None
        operation = None
        tool_execution = None
        approval_request = None
        approval_resolution = None
        run_start = None
        run_completion = None
        run_failure = None

        # Enrich based on event type
        if event_type == AgentEventType.thought_executed:
            category = "thinking"
            thinking = ThinkingData(
                thought=payload.get("thought"),
                idx=payload.get("idx"),
                timestamp=created_at,
            )

        elif event_type == AgentEventType.artifact_created:
            if payload.get("kind") == "thought":
                category = "thinking_output"
                thinking_output = ThinkingOutputData(
                    thought=payload.get("thought"),
                    data=payload.get("data"),
                    output=payload.get("output"),
                    system_prompt_key=payload.get("system_prompt_key"),
                    timestamp=created_at,
                )

        elif event_type == AgentEventType.capability_executed:
            category = "operation"
            operation = OperationData(
                capability=payload.get("capability"),
                status="success" if payload.get("ok") else "failed",
                result=payload.get("result"),
                timestamp=created_at,
            )

        elif event_type == AgentEventType.tool_invoked:
            category = "tool_execution"
            tool_execution = ToolExecutionData(
                server_id=payload.get("server_id"),
                tool_name=payload.get("tool_name"),
                args=payload.get("args"),
                result=payload.get("result"),
                ok=payload.get("ok"),
                risk=payload.get("risk"),
                timestamp=created_at,
            )

        elif event_type == AgentEventType.approval_requested:
            category = "approval"
            approval_request = ApprovalRequestData(
                capability=payload.get("capability"),
                risk=payload.get("risk"),
                reason=payload.get("reason"),
                timestamp=created_at,
            )

        elif event_type == AgentEventType.approval_resolved:
            category = "approval"
            approval_resolution = ApprovalResolutionData(
                decision=payload.get("decision"),
                decided_by=payload.get("decided_by"),
                timestamp=created_at,
            )

        elif event_type == AgentEventType.run_started:
            category = "run_lifecycle"
            run_start = RunStartData(
                run_id=run_id,
                timestamp=created_at,
            )

        elif event_type == AgentEventType.run_completed:
            category = "run_lifecycle"
            run_completion = RunCompletionData(
                status="succeeded",
                timestamp=created_at,
            )

        elif event_type == AgentEventType.run_failed:
            category = "run_lifecycle"
            run_failure = RunFailureData(
                error=payload.get("error"),
                timestamp=created_at,
            )

        # Create SSEEventData with all enriched fields
        sse_event_data = SSEEventData(
            id=event_id,
            type=str(event_type.value),
            category=category,
            created_at=created_at,
            run_id=run_id,
            payload=payload,
            thinking=thinking,
            thinking_output=thinking_output,
            operation=operation,
            tool_execution=tool_execution,
            approval_request=approval_request,
            approval_resolution=approval_resolution,
            run_start=run_start,
            run_completion=run_completion,
            run_failure=run_failure,
        )

        return SSEResponse(data=sse_event_data)

    async def list_available_roles(self) -> List[str]:
        from gearmeshing_ai.agent_core.schemas.domain import AgentRole

        return [role.value for role in AgentRole]

    async def override_role_prompt(self, tenant_id: str, role: str, prompt: str) -> Dict[str, Any]:
        # Placeholder for prompt provider integration
        return {"status": "updated", "role": role, "version": "v_custom_1", "tenant_id": tenant_id}


# Global singleton
_orchestrator: Optional[OrchestratorService] = None


def get_orchestrator() -> OrchestratorService:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorService()
    return _orchestrator
