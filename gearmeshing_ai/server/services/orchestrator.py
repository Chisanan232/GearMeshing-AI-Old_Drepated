from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Union

from gearmeshing_ai.agent_core.factory import build_default_registry
from gearmeshing_ai.agent_core.planning.planner import StructuredPlanner
from gearmeshing_ai.agent_core.policy.models import PolicyConfig
from gearmeshing_ai.agent_core.policy.provider import DatabasePolicyProvider
from gearmeshing_ai.agent_core.repos.interfaces import EventRepository
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
from gearmeshing_ai.server.core.database import async_session_maker, checkpointer_pool
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
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

logger = get_logger(__name__)


class BroadcastingEventRepository:
    """
    Wrapper around EventRepository that broadcasts events to active listeners.
    """

    def __init__(self, inner: EventRepository, listeners: Dict[str, List[asyncio.Queue[AgentEvent]]]):
        self.inner = inner
        self.listeners = listeners

    async def append(self, event: AgentEvent) -> None:
        # Persist to DB first
        await self.inner.append(event)

        # Broadcast to active listeners
        if event.run_id in self.listeners:
            for queue in self.listeners[event.run_id]:
                await queue.put(event)

    async def list(self, run_id: str, limit: int = 100) -> list[AgentEvent]:
        return await self.inner.list(run_id, limit)


class OrchestratorService:
    """
    Service layer for managing agent runs and orchestrating execution.
    Wraps the core AgentService and Repositories to provide business logic for the API.
    """

    def __init__(self) -> None:
        # Manage active event listeners: run_id -> List[Queue]
        self.event_listeners: Dict[str, List[asyncio.Queue[AgentEvent]]] = {}
        # Keep references to background tasks to prevent premature GC
        self.background_tasks: set[asyncio.Task] = set()

        # Build base repositories
        base_repos = build_sql_repos(session_factory=async_session_maker)

        # Store checkpointer pool for lazy initialization
        # AsyncPostgresSaver requires an event loop at instantiation, so we defer creation
        # until it's needed in an async context (in _build_engine_deps)
        self.checkpointer_pool = checkpointer_pool
        self._checkpointer: Optional[AsyncPostgresSaver] = None

        # Wrap event repository for broadcasting
        self.event_repo_wrapper = BroadcastingEventRepository(base_repos.events, self.event_listeners)

        # Reconstruct SqlRepoBundle with the wrapper
        # We use the wrapper for 'events' but keep others as is
        self.repos: SqlRepoBundle = SqlRepoBundle(
            runs=base_repos.runs,
            events=self.event_repo_wrapper,  # type: ignore
            approvals=base_repos.approvals,
            checkpoints=base_repos.checkpoints,
            tool_invocations=base_repos.tool_invocations,
            usage=base_repos.usage,
            policies=base_repos.policies,
        )

        # Defer building engine_deps to avoid event loop issues during init
        self._engine_deps_cached: Optional[AgentServiceDeps] = None
        self.deps: Optional[AgentServiceDeps] = None  # Will be set lazily
        self._agent_service_cached: Optional[AgentService] = None

        # Create default policy config
        default_policy = PolicyConfig()

        # Initialize DatabasePolicyProvider for tenant-specific policies
        # This loads policies from the persistence layer based on tenant_id
        self.policy_provider = DatabasePolicyProvider(
            policy_repository=self.repos.policies,
            default=default_policy,
        )
        
        self._default_policy = default_policy

    @property
    def checkpointer(self) -> AsyncPostgresSaver:
        """
        Lazily create and return the AsyncPostgresSaver.
        
        This is necessary because AsyncPostgresSaver requires an event loop at instantiation,
        but OrchestratorService.__init__() is called from a synchronous dependency injection context.
        The checkpointer is only created when first accessed in an async context.
        """
        if self._checkpointer is None:
            self._checkpointer = AsyncPostgresSaver(self.checkpointer_pool)
        return self._checkpointer

    def _build_engine_deps(self):
        from gearmeshing_ai.agent_core.runtime import EngineDeps

        return EngineDeps(
            runs=self.repos.runs,
            events=self.repos.events,  # This uses the broadcasting wrapper
            approvals=self.repos.approvals,
            checkpoints=self.repos.checkpoints,
            tool_invocations=self.repos.tool_invocations,
            usage=self.repos.usage,
            capabilities=build_default_registry(),
            checkpointer=self.checkpointer,
        )

    def _get_engine_deps(self) -> AgentServiceDeps:
        """Lazily build and cache engine dependencies."""
        if self._engine_deps_cached is None:
            self._engine_deps_cached = AgentServiceDeps(
                engine_deps=self._build_engine_deps(),
                planner=StructuredPlanner(),
            )
        assert self._engine_deps_cached is not None
        return self._engine_deps_cached

    def _get_agent_service(self) -> AgentService:
        """Lazily build and cache agent service."""
        if self._agent_service_cached is None:
            self._agent_service_cached = AgentService(
                policy_config=self._default_policy,
                deps=self._get_engine_deps(),
                policy_provider=self.policy_provider,
            )
        assert self._agent_service_cached is not None
        return self._agent_service_cached

    async def create_run(self, run: AgentRun) -> AgentRun:
        """
        Create a new agent run in PENDING status (in-memory only).
        
        Note: The run is NOT persisted to the database here. Persistence happens
        in engine.start_run() when the workflow execution begins. This follows the
        async-first pattern where the API layer creates the run object in memory,
        passes it to the background worker, and the engine persists it when ready.
        """

        # Set status to pending initially
        run.status = AgentRunStatus.pending

        logger.info(f"Run {run.id} created with status PENDING")
        return run

    async def execute_workflow(self, run: AgentRun) -> None:
        """
        Execute the agent workflow in the background.
        
        The run object is passed directly from the API layer, avoiding the need to
        fetch it from the database. The engine.start_run() will persist the run
        to the database when execution begins.
        """
        run_id = run.id
        try:
            logger.info(f"Starting execution for run {run_id}")

            # Ensure status is PENDING before execution
            run.status = AgentRunStatus.pending

            # Execute the workflow - engine.start_run() will persist the run
            await self._get_agent_service().run(run=run)
            logger.info(f"Execution finished for run {run_id}")

        except Exception as e:
            logger.error(f"Error executing workflow for run {run_id}: {e}", exc_info=True)
            # Update status to failed in database
            await self.repos.runs.update_status(run_id, status=AgentRunStatus.failed.value)
            await self.repos.events.append(
                AgentEvent(run_id=run_id, type=AgentEventType.run_failed, payload={"error": str(e)})
            )

    async def execute_resume(self, run_id: str, approval_id: str) -> None:
        """
        Resume the agent workflow in the background.
        """
        try:
            logger.info(f"Resuming execution for run {run_id} (approval {approval_id})")
            
            # Update status to running to reflect active processing
            await self.repos.runs.update_status(run_id, status=AgentRunStatus.running.value)
            
            await self._get_agent_service().resume(run_id=run_id, approval_id=approval_id)
            logger.info(f"Resume finished for run {run_id}")
        except Exception as e:
            logger.error(f"Error resuming workflow for run {run_id}: {e}", exc_info=True)
            await self.repos.runs.update_status(run_id, status=AgentRunStatus.failed.value)
            await self.repos.events.append(
                AgentEvent(run_id=run_id, type=AgentEventType.run_failed, payload={"error": str(e)})
            )

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

        # Note: We do NOT trigger resumption here anymore.
        # The API controller is responsible for scheduling execute_resume via BackgroundTasks
        # if the decision is 'approved'.

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

    async def stream_events(
        self,
        run_id: str,
        on_event_persisted: Optional[Callable[[str, str, str], Awaitable[None]]] = None,
    ) -> AsyncGenerator[Union[SSEResponse, KeepAliveEvent, ErrorEvent], None]:
        """
        Yields events for a run using a combination of historical DB fetch and real-time queue subscription.
        """
        queue: asyncio.Queue = asyncio.Queue()
        if run_id not in self.event_listeners:
            self.event_listeners[run_id] = []
        self.event_listeners[run_id].append(queue)

        try:
            # 1. Fetch historical events from DB
            try:
                history = await self.repos.events.list(run_id=run_id, limit=1000)
                seen_ids = set()

                for event in history:
                    seen_ids.add(event.id)
                    sse_event = await self._enrich_event_for_sse(event)
                    yield sse_event

                    # Chat persistence for history is tricky - do we re-persist?
                    # Usually history is already persisted in chat if it happened.
                    # Only new real-time events need persistence if they weren't caught.
                    # But `on_event_persisted` is likely for the chat UI sync.
                    # If we assume chat service is separate, we might not need to call it for history.
            except Exception as e:
                logger.error(f"Error fetching history for run {run_id}: {e}")
                yield ErrorEvent(error=str(e))
                # If history fetch fails, we might still want to try streaming?
                # Or just stop? The previous implementation would just yield error and retry loop.
                # Here we are before the loop. If history fails, we probably shouldn't proceed to real-time
                # without signaling. We yielded ErrorEvent.
                # Let's continue to streaming to attempt recovery or at least keep connection open?
                # But usually DB error is fatal for that request.
                # Let's just proceed to queue check, maybe DB recovers?

            # 2. Stream real-time events from queue
            keep_alive_interval = 15

            while True:
                try:
                    # Wait for event with timeout for keep-alive
                    event = await asyncio.wait_for(queue.get(), timeout=keep_alive_interval)

                    if event.id in seen_ids:
                        continue
                    seen_ids.add(event.id)

                    sse_event = await self._enrich_event_for_sse(event)

                    # Persist new events to chat
                    if on_event_persisted and sse_event.data:
                        display_text = self._format_event_for_chat(sse_event.data)
                        if display_text:
                            try:
                                await on_event_persisted(
                                    run_id,
                                    display_text,
                                    sse_event.data.type,
                                )
                            except Exception as e:
                                logger.warning(f"Failed to persist event to chat: {e}")

                    yield sse_event

                    # Check for terminal events
                    if event.type in (
                        AgentEventType.run_completed,
                        AgentEventType.run_failed,
                        AgentEventType.run_cancelled,
                    ):
                        # Run ended, close stream after sending the terminal event
                        break

                except asyncio.TimeoutError:
                    yield KeepAliveEvent()

                except asyncio.CancelledError:
                    raise

        finally:
            # Cleanup listener
            if run_id in self.event_listeners:
                self.event_listeners[run_id].remove(queue)
                if not self.event_listeners[run_id]:
                    del self.event_listeners[run_id]

    async def _enrich_event_for_sse(self, event: AgentEvent) -> SSEResponse:
        """
        Enrich an event with additional context for SSE streaming.

        Returns a Pydantic SSEResponse model that is JSON-serializable.
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

    def _format_event_for_chat(self, event_data: SSEEventData) -> str:
        """
        Format SSE event data as display text for chat persistence.
        """
        category = event_data.category
        display_text = ""

        # Skip thinking events - they're internal and not user-facing
        if category == "thinking" or category == "thinking_output":
            return ""

        # Format operation events
        if category == "operation" and event_data.operation:
            op = event_data.operation
            status_icon = "✓" if op.status == "success" else "✗"
            display_text = f"{status_icon} Operation: {op.capability} ({op.status})"
            if op.result:
                display_text += f"\n  Result: {str(op.result)[:200]}"

        # Format tool execution events
        elif category == "tool_execution" and event_data.tool_execution:
            tool = event_data.tool_execution
            status_icon = "✓" if tool.ok else "✗"
            display_text = f"{status_icon} Tool: {tool.tool_name} ({tool.server_id})"
            if tool.result:
                display_text += f"\n  Result: {str(tool.result)[:200]}"

        # Format approval request events
        elif category == "approval" and event_data.approval_request:
            approval = event_data.approval_request
            display_text = f"⚠️ Approval Required: {approval.capability}"
            if approval.reason:
                display_text += f"\n  Reason: {approval.reason}"
            if approval.risk:
                display_text += f"\n  Risk Level: {approval.risk}"

        # Format approval resolution events
        elif category == "approval" and event_data.approval_resolution:
            resolution = event_data.approval_resolution
            decision_icon = "✓" if resolution.decision == "approved" else "✗"
            display_text = (
                f"{decision_icon} Approval {resolution.decision.upper() if resolution.decision else 'UNKNOWN'}"
            )
            if resolution.decided_by:
                display_text += f" (by {resolution.decided_by})"

        # Format run lifecycle events
        elif category == "run_lifecycle":
            if event_data.run_start:
                display_text = "▶️ Run Started"
            elif event_data.run_completion:
                display_text = "✓ Run Completed Successfully"
            elif event_data.run_failure:
                display_text = f"✗ Run Failed: {event_data.run_failure.error}"

        return display_text

    async def list_available_roles(self) -> List[str]:
        from gearmeshing_ai.agent_core.schemas.domain import AgentRole

        return [role.value for role in AgentRole]


# Global singleton
_orchestrator: Optional[OrchestratorService] = None


def get_orchestrator() -> OrchestratorService:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorService()
    return _orchestrator
