"""
Agent Runs API Endpoints.

This module provides the primary interface for creating, managing, and retrieving
agent execution runs. It handles the lifecycle of an agent task from start to finish.

Includes:
- Run CRUD operations (create, list, get, resume, cancel)
- Real-time event streaming via Server-Sent Events (SSE)
- Monitoring and tracing via Pydantic AI Logfire for:
  - Agent run lifecycle tracking
  - Performance metrics
  - Error tracking
  - Resource usage monitoring
"""

import asyncio
import json
from datetime import datetime
from typing import List, Union

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from gearmeshing_ai.agent_core.schemas.domain import (
    AgentEvent,
    AgentRun,
    AgentRunStatus,
)
from gearmeshing_ai.core.logging_config import get_logger
from gearmeshing_ai.core.monitoring import (
    log_agent_run,
    log_error,
)
from gearmeshing_ai.server.core.database import get_session
from gearmeshing_ai.server.schemas import RunCreate, RunResume, SSEResponse, KeepAliveEvent, ErrorEvent
from gearmeshing_ai.server.services.chat_persistence import ChatPersistenceService
from gearmeshing_ai.server.services.deps import OrchestratorDep

logger = get_logger(__name__)
router = APIRouter()


def serialize_event(event: Union[SSEResponse, KeepAliveEvent, ErrorEvent, BaseModel]) -> str:
    """
    Serialize a Pydantic model event to JSON string.
    
    Converts Pydantic models to JSON-serializable format with proper datetime handling.
    
    Args:
        event: Pydantic model (SSEResponse, KeepAliveEvent, ErrorEvent, or other BaseModel)
        
    Returns:
        JSON string representation of the event
    """
    try:
        # Use Pydantic's model_dump_json for proper serialization
        if isinstance(event, BaseModel):
            return event.model_dump_json()
        else:
            # Fallback for non-Pydantic objects
            return json.dumps(event)
    except Exception as e:
        logger.error(f"Failed to serialize event: {e}", exc_info=True)
        # Return error event instead of crashing
        error_event = ErrorEvent(error="Failed to serialize event", details=str(e))
        return error_event.model_dump_json()


@router.post(
    "/",
    response_model=AgentRun,
    status_code=201,
    summary="Create Agent Run",
    description="Initiates a new agent run with the specified objective, role, and configuration.",
    response_description="The created agent run object.",
)
async def create_run(run_in: RunCreate, orchestrator: OrchestratorDep, background_tasks: BackgroundTasks):
    """
    Create a new agent run.

    This endpoint initializes a new agent run session. It accepts the user's objective,
    tenant identification, and optional configuration for the agent's role and autonomy.

    - **objective**: The goal for the agent.
    - **tenant_id**: The tenant identifier.
    - **role**: The agent role (optional, defaults to 'planner').
    - **autonomy_profile**: Autonomy level (optional, defaults to 'balanced').
    - **input**: Initial input payload (optional).
    """
    logger.info(f"Creating new run for tenant: {run_in.tenant_id}, objective: {run_in.objective}")

    if not run_in.role:
        run_in.role = "planner"
        logger.debug(f"Role not specified, defaulting to: {run_in.role}")

    # Map API schema to Domain Schema
    # Note: AgentRun domain object does not strictly have 'input_payload' in the constructor
    # if it wasn't added to the domain class. The AgentRunTable has it.
    # The domain AgentRun class in agent_core/schemas/domain.py does NOT have input_payload.
    # However, for now we will just pass what matches.

    from gearmeshing_ai.agent_core.schemas.domain import AutonomyProfile

    autonomy = AutonomyProfile(run_in.autonomy_profile) if run_in.autonomy_profile else AutonomyProfile.balanced
    logger.debug(f"Autonomy profile set to: {autonomy}")

    run_domain = AgentRun(
        tenant_id=run_in.tenant_id,
        role=run_in.role,
        objective=run_in.objective,
        autonomy_profile=autonomy,
        status=AgentRunStatus.running,
    )

    # Delegate to Orchestrator
    # Note: We are using await here. Ideally run should be backgrounded if it takes time.
    # But Orchestrator.create_run calls AgentService.run which plans + starts.
    # Planning might take a few seconds.
    try:
        logger.debug("Delegating to orchestrator for run creation")
        created_run = await orchestrator.create_run(run_domain)
        logger.info(f"Run created successfully: {created_run.id}")

        # Log to Logfire for monitoring
        log_agent_run(
            run_id=created_run.id,
            tenant_id=run_in.tenant_id,
            objective=run_in.objective,
            role=run_in.role,
        )

        return created_run
    except Exception as e:
        logger.error(f"Failed to create run for tenant {run_in.tenant_id}: {e}", exc_info=True)

        # Log error to Logfire for monitoring
        log_error(
            error_type="RunCreationError",
            error_message=str(e),
            context={
                "tenant_id": run_in.tenant_id,
                "objective": run_in.objective,
                "role": run_in.role,
            },
        )

        raise


@router.get(
    "/",
    response_model=List[AgentRun],
    summary="List Agent Runs",
    description="Retrieve a list of agent runs, optionally filtered by tenant.",
    response_description="A list of agent runs.",
)
async def list_runs(orchestrator: OrchestratorDep, tenant_id: str | None = None, limit: int = 100, offset: int = 0):
    """
    List agent runs.

    Returns a paginated list of runs. If `tenant_id` is provided, filters the results
    to only show runs belonging to that tenant.
    """
    return await orchestrator.list_runs(tenant_id=tenant_id, limit=limit, offset=offset)


@router.get(
    "/{run_id}",
    response_model=AgentRun,
    summary="Get Run Details",
    description="Retrieve detailed information about a specific agent run by its ID.",
    response_description="The agent run object.",
    responses={404: {"description": "Run not found"}},
)
async def get_run(run_id: str, orchestrator: OrchestratorDep):
    """
    Get run details.

    Fetches the current state and metadata of a specific run.
    """
    run = await orchestrator.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.post(
    "/{run_id}/resume",
    response_model=AgentRun,
    summary="Resume Paused Run",
    description="Resume a run that is paused waiting for approval.",
    response_description="The updated agent run object.",
    responses={404: {"description": "Run not found"}},
)
async def resume_run(run_id: str, resume_in: RunResume, orchestrator: OrchestratorDep):
    """
    Resume a paused run.

    This endpoint signals the agent to continue execution after a pause (e.g., for approval).
    """
    # Logic is now handled via approval submission usually.
    # If explicit resume is needed without approval ID, it implies something else.
    # However, AgentService.resume requires an approval_id.
    # This endpoint seems slightly redundant if we assume resume happens on approval.
    # But maybe it is for resuming from a generic pause?
    # For now, let's just return the run status.
    run = await orchestrator.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.post(
    "/{run_id}/cancel",
    response_model=AgentRun,
    summary="Cancel Run",
    description="Cancel an active agent run.",
    response_description="The cancelled agent run object.",
    responses={404: {"description": "Run not found"}},
)
async def cancel_run(run_id: str, orchestrator: OrchestratorDep):
    """
    Cancel an active run.

    Stops the execution of the agent run and marks it as cancelled.
    """
    run = await orchestrator.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    # Update run status to cancelled
    await orchestrator.cancel_run(run_id)

    # Fetch and return updated run
    updated_run = await orchestrator.get_run(run_id)
    return updated_run


@router.get(
    "/{run_id}/events",
    response_model=List[AgentEvent],
    summary="List Run Events",
    description="Retrieve the event history for a specific run.",
    response_description="A list of agent events.",
)
async def list_run_events(run_id: str, orchestrator: OrchestratorDep, limit: int = 100):
    """
    List run events.

    Returns the timeline of events (thoughts, actions, tool outputs) for the specified run.
    Useful for debugging and auditing the agent's behavior.
    """
    return await orchestrator.get_run_events(run_id, limit=limit)


@router.get(
    "/{run_id}/event",
    summary="Stream Run Events",
    description="Subscribe to a Server-Sent Events (SSE) stream for a specific run.",
    response_description="A stream of event objects.",
    responses={
        200: {
            "description": "SSE stream established",
            "content": {
                "text/event-stream": {
                    "example": "data: {\"comment\": \"keep-alive\"}\n\n"
                }
            }
        }
    }
)
async def stream_run_events(
    run_id: str,
    request: Request,
    orchestrator: OrchestratorDep,
    db_session: AsyncSession = Depends(get_session),
):
    """
    Stream events for a specific run via Server-Sent Events (SSE).

    Provides a real-time feed of events (like 'thought', 'tool_invoked', 'run_completed')
    for the given run ID. This allows clients to follow the agent's progress live.

    The stream emits JSON-formatted events. Each event payload mirrors the `AgentEvent` schema.
    
    **Path Parameters:**
    - `run_id`: The unique identifier of the run to stream events from
    
    **Response:**
    - Streams JSON-formatted event objects in real-time
    - Connection maintains until client disconnects or run completes
    - Chat history is automatically persisted on the backend
    """
    logger.info(f"Starting event stream for run: {run_id}")
    
    async def event_generator():
        chat_service = ChatPersistenceService(db_session)
        chat_session = None
        user_message_persisted = False
        
        async def persist_event_to_chat(run_id: str, display_text: str, event_type: str) -> None:
            """Callback to persist events to chat history."""
            nonlocal chat_session, user_message_persisted
            
            # Initialize chat session on first event
            if chat_session is None:
                try:
                    run = await orchestrator.get_run(run_id)
                    if run:
                        chat_session = await chat_service.get_or_create_session(
                            run_id=run_id,
                            tenant_id=run.tenant_id,
                            agent_role=run.role,
                            title=f"Chat - {run.role}",
                            description=f"Chat history for run {run_id}",
                        )
                        logger.debug(f"Initialized chat session {chat_session.id} for run {run_id}")
                        
                        # Persist user's initial objective as the first message
                        if not user_message_persisted and run.objective:
                            try:
                                await chat_service.add_user_message(
                                    session_id=chat_session.id,
                                    content=run.objective,
                                    metadata={"type": "initial_objective", "run_id": run_id},
                                )
                                user_message_persisted = True
                                logger.debug(f"Persisted user objective to chat session {chat_session.id}")
                            except Exception as e:
                                logger.warning(f"Failed to persist user objective to chat session: {e}")
                except Exception as e:
                    logger.warning(f"Failed to initialize chat session for run {run_id}: {e}")
                    return
            
            # Persist message to chat session
            if chat_session:
                try:
                    await chat_service.add_agent_message(
                        session_id=chat_session.id,
                        content=display_text,
                        event_type=event_type,
                    )
                except Exception as e:
                    logger.warning(f"Failed to persist event to chat session: {e}")
        
        try:
            async for event in orchestrator.stream_events(run_id, on_event_persisted=persist_event_to_chat):
                if await request.is_disconnected():
                    logger.info(f"Client disconnected from stream for run: {run_id}")
                    break
                
                # Format event as JSON string for SSE, handling datetime serialization
                yield serialize_event(event)
        except Exception as e:
            logger.error(f"Error in event stream for run {run_id}: {e}", exc_info=True)
            yield serialize_event({"error": str(e)})

    return EventSourceResponse(event_generator())
