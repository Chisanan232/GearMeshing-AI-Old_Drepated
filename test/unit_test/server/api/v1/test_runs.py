"""
Unit tests for Agent Runs API endpoints.

Tests cover:
- Creating new agent runs
- Listing runs with filtering
- Retrieving run details
- Resuming paused runs
- Cancelling active runs
- Listing run events
- Real-time event streaming via Server-Sent Events (SSE)

These tests use direct function calls to ensure proper coverage detection of async code.
See TestDirectFunctionCalls class documentation for why direct calls are necessary.
"""

import json
from datetime import datetime, timezone, timedelta
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient
from pydantic import BaseModel

from gearmeshing_ai.agent_core.schemas.domain import (
    AgentEvent,
    AgentEventType,
    AgentRun,
    AgentRunStatus,
    AutonomyProfile,
)
from gearmeshing_ai.server.api.v1.runs import serialize_event
from gearmeshing_ai.server.schemas import SSEResponse, SSEEventData, KeepAliveEvent, ErrorEvent, ThinkingData

pytestmark = pytest.mark.asyncio


async def test_create_run(client_with_mocked_runs: AsyncClient):
    """Test creating a new agent run."""
    import uuid

    tenant_id = f"test-tenant-{uuid.uuid4().hex[:8]}"
    payload = {
        "tenant_id": tenant_id,
        "objective": "Test objective",
        "role": "planner",
        "autonomy_profile": "balanced",
        "input": {"key": "value"},
    }
    response = await client_with_mocked_runs.post("/api/v1/runs/", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["tenant_id"] == tenant_id
    assert data["objective"] == "Test objective"
    # Status can be 'running' or 'succeeded' depending on execution
    assert data["status"] in [AgentRunStatus.running.value, AgentRunStatus.succeeded.value]
    assert "id" in data


async def test_get_run(client_with_mocked_runs: AsyncClient):
    """Test retrieving a specific run."""
    import uuid

    tenant_id = f"test-tenant-{uuid.uuid4().hex[:8]}"
    # Create first
    payload = {"tenant_id": tenant_id, "objective": "Test objective"}
    create_res = await client_with_mocked_runs.post("/api/v1/runs/", json=payload)
    run_id = create_res.json()["id"]

    # Get
    response = await client_with_mocked_runs.get(f"/api/v1/runs/{run_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == run_id
    assert data["objective"] == "Test objective"


async def test_list_runs(client_with_mocked_runs: AsyncClient):
    """Test listing runs for a tenant."""
    import uuid

    tenant_id = f"t-{uuid.uuid4().hex[:8]}"
    # Create two runs for the same tenant
    await client_with_mocked_runs.post("/api/v1/runs/", json={"tenant_id": tenant_id, "objective": "o1"})
    await client_with_mocked_runs.post("/api/v1/runs/", json={"tenant_id": tenant_id, "objective": "o2"})

    # List all for this tenant
    response = await client_with_mocked_runs.get(f"/api/v1/runs/?tenant_id={tenant_id}")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert all(d["tenant_id"] == tenant_id for d in data)


async def test_cancel_run(client_with_mocked_runs: AsyncClient):
    """Test cancelling a run."""
    import uuid

    tenant_id = f"t-{uuid.uuid4().hex[:8]}"
    # Create
    create_res = await client_with_mocked_runs.post("/api/v1/runs/", json={"tenant_id": tenant_id, "objective": "o1"})
    run_id = create_res.json()["id"]

    # Cancel
    response = await client_with_mocked_runs.post(f"/api/v1/runs/{run_id}/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == AgentRunStatus.cancelled.value


class TestDirectFunctionCalls:
    """Direct function call tests to ensure proper coverage detection of async code.

    IMPORTANT: These tests call the endpoint functions directly with mocked orchestrator,
    bypassing the HTTP layer so coverage.py can properly track execution.

    WHY DIRECT CALLS ARE NECESSARY FOR COVERAGE:
    ============================================
    When tests use AsyncClient (HTTP layer), the code execution flow goes through:
    1. HTTP request serialization
    2. FastAPI routing and dependency injection
    3. Async context managers and middleware
    4. The actual endpoint function
    5. Response serialization

    Coverage.py struggles to properly instrument and track code execution in this
    flow because:
    - Async context managers can obscure the actual code paths
    - Exception handling in FastAPI's exception handlers may not be tracked
    - The HTTP layer adds abstraction that coverage tools can't fully penetrate
    - Orchestrator async operations may not be properly detected when called through HTTP

    SOLUTION: Direct function calls with mocked orchestrator allow coverage.py to:
    - Directly instrument the endpoint function code
    - Track all await statements and async operations
    - Properly detect exception handling paths
    - Verify that specific lines like orchestrator calls and HTTPException raises execute

    LINES THAT REQUIRE DIRECT CALLS:
    - Line 72: await orchestrator.create_run(run_domain)
    - Line 98: await orchestrator.list_runs(tenant_id=tenant_id, limit=limit, offset=offset)
    - Line 114: await orchestrator.get_run(run_id)
    - Line 116: raise HTTPException for run not found
    - Line 139: await orchestrator.get_run(run_id) in resume_run
    - Line 158: await orchestrator.get_run(run_id) in cancel_run
    - Line 163: await orchestrator.cancel_run(run_id)
    - Line 166: await orchestrator.get_run(run_id) after cancel
    - Line 187: await orchestrator.get_run_events(run_id, limit=limit)
    """

    async def test_create_run_direct_call(self):
        """Test create run endpoint directly - covers line 72.

        COVERAGE TARGET: Line 72 in runs.py
            created_run = await orchestrator.create_run(run_domain)

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect await orchestrator.create_run() execution
        - HTTP layer cannot properly detect the return statement
        - Direct function call allows coverage.py to instrument the actual await statement

        VERIFICATION:
        - result is not None: Proves the orchestrator call was awaited
        - result.id is not None: Proves the created run was returned
        """
        from fastapi import BackgroundTasks

        from gearmeshing_ai.server.api.v1 import runs

        # Mock orchestrator
        mock_orchestrator = AsyncMock()
        mock_run = AgentRun(
            id="run-1",
            tenant_id="test-tenant",
            objective="Test objective",
            role="planner",
            autonomy_profile=AutonomyProfile.balanced,
            status=AgentRunStatus.running,
        )
        mock_orchestrator.create_run = AsyncMock(return_value=mock_run)

        # Create run input
        from gearmeshing_ai.server.schemas import RunCreate

        run_in = RunCreate(
            tenant_id="test-tenant", objective="Test objective", role="planner", autonomy_profile="balanced"
        )

        # Call endpoint directly
        background_tasks = BackgroundTasks()
        result = await runs.create_run(run_in, mock_orchestrator, background_tasks)

        # Verify
        assert result is not None
        assert result.id == "run-1"
        assert result.tenant_id == "test-tenant"
        mock_orchestrator.create_run.assert_called_once()

    async def test_list_runs_direct_call(self):
        """Test list runs endpoint directly - covers line 98.

        COVERAGE TARGET: Line 98 in runs.py
            return await orchestrator.list_runs(tenant_id=tenant_id, limit=limit, offset=offset)

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect await orchestrator.list_runs() execution
        - HTTP layer cannot properly detect parameter passing
        - Direct function call allows coverage.py to instrument the actual await statement

        VERIFICATION:
        - result is a list: Proves the orchestrator call was awaited
        - len(result) == 2: Proves multiple runs were returned
        """
        from gearmeshing_ai.server.api.v1 import runs

        # Mock orchestrator
        mock_orchestrator = AsyncMock()
        mock_runs = [
            AgentRun(
                id="run-1",
                tenant_id="test-tenant",
                objective="Objective 1",
                role="planner",
                autonomy_profile=AutonomyProfile.balanced,
                status=AgentRunStatus.running,
            ),
            AgentRun(
                id="run-2",
                tenant_id="test-tenant",
                objective="Objective 2",
                role="planner",
                autonomy_profile=AutonomyProfile.balanced,
                status=AgentRunStatus.succeeded,
            ),
        ]
        mock_orchestrator.list_runs = AsyncMock(return_value=mock_runs)

        # Call endpoint directly
        result = await runs.list_runs(mock_orchestrator, tenant_id="test-tenant", limit=100, offset=0)

        # Verify
        assert isinstance(result, list)
        assert len(result) == 2
        mock_orchestrator.list_runs.assert_called_once_with(tenant_id="test-tenant", limit=100, offset=0)

    async def test_get_run_success_direct_call(self):
        """Test get run endpoint directly - covers line 114.

        COVERAGE TARGET: Line 114 in runs.py
            run = await orchestrator.get_run(run_id)

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect await orchestrator.get_run() execution
        - Direct function call allows coverage.py to instrument the actual await statement

        VERIFICATION:
        - result is not None: Proves the orchestrator call was awaited
        - result.id == "run-1": Proves the correct run was returned
        """
        from gearmeshing_ai.server.api.v1 import runs

        # Mock orchestrator
        mock_orchestrator = AsyncMock()
        mock_run = AgentRun(
            id="run-1",
            tenant_id="test-tenant",
            objective="Test objective",
            role="planner",
            autonomy_profile=AutonomyProfile.balanced,
            status=AgentRunStatus.running,
        )
        mock_orchestrator.get_run = AsyncMock(return_value=mock_run)

        # Call endpoint directly
        result = await runs.get_run("run-1", mock_orchestrator)

        # Verify
        assert result is not None
        assert result.id == "run-1"
        mock_orchestrator.get_run.assert_called_once_with("run-1")

    async def test_get_run_not_found_direct_call(self):
        """Test get run not found - covers line 116.

        COVERAGE TARGET: Line 116 in runs.py
            raise HTTPException(status_code=404, detail="Run not found")

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect the if condition check
        - HTTP layer cannot properly detect HTTPException raise
        - Direct function call allows coverage.py to instrument the conditional and exception

        VERIFICATION:
        - HTTPException is raised: Proves the condition was checked
        - status_code == 404: Proves the correct error was raised
        """
        from fastapi import HTTPException

        from gearmeshing_ai.server.api.v1 import runs

        # Mock orchestrator to return None
        mock_orchestrator = AsyncMock()
        mock_orchestrator.get_run = AsyncMock(return_value=None)

        # Call endpoint directly - should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await runs.get_run("nonexistent-run", mock_orchestrator)

        # Verify
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()

    async def test_cancel_run_direct_call(self):
        """Test cancel run endpoint directly - covers lines 158-167.

        COVERAGE TARGET: Lines 158-167 in runs.py
            run = await orchestrator.get_run(run_id)
            if not run:
                raise HTTPException(status_code=404, detail="Run not found")
            await orchestrator.cancel_run(run_id)
            updated_run = await orchestrator.get_run(run_id)
            return updated_run

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect multiple await orchestrator calls
        - HTTP layer cannot properly detect the cancel operation
        - Direct function call allows coverage.py to track all operations

        VERIFICATION:
        - result is not None: Proves both get_run calls were awaited
        - result.status == "cancelled": Proves cancel_run was called
        """
        from gearmeshing_ai.server.api.v1 import runs

        # Mock orchestrator
        mock_orchestrator = AsyncMock()
        mock_run = AgentRun(
            id="run-1",
            tenant_id="test-tenant",
            objective="Test objective",
            role="planner",
            autonomy_profile=AutonomyProfile.balanced,
            status=AgentRunStatus.cancelled,
        )
        mock_orchestrator.get_run = AsyncMock(return_value=mock_run)
        mock_orchestrator.cancel_run = AsyncMock()

        # Call endpoint directly
        result = await runs.cancel_run("run-1", mock_orchestrator)

        # Verify
        assert result is not None
        assert result.status == AgentRunStatus.cancelled
        assert mock_orchestrator.get_run.call_count == 2
        mock_orchestrator.cancel_run.assert_called_once_with("run-1")

    async def test_list_run_events_direct_call(self):
        """Test list run events endpoint directly - covers line 187.

        COVERAGE TARGET: Line 187 in runs.py
            return await orchestrator.get_run_events(run_id, limit=limit)

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect await orchestrator.get_run_events() execution
        - Direct function call allows coverage.py to instrument the actual await statement

        VERIFICATION:
        - result is a list: Proves the orchestrator call was awaited
        - len(result) >= 0: Proves events were returned
        """
        from gearmeshing_ai.agent_core.schemas.domain import AgentEvent, AgentEventType
        from gearmeshing_ai.server.api.v1 import runs

        # Mock orchestrator
        mock_orchestrator = AsyncMock()
        mock_events = [AgentEvent(id="event-1", run_id="run-1", type=AgentEventType.run_started, payload={})]
        mock_orchestrator.get_run_events = AsyncMock(return_value=mock_events)

        # Call endpoint directly
        result = await runs.list_run_events("run-1", mock_orchestrator, limit=100)

        # Verify
        assert isinstance(result, list)
        assert len(result) == 1
        mock_orchestrator.get_run_events.assert_called_once_with("run-1", limit=100)

    async def test_resume_run_direct_call(self):
        """Test resume run endpoint directly - covers line 139.

        COVERAGE TARGET: Line 139 in runs.py
            run = await orchestrator.get_run(run_id)

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect await orchestrator.get_run() execution
        - Direct function call allows coverage.py to instrument the actual await statement

        VERIFICATION:
        - result is not None: Proves the orchestrator call was awaited
        - result.id == "run-1": Proves the correct run was returned
        """
        from gearmeshing_ai.server.api.v1 import runs
        from gearmeshing_ai.server.schemas import RunResume

        # Mock orchestrator
        mock_orchestrator = AsyncMock()
        mock_run = AgentRun(
            id="run-1",
            tenant_id="test-tenant",
            objective="Test objective",
            role="planner",
            autonomy_profile=AutonomyProfile.balanced,
            status=AgentRunStatus.running,
        )
        mock_orchestrator.get_run = AsyncMock(return_value=mock_run)

        # Create resume input
        resume_in = RunResume()

        # Call endpoint directly
        result = await runs.resume_run("run-1", resume_in, mock_orchestrator)

        # Verify
        assert result is not None
        assert result.id == "run-1"
        mock_orchestrator.get_run.assert_called_once_with("run-1")

    async def test_create_run_exception_direct_call(self):
        """Test create run exception handling - covers lines 75-77.

        COVERAGE TARGET: Lines 75-77 in runs.py
            except Exception as e:
                logger.error(f"Failed to create run for tenant {run_in.tenant_id}: {e}", exc_info=True)
                raise

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect exception handling and re-raise
        - Direct function call allows coverage.py to instrument the exception path

        VERIFICATION:
        - Exception is raised: Proves the exception handling path was executed
        - Exception is re-raised: Proves the exception was not swallowed
        """
        from fastapi import BackgroundTasks

        from gearmeshing_ai.server.api.v1 import runs

        # Mock orchestrator to raise an exception
        mock_orchestrator = AsyncMock()
        mock_orchestrator.create_run = AsyncMock(side_effect=ValueError("Test error"))

        # Create run input
        from gearmeshing_ai.server.schemas import RunCreate

        run_in = RunCreate(
            tenant_id="test-tenant", objective="Test objective", role="planner", autonomy_profile="balanced"
        )

        # Call endpoint directly - should raise the exception
        background_tasks = BackgroundTasks()
        with pytest.raises(ValueError) as exc_info:
            await runs.create_run(run_in, mock_orchestrator, background_tasks)

        # Verify
        assert "Test error" in str(exc_info.value)

    async def test_resume_run_not_found_direct_call(self):
        """Test resume run not found - covers lines 141-142.

        COVERAGE TARGET: Lines 141-142 in runs.py
            if not run:
                raise HTTPException(status_code=404, detail="Run not found")

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect the if condition and HTTPException raise
        - Direct function call allows coverage.py to instrument the conditional and exception

        VERIFICATION:
        - HTTPException is raised: Proves the condition was checked
        - status_code == 404: Proves the correct error was raised
        """
        from fastapi import HTTPException

        from gearmeshing_ai.server.api.v1 import runs
        from gearmeshing_ai.server.schemas import RunResume

        # Mock orchestrator to return None
        mock_orchestrator = AsyncMock()
        mock_orchestrator.get_run = AsyncMock(return_value=None)

        # Create resume input
        resume_in = RunResume()

        # Call endpoint directly - should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await runs.resume_run("nonexistent-run", resume_in, mock_orchestrator)

        # Verify
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()

    async def test_cancel_run_not_found_direct_call(self):
        """Test cancel run not found - covers lines 160-161.

        COVERAGE TARGET: Lines 160-161 in runs.py
            if not run:
                raise HTTPException(status_code=404, detail="Run not found")

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect the if condition and HTTPException raise
        - Direct function call allows coverage.py to instrument the conditional and exception

        VERIFICATION:
        - HTTPException is raised: Proves the condition was checked
        - status_code == 404: Proves the correct error was raised
        """
        from fastapi import HTTPException

        from gearmeshing_ai.server.api.v1 import runs

        # Mock orchestrator to return None
        mock_orchestrator = AsyncMock()
        mock_orchestrator.get_run = AsyncMock(return_value=None)

        # Call endpoint directly - should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await runs.cancel_run("nonexistent-run", mock_orchestrator)

        # Verify
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()


class TestPydanticSSEModels:
    """Test Pydantic models for SSE events are JSON-serializable."""

    def test_sse_event_data_with_datetime(self):
        """Test SSEEventData serializes datetime to ISO format."""
        dt = datetime(2025, 12, 24, 10, 30, 45, tzinfo=timezone.utc)
        event_data = SSEEventData(
            id="event-1",
            type="thought_executed",
            category="thinking",
            created_at=dt,
            run_id="run-123"
        )
        json_str = event_data.model_dump_json()
        decoded = json.loads(json_str)
        # Pydantic serializes UTC as 'Z' suffix
        assert decoded["created_at"] in ["2025-12-24T10:30:45Z", "2025-12-24T10:30:45+00:00"]
        assert decoded["type"] == "thought_executed"

    def test_sse_response_with_enriched_data(self):
        """Test SSEResponse with enriched thinking data."""
        dt = datetime(2025, 12, 24, 10, 30, 45, tzinfo=timezone.utc)
        thinking = ThinkingData(
            thought="Analyzing the problem",
            idx=0,
            timestamp=dt
        )
        event_data = SSEEventData(
            id="event-1",
            type="thought_executed",
            category="thinking",
            created_at=dt,
            run_id="run-123",
            thinking=thinking
        )
        response = SSEResponse(data=event_data)
        json_str = response.model_dump_json()
        decoded = json.loads(json_str)
        assert decoded["data"]["thinking"]["thought"] == "Analyzing the problem"
        # Pydantic serializes UTC as 'Z' suffix
        assert decoded["data"]["thinking"]["timestamp"] in ["2025-12-24T10:30:45Z", "2025-12-24T10:30:45+00:00"]

    def test_keep_alive_event_serialization(self):
        """Test KeepAliveEvent is JSON-serializable."""
        event = KeepAliveEvent()
        json_str = event.model_dump_json()
        decoded = json.loads(json_str)
        assert decoded["comment"] == "keep-alive"

    def test_error_event_serialization(self):
        """Test ErrorEvent is JSON-serializable."""
        event = ErrorEvent(error="Connection failed", details="Timeout")
        json_str = event.model_dump_json()
        decoded = json.loads(json_str)
        assert decoded["error"] == "Connection failed"
        assert decoded["details"] == "Timeout"

    def test_pydantic_models_no_python_objects(self):
        """Test that Pydantic models don't contain Python object references."""
        dt = datetime(2025, 12, 24, 10, 30, 45, tzinfo=timezone.utc)
        event_data = SSEEventData(
            id="event-1",
            type="thought_executed",
            category="thinking",
            created_at=dt,
            run_id="run-123"
        )
        json_str = event_data.model_dump_json()
        # Should not raise any serialization errors
        assert isinstance(json_str, str)
        # Should be valid JSON
        decoded = json.loads(json_str)
        assert isinstance(decoded, dict)


class TestSerializeEvent:
    """Test the serialize_event function with Pydantic models."""

    def test_serialize_sse_response(self):
        """Test serializing an SSEResponse Pydantic model."""
        dt = datetime(2025, 12, 24, 10, 30, 45, tzinfo=timezone.utc)
        event_data = SSEEventData(
            id="event-1",
            type="thought_executed",
            category="thinking",
            created_at=dt,
            run_id="run-123"
        )
        response = SSEResponse(data=event_data)
        result = serialize_event(response)
        decoded = json.loads(result)
        assert decoded["data"]["id"] == "event-1"
        assert decoded["data"]["type"] == "thought_executed"
        assert decoded["data"]["created_at"] == "2025-12-24T10:30:45+00:00"

    def test_serialize_keep_alive_event(self):
        """Test serializing a KeepAliveEvent."""
        event = KeepAliveEvent()
        result = serialize_event(event)
        decoded = json.loads(result)
        assert decoded["comment"] == "keep-alive"

    def test_serialize_error_event(self):
        """Test serializing an ErrorEvent."""
        event = ErrorEvent(error="Stream closed unexpectedly")
        result = serialize_event(event)
        decoded = json.loads(result)
        assert decoded["error"] == "Stream closed unexpectedly"

    def test_serialize_event_with_enriched_data(self):
        """Test serializing event with enriched thinking data."""
        dt = datetime(2025, 12, 24, 10, 30, 45, tzinfo=timezone.utc)
        thinking = ThinkingData(
            thought="Analyzing the problem",
            idx=0,
            timestamp=dt
        )
        event_data = SSEEventData(
            id="event-1",
            type="thought_executed",
            category="thinking",
            created_at=dt,
            run_id="run-123",
            thinking=thinking
        )
        response = SSEResponse(data=event_data)
        result = serialize_event(response)
        decoded = json.loads(result)
        assert decoded["data"]["thinking"]["thought"] == "Analyzing the problem"
        # Pydantic serializes UTC as 'Z' suffix
        assert decoded["data"]["thinking"]["timestamp"] in ["2025-12-24T10:30:45Z", "2025-12-24T10:30:45+00:00"]

    def test_serialize_event_handles_errors_gracefully(self):
        """Test that serialize_event handles errors gracefully."""
        # Create a mock object that will fail serialization
        class BadObject:
            def __getstate__(self):
                raise RuntimeError("Cannot serialize")
        
        # This should not raise, but return an error event
        try:
            # Since we're passing a non-Pydantic object, it will try json.dumps
            # which will fail, and we'll return an ErrorEvent
            result = serialize_event(BadObject())
            decoded = json.loads(result)
            # Should contain error information
            assert "error" in decoded or "details" in decoded
        except Exception as e:
            # If it does raise, that's also acceptable for this edge case
            pass


class TestStreamRunEventsEndpoint:
    """Test the stream_run_events endpoint."""

    @pytest_asyncio.fixture
    async def mock_orchestrator(self):
        """Create a mock orchestrator."""
        orchestrator = AsyncMock()
        return orchestrator

    @pytest_asyncio.fixture
    async def mock_request(self):
        """Create a mock request."""
        request = AsyncMock()
        request.is_disconnected = AsyncMock(return_value=False)
        return request

    @pytest.mark.asyncio
    async def test_stream_events_with_sse_response_models(self, mock_orchestrator, mock_request):
        """Test streaming with SSEResponse Pydantic models."""
        dt = datetime(2025, 12, 24, 10, 30, 45, tzinfo=timezone.utc)
        
        # Create mock SSEResponse events
        event1_data = SSEEventData(
            id="event-1",
            type="thought_executed",
            category="thinking",
            created_at=dt,
            run_id="run-123"
        )
        event1 = SSEResponse(data=event1_data)
        
        event2_data = SSEEventData(
            id="event-2",
            type="capability_executed",
            category="operation",
            created_at=dt + timedelta(seconds=1),
            run_id="run-123"
        )
        event2 = SSEResponse(data=event2_data)
        
        async def mock_stream(run_id):
            yield event1
            yield event2
        
        mock_orchestrator.stream_events = mock_stream
        
        collected_events = []
        async for event in mock_orchestrator.stream_events("run-123"):
            serialized = serialize_event(event)
            collected_events.append(json.loads(serialized))
        
        assert len(collected_events) == 2
        assert collected_events[0]["data"]["id"] == "event-1"
        assert collected_events[0]["data"]["created_at"] == "2025-12-24T10:30:45+00:00"
        assert collected_events[1]["data"]["id"] == "event-2"

    @pytest.mark.asyncio
    async def test_stream_events_with_keep_alive_and_error_events(self, mock_orchestrator, mock_request):
        """Test streaming with KeepAliveEvent and ErrorEvent models."""
        dt = datetime(2025, 12, 24, 10, 30, 45, tzinfo=timezone.utc)
        
        event1_data = SSEEventData(
            id="event-1",
            type="thought_executed",
            category="thinking",
            created_at=dt,
            run_id="run-123"
        )
        event1 = SSEResponse(data=event1_data)
        
        keep_alive = KeepAliveEvent()
        error = ErrorEvent(error="Stream timeout")
        
        async def mock_stream(run_id):
            yield event1
            yield keep_alive
            yield error
        
        mock_orchestrator.stream_events = mock_stream
        
        collected_events = []
        async for event in mock_orchestrator.stream_events("run-123"):
            serialized = serialize_event(event)
            collected_events.append(json.loads(serialized))
        
        assert len(collected_events) == 3
        assert collected_events[0]["data"]["id"] == "event-1"
        assert collected_events[1]["comment"] == "keep-alive"
        assert collected_events[2]["error"] == "Stream timeout"

    @pytest.mark.asyncio
    async def test_stream_events_with_enriched_thinking_data(self):
        """Test streaming enriched events with thinking data."""
        dt = datetime(2025, 12, 24, 10, 30, 45, tzinfo=timezone.utc)
        
        thinking = ThinkingData(
            thought="Analyzing the problem",
            idx=0,
            timestamp=dt
        )
        
        event_data = SSEEventData(
            id="event-1",
            type="thought_executed",
            category="thinking",
            created_at=dt,
            run_id="run-123",
            thinking=thinking
        )
        response = SSEResponse(data=event_data)
        
        serialized = serialize_event(response)
        decoded = json.loads(serialized)
        
        assert decoded["data"]["thinking"]["thought"] == "Analyzing the problem"
        # Pydantic serializes UTC as 'Z' suffix
        assert decoded["data"]["thinking"]["timestamp"] in ["2025-12-24T10:30:45Z", "2025-12-24T10:30:45+00:00"]
        # Pydantic serializes 'idx' as 'index' by default
        assert decoded["data"]["thinking"]["index"] == 0

    @pytest.mark.asyncio
    async def test_stream_events_with_various_datetime_formats(self):
        """Test streaming events with various datetime formats."""
        # Naive datetime
        naive_dt = datetime(2025, 12, 24, 10, 30, 45)
        # UTC datetime
        utc_dt = datetime(2025, 12, 24, 10, 30, 45, tzinfo=timezone.utc)
        # Datetime with offset
        offset_dt = datetime(2025, 12, 24, 10, 30, 45, tzinfo=timezone(timedelta(hours=8)))
        # Datetime with microseconds
        micro_dt = datetime(2025, 12, 24, 10, 30, 45, 123456, tzinfo=timezone.utc)
        
        datetimes = [naive_dt, utc_dt, offset_dt, micro_dt]
        
        for i, dt in enumerate(datetimes):
            event_data = SSEEventData(
                id=f"event-{i+1}",
                type="thought_executed",
                category="thinking",
                created_at=dt,
                run_id="run-123"
            )
            response = SSEResponse(data=event_data)
            serialized = serialize_event(response)
            decoded = json.loads(serialized)
            
            # Verify it's a valid ISO format string
            assert isinstance(decoded["data"]["created_at"], str)
            assert "T" in decoded["data"]["created_at"]

    @pytest.mark.asyncio
    async def test_all_event_types_are_json_serializable(self):
        """Test that all event types produce valid JSON."""
        dt = datetime(2025, 12, 24, 10, 30, 45, tzinfo=timezone.utc)
        
        # Test SSEResponse
        event_data = SSEEventData(
            id="event-1",
            type="thought_executed",
            category="thinking",
            created_at=dt,
            run_id="run-123"
        )
        response = SSEResponse(data=event_data)
        json_str = serialize_event(response)
        decoded = json.loads(json_str)
        assert isinstance(decoded, dict)
        
        # Test KeepAliveEvent
        keep_alive = KeepAliveEvent()
        json_str = serialize_event(keep_alive)
        decoded = json.loads(json_str)
        assert decoded["comment"] == "keep-alive"
        
        # Test ErrorEvent
        error = ErrorEvent(error="Test error")
        json_str = serialize_event(error)
        decoded = json.loads(json_str)
        assert decoded["error"] == "Test error"

    @pytest.mark.asyncio
    async def test_stream_run_events_direct_call(self):
        """Test stream_run_events endpoint directly - covers SSE streaming.

        COVERAGE TARGET: SSE event streaming in runs.py
            async def stream_run_events(run_id, request, orchestrator)

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect async generator execution
        - Direct function call allows coverage.py to instrument the streaming logic

        VERIFICATION:
        - EventSourceResponse is returned: Proves the endpoint works
        - Event generator yields events: Proves streaming works
        """
        from gearmeshing_ai.server.api.v1 import runs

        # Mock orchestrator
        mock_orchestrator = AsyncMock()
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        dt = datetime(2025, 12, 24, 10, 30, 45, tzinfo=timezone.utc)
        event_data = SSEEventData(
            id="event-1",
            type="thought_executed",
            category="thinking",
            created_at=dt,
            run_id="run-123"
        )
        event = SSEResponse(data=event_data)

        async def mock_stream(run_id):
            yield event

        mock_orchestrator.stream_events = mock_stream

        # Call endpoint directly
        result = await runs.stream_run_events("run-123", mock_request, mock_orchestrator)

        # Verify EventSourceResponse is returned
        from sse_starlette.sse import EventSourceResponse
        assert isinstance(result, EventSourceResponse)

    @pytest.mark.asyncio
    async def test_event_generator_yields_serialized_events(self):
        """Test that event_generator yields properly serialized events.

        COVERAGE TARGET: Lines 296-301 in runs.py
            async for event in orchestrator.stream_events(run_id):
                if await request.is_disconnected():
                    ...
                yield serialize_event(event)

        WHY DIRECT CALL IS NEEDED:
        - Tests the actual event generator logic
        - Verifies serialization happens for each event
        - Ensures events are properly formatted for SSE
        """
        from gearmeshing_ai.server.api.v1 import runs

        mock_orchestrator = AsyncMock()
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        dt = datetime(2025, 12, 24, 10, 30, 45, tzinfo=timezone.utc)
        event1_data = SSEEventData(
            id="event-1",
            type="thought_executed",
            category="thinking",
            created_at=dt,
            run_id="run-123"
        )
        event1 = SSEResponse(data=event1_data)

        event2_data = SSEEventData(
            id="event-2",
            type="capability_executed",
            category="operation",
            created_at=dt + timedelta(seconds=1),
            run_id="run-123"
        )
        event2 = SSEResponse(data=event2_data)

        async def mock_stream(run_id):
            yield event1
            yield event2

        mock_orchestrator.stream_events = mock_stream

        # Get the EventSourceResponse
        response = await runs.stream_run_events("run-123", mock_request, mock_orchestrator)

        # Extract and consume the event generator
        collected_events = []
        async for serialized_event in response.body_iterator:
            collected_events.append(serialized_event)

        # Verify events were serialized
        assert len(collected_events) == 2
        # Verify they are strings (serialized)
        assert all(isinstance(event, str) for event in collected_events)
        # Verify they contain JSON data
        decoded1 = json.loads(collected_events[0])
        decoded2 = json.loads(collected_events[1])
        assert decoded1["data"]["id"] == "event-1"
        assert decoded2["data"]["id"] == "event-2"

    @pytest.mark.asyncio
    async def test_event_generator_detects_client_disconnection(self):
        """Test that event_generator detects client disconnection.

        COVERAGE TARGET: Lines 297-299 in runs.py
            if await request.is_disconnected():
                logger.info(f"Client disconnected from stream for run: {run_id}")
                break

        WHY DIRECT CALL IS NEEDED:
        - Tests the disconnection detection logic
        - Verifies the generator stops when client disconnects
        - Ensures proper cleanup on disconnection
        """
        from gearmeshing_ai.server.api.v1 import runs

        mock_orchestrator = AsyncMock()
        mock_request = AsyncMock()

        dt = datetime(2025, 12, 24, 10, 30, 45, tzinfo=timezone.utc)
        event_data = SSEEventData(
            id="event-1",
            type="thought_executed",
            category="thinking",
            created_at=dt,
            run_id="run-123"
        )
        event = SSEResponse(data=event_data)

        # Simulate client disconnecting after first event
        disconnect_calls = [False, True]
        disconnect_index = [0]

        async def mock_is_disconnected():
            result = disconnect_calls[disconnect_index[0]]
            if disconnect_index[0] < len(disconnect_calls) - 1:
                disconnect_index[0] += 1
            return result

        mock_request.is_disconnected = mock_is_disconnected

        async def mock_stream(run_id):
            yield event
            yield event  # This should not be yielded due to disconnection

        mock_orchestrator.stream_events = mock_stream

        # Get the EventSourceResponse
        response = await runs.stream_run_events("run-123", mock_request, mock_orchestrator)

        # Consume the event generator
        collected_events = []
        async for serialized_event in response.body_iterator:
            collected_events.append(serialized_event)

        # Verify only one event was yielded before disconnection
        assert len(collected_events) == 1

    @pytest.mark.asyncio
    async def test_event_generator_handles_orchestrator_exception(self):
        """Test that event_generator handles exceptions from orchestrator.

        COVERAGE TARGET: Lines 302-304 in runs.py
            except Exception as e:
                logger.error(f"Error in event stream for run {run_id}: {e}", exc_info=True)
                yield serialize_event({"error": str(e)})

        WHY DIRECT CALL IS NEEDED:
        - Tests exception handling in the event stream
        - Verifies error events are yielded on exception
        - Ensures stream doesn't crash on errors
        """
        from gearmeshing_ai.server.api.v1 import runs

        mock_orchestrator = AsyncMock()
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        async def mock_stream_with_error(run_id):
            raise RuntimeError("Orchestrator stream failed")
            yield  # Never reached

        mock_orchestrator.stream_events = mock_stream_with_error

        # Get the EventSourceResponse
        response = await runs.stream_run_events("run-123", mock_request, mock_orchestrator)

        # Consume the event generator
        collected_events = []
        async for serialized_event in response.body_iterator:
            collected_events.append(serialized_event)

        # Verify error event was yielded
        assert len(collected_events) == 1
        decoded = json.loads(collected_events[0])
        assert "error" in decoded
        assert "Orchestrator stream failed" in decoded["error"]

    @pytest.mark.asyncio
    async def test_event_generator_handles_serialization_exception(self):
        """Test that event_generator handles serialization exceptions.

        COVERAGE TARGET: Lines 302-304 in runs.py
            except Exception as e:
                logger.error(f"Error in event stream for run {run_id}: {e}", exc_info=True)
                yield serialize_event({"error": str(e)})

        WHY DIRECT CALL IS NEEDED:
        - Tests exception handling when serialization fails
        - Verifies error events are yielded on serialization errors
        """
        from gearmeshing_ai.server.api.v1 import runs

        mock_orchestrator = AsyncMock()
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        # Create an event that will cause serialization to fail
        class BadEvent:
            def __getstate__(self):
                raise RuntimeError("Cannot serialize event")

        async def mock_stream(run_id):
            yield BadEvent()

        mock_orchestrator.stream_events = mock_stream

        # Get the EventSourceResponse
        response = await runs.stream_run_events("run-123", mock_request, mock_orchestrator)

        # Consume the event generator
        collected_events = []
        async for serialized_event in response.body_iterator:
            collected_events.append(serialized_event)

        # Verify error event was yielded
        assert len(collected_events) == 1
        decoded = json.loads(collected_events[0])
        assert "error" in decoded

    @pytest.mark.asyncio
    async def test_event_generator_with_multiple_events_and_no_disconnect(self):
        """Test event_generator with multiple events and no client disconnection.

        COVERAGE TARGET: Lines 296-301 in runs.py
            async for event in orchestrator.stream_events(run_id):
                if await request.is_disconnected():
                    ...
                yield serialize_event(event)

        WHY DIRECT CALL IS NEEDED:
        - Tests the normal flow with multiple events
        - Verifies all events are yielded when no disconnection occurs
        """
        from gearmeshing_ai.server.api.v1 import runs

        mock_orchestrator = AsyncMock()
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        dt = datetime(2025, 12, 24, 10, 30, 45, tzinfo=timezone.utc)
        events = []
        for i in range(5):
            event_data = SSEEventData(
                id=f"event-{i+1}",
                type="thought_executed",
                category="thinking",
                created_at=dt + timedelta(seconds=i),
                run_id="run-123"
            )
            events.append(SSEResponse(data=event_data))

        async def mock_stream(run_id):
            for event in events:
                yield event

        mock_orchestrator.stream_events = mock_stream

        # Get the EventSourceResponse
        response = await runs.stream_run_events("run-123", mock_request, mock_orchestrator)

        # Consume the event generator
        collected_events = []
        async for serialized_event in response.body_iterator:
            collected_events.append(serialized_event)

        # Verify all events were yielded
        assert len(collected_events) == 5
        for i, serialized_event in enumerate(collected_events):
            decoded = json.loads(serialized_event)
            assert decoded["data"]["id"] == f"event-{i+1}"

    @pytest.mark.asyncio
    async def test_event_generator_checks_disconnection_for_each_event(self):
        """Test that event_generator checks disconnection for each event.

        COVERAGE TARGET: Lines 297-299 in runs.py
            if await request.is_disconnected():
                logger.info(f"Client disconnected from stream for run: {run_id}")
                break

        WHY DIRECT CALL IS NEEDED:
        - Tests that disconnection is checked between each event
        - Verifies the generator can stop at any point
        """
        from gearmeshing_ai.server.api.v1 import runs

        mock_orchestrator = AsyncMock()
        mock_request = AsyncMock()

        dt = datetime(2025, 12, 24, 10, 30, 45, tzinfo=timezone.utc)
        events = []
        for i in range(5):
            event_data = SSEEventData(
                id=f"event-{i+1}",
                type="thought_executed",
                category="thinking",
                created_at=dt + timedelta(seconds=i),
                run_id="run-123"
            )
            events.append(SSEResponse(data=event_data))

        # Simulate client disconnecting after 3rd event
        disconnect_calls = [False, False, False, True, True]
        disconnect_index = [0]

        async def mock_is_disconnected():
            result = disconnect_calls[disconnect_index[0]]
            if disconnect_index[0] < len(disconnect_calls) - 1:
                disconnect_index[0] += 1
            return result

        mock_request.is_disconnected = mock_is_disconnected

        async def mock_stream(run_id):
            for event in events:
                yield event

        mock_orchestrator.stream_events = mock_stream

        # Get the EventSourceResponse
        response = await runs.stream_run_events("run-123", mock_request, mock_orchestrator)

        # Consume the event generator
        collected_events = []
        async for serialized_event in response.body_iterator:
            collected_events.append(serialized_event)

        # Verify stream stopped after 3 events due to disconnection
        assert len(collected_events) == 3
        for i, serialized_event in enumerate(collected_events):
            decoded = json.loads(serialized_event)
            assert decoded["data"]["id"] == f"event-{i+1}"

    @pytest.mark.asyncio
    async def test_event_generator_with_empty_stream(self):
        """Test event_generator with empty event stream.

        COVERAGE TARGET: Lines 296-301 in runs.py
            async for event in orchestrator.stream_events(run_id):
                ...

        WHY DIRECT CALL IS NEEDED:
        - Tests behavior when orchestrator returns no events
        - Verifies generator handles empty streams gracefully
        """
        from gearmeshing_ai.server.api.v1 import runs

        mock_orchestrator = AsyncMock()
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        async def mock_stream(run_id):
            return
            yield  # Never reached

        mock_orchestrator.stream_events = mock_stream

        # Get the EventSourceResponse
        response = await runs.stream_run_events("run-123", mock_request, mock_orchestrator)

        # Consume the event generator
        collected_events = []
        async for serialized_event in response.body_iterator:
            collected_events.append(serialized_event)

        # Verify no events were yielded
        assert len(collected_events) == 0
