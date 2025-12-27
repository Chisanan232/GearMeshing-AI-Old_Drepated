"""
Unit tests for Agent Runs API endpoints.

Tests cover:
- Creating new agent runs
- Listing runs with filtering
- Retrieving run details
- Resuming paused runs
- Cancelling active runs
- Listing run events

These tests use direct function calls to ensure proper coverage detection of async code.
See TestDirectFunctionCalls class documentation for why direct calls are necessary.
"""

from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient

from gearmeshing_ai.agent_core.schemas.domain import (
    AgentRun,
    AgentRunStatus,
    AutonomyProfile,
)

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
