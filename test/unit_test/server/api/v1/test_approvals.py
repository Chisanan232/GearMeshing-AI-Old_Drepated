"""
Unit tests for Approvals API endpoints.

Tests cover:
- Listing pending approvals for a run
- Submitting approval decisions (approve/reject)
- Error handling for missing approvals
- Edge cases and various approval states

These tests use direct function calls to ensure proper coverage detection of async code.
See TestDirectFunctionCalls class documentation for why direct calls are necessary.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gearmeshing_ai.core.models.domain import ApprovalDecision
from gearmeshing_ai.server.schemas import ApprovalSubmit

pytestmark = pytest.mark.asyncio


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
    - Line 30: await orchestrator.get_pending_approvals(run_id)
    - Lines 58-64: await orchestrator.submit_approval() with all parameters
    - Lines 65-66: HTTPException raise for approval not found
    """

    async def test_list_approvals_direct_call(self):
        """Test list approvals endpoint directly - covers line 30.

        COVERAGE TARGET: Line 30 in approvals.py
            return await orchestrator.get_pending_approvals(run_id)

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect await orchestrator.get_pending_approvals() execution
        - HTTP layer cannot properly detect the return statement
        - Direct function call allows coverage.py to instrument the actual await statement

        VERIFICATION:
        - result is a list: Proves the orchestrator call was awaited and returned
        - isinstance(result, list): Proves the return statement executed
        """
        from gearmeshing_ai.server.api.v1 import approvals

        # Mock orchestrator
        mock_orchestrator = AsyncMock()
        mock_orchestrator.get_pending_approvals = AsyncMock(
            return_value=[
                {
                    "id": "approval-1",
                    "run_id": "test-run",
                    "decision": None,
                },
                {
                    "id": "approval-2",
                    "run_id": "test-run",
                    "decision": None,
                },
            ]
        )

        # Call endpoint directly
        result = await approvals.list_approvals("test-run", mock_orchestrator)

        # Verify
        assert isinstance(result, list)
        assert len(result) == 2
        mock_orchestrator.get_pending_approvals.assert_called_once_with("test-run")

    async def test_list_approvals_empty_direct_call(self):
        """Test list approvals returning empty list - covers line 30.

        COVERAGE TARGET: Line 30 in approvals.py (with empty list)
            return await orchestrator.get_pending_approvals(run_id)

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect the return of empty list
        - Direct function call allows coverage.py to track the empty list return path

        VERIFICATION:
        - result is an empty list: Proves the return statement executed with empty result
        """
        from gearmeshing_ai.server.api.v1 import approvals

        # Mock orchestrator to return empty list
        mock_orchestrator = AsyncMock()
        mock_orchestrator.get_pending_approvals = AsyncMock(return_value=[])

        # Call endpoint directly
        result = await approvals.list_approvals("test-run", mock_orchestrator)

        # Verify
        assert isinstance(result, list)
        assert len(result) == 0
        mock_orchestrator.get_pending_approvals.assert_called_once_with("test-run")

    async def test_submit_approval_success_direct_call(self):
        """Test submit approval endpoint directly - covers lines 58-64.

        COVERAGE TARGET: Lines 58-64 in approvals.py
            approval = await orchestrator.submit_approval(
                run_id=run_id,
                approval_id=approval_id,
                decision=submission.decision.value,
                note=submission.note,
                decided_by="user-placeholder"
            )

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect await orchestrator.submit_approval() execution
        - HTTP layer cannot properly detect the parameter passing
        - Direct function call allows coverage.py to instrument the actual await statement

        VERIFICATION:
        - result is not None: Proves the orchestrator call was awaited
        - result.id == "approval-1": Proves the correct approval was returned
        """
        from fastapi import BackgroundTasks

        from gearmeshing_ai.server.api.v1 import approvals

        # Mock orchestrator
        mock_orchestrator = AsyncMock()
        mock_approval = {
            "id": "approval-1",
            "run_id": "test-run",
            "decision": "approved",
            "decided_by": "user-placeholder",
            "note": "Looks good",
        }
        mock_orchestrator.submit_approval = AsyncMock(return_value=mock_approval)

        # Mock background tasks
        mock_background_tasks = MagicMock(spec=BackgroundTasks)

        # Create submission
        submission = ApprovalSubmit(decision=ApprovalDecision.approved, note="Looks good")

        # Call endpoint directly
        result = await approvals.submit_approval(
            "test-run", "approval-1", submission, mock_orchestrator, mock_background_tasks
        )

        # Verify
        assert result is not None
        assert result["id"] == "approval-1"
        mock_orchestrator.submit_approval.assert_called_once_with(
            run_id="test-run",
            approval_id="approval-1",
            decision="approved",
            note="Looks good",
            decided_by="user-placeholder",
        )
        # Verify background task scheduled
        mock_background_tasks.add_task.assert_called_once_with(
            mock_orchestrator.execute_resume, "test-run", "approval-1"
        )

    async def test_submit_approval_reject_direct_call(self):
        """Test submit approval with reject decision - covers lines 58-64.

        COVERAGE TARGET: Lines 58-64 in approvals.py (with rejected decision)
            approval = await orchestrator.submit_approval(
                run_id=run_id,
                approval_id=approval_id,
                decision=submission.decision.value,
                note=submission.note,
                decided_by="user-placeholder"
            )

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect the parameter passing with rejected decision
        - Direct function call allows coverage.py to track the rejected decision path

        VERIFICATION:
        - result is not None: Proves the orchestrator call was awaited
        - result.decision == "rejected": Proves the correct decision was passed
        """
        from fastapi import BackgroundTasks

        from gearmeshing_ai.server.api.v1 import approvals

        # Mock orchestrator
        mock_orchestrator = AsyncMock()
        mock_approval = {
            "id": "approval-2",
            "run_id": "test-run",
            "decision": "rejected",
            "decided_by": "user-placeholder",
            "note": "Not approved",
        }
        mock_orchestrator.submit_approval = AsyncMock(return_value=mock_approval)

        # Mock background tasks
        mock_background_tasks = MagicMock(spec=BackgroundTasks)

        # Create submission with reject decision
        submission = ApprovalSubmit(decision=ApprovalDecision.rejected, note="Not approved")

        # Call endpoint directly
        result = await approvals.submit_approval(
            "test-run", "approval-2", submission, mock_orchestrator, mock_background_tasks
        )

        # Verify
        assert result is not None
        assert result["decision"] == "rejected"
        mock_orchestrator.submit_approval.assert_called_once_with(
            run_id="test-run",
            approval_id="approval-2",
            decision="rejected",
            note="Not approved",
            decided_by="user-placeholder",
        )
        # Verify NO background task scheduled
        mock_background_tasks.add_task.assert_not_called()

    async def test_submit_approval_without_note_direct_call(self):
        """Test submit approval without note - covers lines 58-64.

        COVERAGE TARGET: Lines 58-64 in approvals.py (with note=None)
            approval = await orchestrator.submit_approval(
                run_id=run_id,
                approval_id=approval_id,
                decision=submission.decision.value,
                note=submission.note,  # None in this case
                decided_by="user-placeholder"
            )

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect the parameter passing with None value
        - Direct function call allows coverage.py to track the None parameter path

        VERIFICATION:
        - result is not None: Proves the orchestrator call was awaited
        - orchestrator was called with note=None: Proves None was passed correctly
        """
        from fastapi import BackgroundTasks

        from gearmeshing_ai.server.api.v1 import approvals

        # Mock orchestrator
        mock_orchestrator = AsyncMock()
        mock_approval = {"id": "approval-3", "run_id": "test-run", "decision": "approved", "note": None}
        mock_orchestrator.submit_approval = AsyncMock(return_value=mock_approval)

        # Mock background tasks
        mock_background_tasks = MagicMock(spec=BackgroundTasks)

        # Create submission without note
        submission = ApprovalSubmit(decision=ApprovalDecision.approved)

        # Call endpoint directly
        result = await approvals.submit_approval(
            "test-run", "approval-3", submission, mock_orchestrator, mock_background_tasks
        )

        # Verify
        assert result is not None
        mock_orchestrator.submit_approval.assert_called_once()
        call_kwargs = mock_orchestrator.submit_approval.call_args[1]
        assert call_kwargs["note"] is None
        # Verify background task scheduled
        mock_background_tasks.add_task.assert_called_once()

    async def test_submit_approval_not_found_direct_call(self):
        """Test submit approval not found - covers lines 65-66.

        COVERAGE TARGET: Lines 65-66 in approvals.py
            if not approval:
                raise HTTPException(status_code=404, detail="Approval not found")

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect the if condition check
        - HTTP layer cannot properly detect HTTPException raise
        - Direct function call allows coverage.py to instrument the conditional and exception

        VERIFICATION:
        - HTTPException is raised: Proves the condition was checked
        - status_code == 404: Proves the correct error was raised
        """
        from fastapi import BackgroundTasks, HTTPException

        from gearmeshing_ai.server.api.v1 import approvals

        # Mock orchestrator to return None
        mock_orchestrator = AsyncMock()
        mock_orchestrator.submit_approval = AsyncMock(return_value=None)

        # Mock background tasks
        mock_background_tasks = MagicMock(spec=BackgroundTasks)

        # Create submission
        submission = ApprovalSubmit(decision=ApprovalDecision.rejected, note="Not approved")

        # Call endpoint directly - should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await approvals.submit_approval(
                "test-run", "nonexistent-approval", submission, mock_orchestrator, mock_background_tasks
            )

        # Verify
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()

    async def test_submit_approval_with_long_note_direct_call(self):
        """Test submit approval with long note - covers lines 58-64.

        COVERAGE TARGET: Lines 58-64 in approvals.py (with long note)
            approval = await orchestrator.submit_approval(
                run_id=run_id,
                approval_id=approval_id,
                decision=submission.decision.value,
                note=submission.note,  # Long string in this case
                decided_by="user-placeholder"
            )

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect the parameter passing with long string
        - Direct function call allows coverage.py to track the long note parameter path

        VERIFICATION:
        - result is not None: Proves the orchestrator call was awaited
        - orchestrator was called with long note: Proves long string was passed correctly
        """
        from fastapi import BackgroundTasks

        from gearmeshing_ai.server.api.v1 import approvals

        # Mock orchestrator
        long_note = "A" * 1000
        mock_orchestrator = AsyncMock()
        mock_approval = {"id": "approval-4", "run_id": "test-run", "decision": "approved", "note": long_note}
        mock_orchestrator.submit_approval = AsyncMock(return_value=mock_approval)

        # Mock background tasks
        mock_background_tasks = MagicMock(spec=BackgroundTasks)

        # Create submission with long note
        submission = ApprovalSubmit(decision=ApprovalDecision.approved, note=long_note)

        # Call endpoint directly
        result = await approvals.submit_approval(
            "test-run", "approval-4", submission, mock_orchestrator, mock_background_tasks
        )

        # Verify
        assert result is not None
        mock_orchestrator.submit_approval.assert_called_once()
        call_kwargs = mock_orchestrator.submit_approval.call_args[1]
        assert call_kwargs["note"] == long_note
        assert len(call_kwargs["note"]) == 1000

    async def test_submit_approval_with_special_characters_direct_call(self):
        """Test submit approval with special characters - covers lines 58-64.

        COVERAGE TARGET: Lines 58-64 in approvals.py (with special characters)
            approval = await orchestrator.submit_approval(
                run_id=run_id,
                approval_id=approval_id,
                decision=submission.decision.value,
                note=submission.note,  # Special characters in this case
                decided_by="user-placeholder"
            )

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect the parameter passing with special characters
        - Direct function call allows coverage.py to track the special character parameter path

        VERIFICATION:
        - result is not None: Proves the orchestrator call was awaited
        - orchestrator was called with special characters: Proves special chars were passed correctly
        """
        from fastapi import BackgroundTasks

        from gearmeshing_ai.server.api.v1 import approvals

        # Mock orchestrator
        special_note = "Special chars: @#$%^&*() 中文 العربية"
        mock_orchestrator = AsyncMock()
        mock_approval = {"id": "approval-5", "run_id": "test-run", "decision": "rejected", "note": special_note}
        mock_orchestrator.submit_approval = AsyncMock(return_value=mock_approval)

        # Mock background tasks
        mock_background_tasks = MagicMock(spec=BackgroundTasks)

        # Create submission with special characters
        submission = ApprovalSubmit(decision=ApprovalDecision.rejected, note=special_note)

        # Call endpoint directly
        result = await approvals.submit_approval(
            "test-run", "approval-5", submission, mock_orchestrator, mock_background_tasks
        )

        # Verify
        assert result is not None
        mock_orchestrator.submit_approval.assert_called_once()
        call_kwargs = mock_orchestrator.submit_approval.call_args[1]
        assert call_kwargs["note"] == special_note

    async def test_list_approvals_different_run_ids_direct_call(self):
        """Test list approvals for different run IDs - covers line 30.

        COVERAGE TARGET: Line 30 in approvals.py (multiple calls)
            return await orchestrator.get_pending_approvals(run_id)

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect multiple calls with different parameters
        - Direct function call allows coverage.py to track parameter variation

        VERIFICATION:
        - Each call returns correct run_id: Proves parameter passing works correctly
        """
        from gearmeshing_ai.server.api.v1 import approvals

        run_ids = ["run-1", "run-2", "run-3"]

        for run_id in run_ids:
            # Mock orchestrator
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_pending_approvals = AsyncMock(return_value=[])

            # Call endpoint directly
            result = await approvals.list_approvals(run_id, mock_orchestrator)

            # Verify
            assert isinstance(result, list)
            mock_orchestrator.get_pending_approvals.assert_called_once_with(run_id)

    async def test_submit_approval_expired_decision_direct_call(self):
        """Test submit approval with expired decision - covers lines 58-64.

        COVERAGE TARGET: Lines 58-64 in approvals.py (with expired decision)
            approval = await orchestrator.submit_approval(
                run_id=run_id,
                approval_id=approval_id,
                decision=submission.decision.value,
                note=submission.note,
                decided_by="user-placeholder"
            )

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect the parameter passing with expired decision
        - Direct function call allows coverage.py to track the expired decision path

        VERIFICATION:
        - result is not None: Proves the orchestrator call was awaited
        - result.decision == "expired": Proves the correct decision was passed
        """
        from fastapi import BackgroundTasks

        from gearmeshing_ai.server.api.v1 import approvals

        # Mock orchestrator
        mock_orchestrator = AsyncMock()
        mock_approval = {
            "id": "approval-6",
            "run_id": "test-run",
            "decision": "expired",
            "decided_by": "system",
            "note": "Approval expired",
        }
        mock_orchestrator.submit_approval = AsyncMock(return_value=mock_approval)

        # Mock background tasks
        mock_background_tasks = MagicMock(spec=BackgroundTasks)

        # Create submission with expired decision
        submission = ApprovalSubmit(decision=ApprovalDecision.expired, note="Approval expired")

        # Call endpoint directly
        result = await approvals.submit_approval(
            "test-run", "approval-6", submission, mock_orchestrator, mock_background_tasks
        )

        # Verify
        assert result is not None
        assert result["decision"] == "expired"
        mock_orchestrator.submit_approval.assert_called_once_with(
            run_id="test-run",
            approval_id="approval-6",
            decision="expired",
            note="Approval expired",
            decided_by="user-placeholder",
        )
        # Verify NO background task scheduled
        mock_background_tasks.add_task.assert_not_called()
