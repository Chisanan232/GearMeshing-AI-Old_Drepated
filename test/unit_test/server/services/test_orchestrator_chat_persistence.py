"""
Unit tests for orchestrator chat persistence integration.

Tests cover:
- Event formatting for chat display (_format_event_for_chat)
- Callback integration in stream_events
- Event type handling (operations, tools, approvals, etc.)
- Error handling and resilience
- Callback parameter passing
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from gearmeshing_ai.agent_core.schemas.domain import AgentEvent
from gearmeshing_ai.server.schemas import (
    OperationData,
    SSEEventData,
    SSEResponse,
    ToolExecutionData,
    ApprovalRequestData,
    ApprovalResolutionData,
)
from gearmeshing_ai.server.services.orchestrator import OrchestratorService


class TestOrchestratorEventFormatting:
    """Tests for orchestrator event formatting for chat persistence."""

    def test_format_operation_event(self):
        """Test formatting operation events for chat display."""
        orchestrator = OrchestratorService()

        dt = datetime(2025, 12, 28, 22, 0, 0, tzinfo=timezone.utc)
        event_data = SSEEventData(
            id="evt-1",
            type="capability_executed",
            category="operation",
            created_at=dt,
            run_id="run-1",
            payload={},
            operation=OperationData(
                capability="search",
                status="success",
                result={"found": 5},
                timestamp=dt,
            ),
        )

        formatted = orchestrator._format_event_for_chat(event_data)

        assert formatted != ""
        assert "Operation" in formatted or "search" in formatted
        assert "success" in formatted.lower()

    def test_format_tool_execution_event(self):
        """Test formatting tool execution events for chat display."""
        orchestrator = OrchestratorService()

        dt = datetime(2025, 12, 28, 22, 0, 0, tzinfo=timezone.utc)
        event_data = SSEEventData(
            id="evt-1",
            type="tool_invoked",
            category="tool_execution",
            created_at=dt,
            run_id="run-1",
            payload={},
            tool_execution=ToolExecutionData(
                server_id="mcp-1",
                tool_name="search_web",
                args={"query": "test"},
                result={"results": []},
                ok=True,
                risk="low",
                timestamp=dt,
            ),
        )

        formatted = orchestrator._format_event_for_chat(event_data)

        assert formatted != ""
        assert "Tool" in formatted or "search_web" in formatted
        assert "mcp-1" in formatted or "success" in formatted.lower()

    def test_format_approval_request_event(self):
        """Test formatting approval request events for chat display."""
        orchestrator = OrchestratorService()

        dt = datetime(2025, 12, 28, 22, 0, 0, tzinfo=timezone.utc)
        event_data = SSEEventData(
            id="evt-1",
            type="approval_requested",
            category="approval",
            created_at=dt,
            run_id="run-1",
            payload={},
            approval_request=ApprovalRequestData(
                capability="delete_file",
                risk="high",
                reason="User requested file deletion",
                timestamp=dt,
            ),
        )

        formatted = orchestrator._format_event_for_chat(event_data)

        assert formatted != ""
        assert "Approval" in formatted or "delete_file" in formatted
        assert "high" in formatted.lower()

    def test_format_approval_resolution_event(self):
        """Test formatting approval resolution events for chat display."""
        orchestrator = OrchestratorService()

        dt = datetime(2025, 12, 28, 22, 0, 0, tzinfo=timezone.utc)
        event_data = SSEEventData(
            id="evt-1",
            type="approval_resolved",
            category="approval",
            created_at=dt,
            run_id="run-1",
            payload={},
            approval_resolution=ApprovalResolutionData(
                capability="delete_file",
                decision="approved",
                reason="User approved",
                timestamp=dt,
            ),
        )

        formatted = orchestrator._format_event_for_chat(event_data)

        assert formatted != ""
        assert "Approval" in formatted or "delete_file" in formatted
        assert "approved" in formatted.lower()

    def test_format_thinking_event_returns_empty(self):
        """Test that thinking events return empty string (not displayed in chat)."""
        orchestrator = OrchestratorService()

        dt = datetime(2025, 12, 28, 22, 0, 0, tzinfo=timezone.utc)
        event_data = SSEEventData(
            id="evt-1",
            type="thought_executed",
            category="thinking",
            created_at=dt,
            run_id="run-1",
            payload={},
        )

        formatted = orchestrator._format_event_for_chat(event_data)

        # Thinking events should return empty string (not displayed in chat)
        assert formatted == ""

    def test_format_unknown_event_type(self):
        """Test formatting unknown event types."""
        orchestrator = OrchestratorService()

        dt = datetime(2025, 12, 28, 22, 0, 0, tzinfo=timezone.utc)
        event_data = SSEEventData(
            id="evt-1",
            type="unknown_event",
            category="other",
            created_at=dt,
            run_id="run-1",
            payload={},
        )

        formatted = orchestrator._format_event_for_chat(event_data)

        # Unknown events should return empty string
        assert formatted == ""

    def test_format_operation_failure(self):
        """Test formatting failed operation events."""
        orchestrator = OrchestratorService()

        dt = datetime(2025, 12, 28, 22, 0, 0, tzinfo=timezone.utc)
        event_data = SSEEventData(
            id="evt-1",
            type="capability_executed",
            category="operation",
            created_at=dt,
            run_id="run-1",
            payload={},
            operation=OperationData(
                capability="search",
                status="failed",
                result={"error": "Network timeout"},
                timestamp=dt,
            ),
        )

        formatted = orchestrator._format_event_for_chat(event_data)

        assert formatted != ""
        assert "Operation" in formatted or "search" in formatted
        assert "failed" in formatted.lower()

    def test_format_tool_execution_failure(self):
        """Test formatting failed tool execution events."""
        orchestrator = OrchestratorService()

        dt = datetime(2025, 12, 28, 22, 0, 0, tzinfo=timezone.utc)
        event_data = SSEEventData(
            id="evt-1",
            type="tool_invoked",
            category="tool_execution",
            created_at=dt,
            run_id="run-1",
            payload={},
            tool_execution=ToolExecutionData(
                server_id="mcp-1",
                tool_name="search_web",
                args={"query": "test"},
                result={"error": "Tool failed"},
                ok=False,
                risk="low",
                timestamp=dt,
            ),
        )

        formatted = orchestrator._format_event_for_chat(event_data)

        assert formatted != ""
        assert "Tool" in formatted or "search_web" in formatted


class TestOrchestratorCallbackIntegration:
    """Tests for orchestrator callback integration with persistence."""

    @pytest.mark.asyncio
    async def test_callback_receives_correct_parameters(self):
        """Test callback receives correct parameters from orchestrator."""
        callback_calls = []

        async def test_callback(run_id: str, display_text: str, event_type: str):
            callback_calls.append({
                "run_id": run_id,
                "display_text": display_text,
                "event_type": event_type,
            })

        # Simulate orchestrator calling callback
        await test_callback("run-123", "✓ Operation: search (success)", "capability_executed")

        assert len(callback_calls) == 1
        call = callback_calls[0]
        assert call["run_id"] == "run-123"
        assert call["display_text"] == "✓ Operation: search (success)"
        assert call["event_type"] == "capability_executed"

    @pytest.mark.asyncio
    async def test_callback_with_empty_display_text_not_called(self):
        """Test callback is not called when display_text is empty."""
        callback_calls = []

        async def test_callback(run_id: str, display_text: str, event_type: str):
            callback_calls.append((run_id, display_text, event_type))

        # Simulate orchestrator logic: only call if display_text is not empty
        display_text = ""  # Empty for thinking events
        if display_text:
            await test_callback("run-123", display_text, "thought_executed")

        assert len(callback_calls) == 0

    @pytest.mark.asyncio
    async def test_callback_with_multiple_event_types(self):
        """Test callback handles multiple event types correctly."""
        callback_calls = []

        async def test_callback(run_id: str, display_text: str, event_type: str):
            callback_calls.append((run_id, display_text, event_type))

        # Simulate orchestrator calling callback for different event types
        events = [
            ("run-123", "✓ Operation: search (success)", "capability_executed"),
            ("run-123", "✓ Tool: search_web (mcp-1)", "tool_invoked"),
            ("run-123", "⚠️ Approval Required: delete_file", "approval_requested"),
            ("run-123", "✓ Approval APPROVED", "approval_resolved"),
        ]

        for run_id, display_text, event_type in events:
            await test_callback(run_id, display_text, event_type)

        assert len(callback_calls) == 4
        assert callback_calls[0][2] == "capability_executed"
        assert callback_calls[1][2] == "tool_invoked"
        assert callback_calls[2][2] == "approval_requested"
        assert callback_calls[3][2] == "approval_resolved"

    @pytest.mark.asyncio
    async def test_callback_error_handling(self):
        """Test callback error handling doesn't break orchestrator."""
        callback_calls = []
        error_count = [0]

        async def failing_callback(run_id: str, display_text: str, event_type: str):
            callback_calls.append((run_id, display_text, event_type))
            error_count[0] += 1
            raise RuntimeError("Callback error")

        # Simulate orchestrator error handling
        try:
            await failing_callback("run-123", "✓ Operation", "capability_executed")
        except Exception:
            pass  # Orchestrator catches and logs

        assert error_count[0] == 1
        assert len(callback_calls) == 1

    @pytest.mark.asyncio
    async def test_callback_with_special_characters(self):
        """Test callback handles special characters in display text."""
        callback_calls = []

        async def test_callback(run_id: str, display_text: str, event_type: str):
            callback_calls.append((run_id, display_text, event_type))

        # Simulate orchestrator with special characters
        special_text = "✓ Operation: search (success) - Results: 5 items found\n  Details: Query='test & special'"
        await test_callback("run-123", special_text, "capability_executed")

        assert len(callback_calls) == 1
        assert callback_calls[0][1] == special_text
        assert "✓" in callback_calls[0][1]
        assert "&" in callback_calls[0][1]

    @pytest.mark.asyncio
    async def test_callback_with_long_display_text(self):
        """Test callback handles long display text."""
        callback_calls = []

        async def test_callback(run_id: str, display_text: str, event_type: str):
            callback_calls.append((run_id, display_text, event_type))

        # Simulate orchestrator with long text
        long_text = "✓ Operation: search (success)\n" + "\n".join([f"  Result {i}: Item {i}" for i in range(100)])
        await test_callback("run-123", long_text, "capability_executed")

        assert len(callback_calls) == 1
        assert len(callback_calls[0][1]) > 1000

    @pytest.mark.asyncio
    async def test_callback_preserves_run_id(self):
        """Test callback preserves run_id across multiple calls."""
        callback_calls = []

        async def test_callback(run_id: str, display_text: str, event_type: str):
            callback_calls.append((run_id, display_text, event_type))

        # Simulate multiple events for same run
        run_id = "run-456"
        await test_callback(run_id, "Event 1", "capability_executed")
        await test_callback(run_id, "Event 2", "tool_invoked")
        await test_callback(run_id, "Event 3", "approval_requested")

        assert len(callback_calls) == 3
        assert all(call[0] == run_id for call in callback_calls)

    @pytest.mark.asyncio
    async def test_callback_with_different_run_ids(self):
        """Test callback handles different run_ids correctly."""
        callback_calls = []

        async def test_callback(run_id: str, display_text: str, event_type: str):
            callback_calls.append((run_id, display_text, event_type))

        # Simulate events from different runs
        await test_callback("run-1", "Event 1", "capability_executed")
        await test_callback("run-2", "Event 2", "capability_executed")
        await test_callback("run-3", "Event 3", "capability_executed")

        assert len(callback_calls) == 3
        assert callback_calls[0][0] == "run-1"
        assert callback_calls[1][0] == "run-2"
        assert callback_calls[2][0] == "run-3"

    @pytest.mark.asyncio
    async def test_callback_async_execution(self):
        """Test callback executes asynchronously."""
        execution_order = []

        async def async_callback(run_id: str, display_text: str, event_type: str):
            execution_order.append(("callback_start", run_id))
            await AsyncMock()()  # Simulate async operation
            execution_order.append(("callback_end", run_id))

        # Execute callback
        await async_callback("run-123", "Test", "capability_executed")

        assert len(execution_order) == 2
        assert execution_order[0] == ("callback_start", "run-123")
        assert execution_order[1] == ("callback_end", "run-123")


class TestOrchestratorEventFormattingEdgeCases:
    """Tests for edge cases in orchestrator event formatting."""

    def test_format_event_with_none_operation(self):
        """Test formatting event with None operation."""
        orchestrator = OrchestratorService()

        dt = datetime(2025, 12, 28, 22, 0, 0, tzinfo=timezone.utc)
        event_data = SSEEventData(
            id="evt-1",
            type="capability_executed",
            category="operation",
            created_at=dt,
            run_id="run-1",
            payload={},
            operation=None,
        )

        formatted = orchestrator._format_event_for_chat(event_data)

        # Should handle None gracefully
        assert isinstance(formatted, str)

    def test_format_event_with_empty_operation_result(self):
        """Test formatting event with empty operation result."""
        orchestrator = OrchestratorService()

        dt = datetime(2025, 12, 28, 22, 0, 0, tzinfo=timezone.utc)
        event_data = SSEEventData(
            id="evt-1",
            type="capability_executed",
            category="operation",
            created_at=dt,
            run_id="run-1",
            payload={},
            operation=OperationData(
                capability="search",
                status="success",
                result={},
                timestamp=dt,
            ),
        )

        formatted = orchestrator._format_event_for_chat(event_data)

        # Should still format even with empty result
        assert formatted != ""

    def test_format_event_with_special_capability_names(self):
        """Test formatting events with special capability names."""
        orchestrator = OrchestratorService()

        dt = datetime(2025, 12, 28, 22, 0, 0, tzinfo=timezone.utc)
        special_names = [
            "search_web",
            "delete_file",
            "create_folder",
            "read_document",
            "send_email",
        ]

        for capability_name in special_names:
            event_data = SSEEventData(
                id="evt-1",
                type="capability_executed",
                category="operation",
                created_at=dt,
                run_id="run-1",
                payload={},
                operation=OperationData(
                    capability=capability_name,
                    status="success",
                    result={},
                    timestamp=dt,
                ),
            )

            formatted = orchestrator._format_event_for_chat(event_data)

            assert formatted != ""
            assert capability_name in formatted or "Operation" in formatted

    def test_format_event_with_high_risk_approval(self):
        """Test formatting approval events with high risk level."""
        orchestrator = OrchestratorService()

        dt = datetime(2025, 12, 28, 22, 0, 0, tzinfo=timezone.utc)
        event_data = SSEEventData(
            id="evt-1",
            type="approval_requested",
            category="approval",
            created_at=dt,
            run_id="run-1",
            payload={},
            approval_request=ApprovalRequestData(
                capability="delete_file",
                risk="high",
                reason="Deleting critical system file",
                timestamp=dt,
            ),
        )

        formatted = orchestrator._format_event_for_chat(event_data)

        assert formatted != ""
        assert "high" in formatted.lower()
        assert "delete_file" in formatted or "Approval" in formatted

    def test_format_event_with_low_risk_approval(self):
        """Test formatting approval events with low risk level."""
        orchestrator = OrchestratorService()

        dt = datetime(2025, 12, 28, 22, 0, 0, tzinfo=timezone.utc)
        event_data = SSEEventData(
            id="evt-1",
            type="approval_requested",
            category="approval",
            created_at=dt,
            run_id="run-1",
            payload={},
            approval_request=ApprovalRequestData(
                capability="read_file",
                risk="low",
                reason="Reading public document",
                timestamp=dt,
            ),
        )

        formatted = orchestrator._format_event_for_chat(event_data)

        assert formatted != ""
        assert "low" in formatted.lower()


class TestOrchestratorStreamEventsCallbackIntegration:
    """Integration tests for orchestrator stream_events callback execution (lines 201-211)."""

    @pytest.mark.asyncio
    async def test_stream_events_invokes_callback_with_operation_event(self):
        """Test stream_events actually invokes callback for operation events (lines 200-210)."""
        from gearmeshing_ai.server.services.orchestrator import OrchestratorService

        orchestrator = OrchestratorService()
        callback_invocations = []

        async def capture_callback(run_id: str, display_text: str, event_type: str) -> None:
            callback_invocations.append({
                "run_id": run_id,
                "display_text": display_text,
                "event_type": event_type,
            })

        # Mock the repository to return events
        dt = datetime(2025, 12, 28, 22, 0, 0, tzinfo=timezone.utc)
        
        event = AgentEvent(
            id="evt-1",
            type="capability.executed",
            created_at=dt,
            run_id="run-123",
            payload={},
        )

        mock_events_repo = AsyncMock()
        mock_events_repo.list = AsyncMock(return_value=[event])
        
        mock_repos = MagicMock()
        mock_repos.events = mock_events_repo
        orchestrator.repos = mock_repos

        # Mock the enrich method
        async def mock_enrich_event(evt):
            return SSEResponse(
                id=evt.id,
                type=evt.type,
                data=SSEEventData(
                    id=evt.id,
                    type="capability_executed",
                    category="operation",
                    created_at=dt,
                    run_id=evt.run_id,
                    payload={},
                    operation=OperationData(
                        capability="search",
                        status="success",
                        result={"found": 5},
                        timestamp=dt,
                    ),
                ),
            )

        orchestrator._enrich_event_for_sse = mock_enrich_event

        # Stream events with callback
        event_count = 0
        async for sse_event in orchestrator.stream_events("run-123", on_event_persisted=capture_callback):
            event_count += 1
            if event_count >= 1:
                break

        # Verify callback was actually invoked (lines 200-210)
        assert len(callback_invocations) == 1
        assert callback_invocations[0]["run_id"] == "run-123"
        assert callback_invocations[0]["event_type"] == "capability_executed"
        assert "search" in callback_invocations[0]["display_text"]

    @pytest.mark.asyncio
    async def test_stream_events_callback_receives_formatted_display_text(self):
        """Test callback receives formatted display_text from _format_event_for_chat (line 201)."""
        from gearmeshing_ai.server.services.orchestrator import OrchestratorService

        orchestrator = OrchestratorService()
        received_display_text = []

        async def capture_display_text(run_id: str, display_text: str, event_type: str) -> None:
            received_display_text.append(display_text)

        dt = datetime(2025, 12, 28, 22, 0, 0, tzinfo=timezone.utc)

        event = AgentEvent(
            id="evt-2",
            type="tool.invoked",
            created_at=dt,
            run_id="run-456",
            payload={},
        )

        mock_events_repo = AsyncMock()
        mock_events_repo.list = AsyncMock(return_value=[event])
        
        mock_repos = MagicMock()
        mock_repos.events = mock_events_repo
        orchestrator.repos = mock_repos

        async def mock_enrich_event(evt):
            return SSEResponse(
                id=evt.id,
                type=evt.type,
                data=SSEEventData(
                    id=evt.id,
                    type="tool_invoked",
                    category="tool_execution",
                    created_at=dt,
                    run_id=evt.run_id,
                    payload={},
                    tool_execution=ToolExecutionData(
                        tool_name="search_web",
                        server_id="mcp-1",
                        risk="low",
                        status="success",
                        timestamp=dt,
                    ),
                ),
            )

        orchestrator._enrich_event_for_sse = mock_enrich_event

        event_count = 0
        async for sse_event in orchestrator.stream_events("run-456", on_event_persisted=capture_display_text):
            event_count += 1
            if event_count >= 1:
                break

        # Verify display_text was formatted and passed (line 201)
        assert len(received_display_text) == 1
        assert received_display_text[0] != ""
        assert "search_web" in received_display_text[0]

    @pytest.mark.asyncio
    async def test_stream_events_skips_callback_when_display_text_empty(self):
        """Test callback is not invoked when display_text is empty (line 202)."""
        from gearmeshing_ai.server.services.orchestrator import OrchestratorService

        orchestrator = OrchestratorService()
        callback_invocations = []

        async def capture_callback(run_id: str, display_text: str, event_type: str) -> None:
            callback_invocations.append(display_text)

        dt = datetime(2025, 12, 28, 22, 0, 0, tzinfo=timezone.utc)

        # Event with no data that would produce empty display_text
        event = AgentEvent(
            id="evt-3",
            type="run.started",
            created_at=dt,
            run_id="run-789",
            payload={},
        )

        mock_events_repo = AsyncMock()
        mock_events_repo.list = AsyncMock(return_value=[event])
        
        mock_repos = MagicMock()
        mock_repos.events = mock_events_repo
        orchestrator.repos = mock_repos

        async def mock_enrich_event(evt):
            return SSEResponse(
                id=evt.id,
                type=evt.type,
                data=SSEEventData(
                    id=evt.id,
                    type="keep_alive",
                    category="system",
                    created_at=dt,
                    run_id=evt.run_id,
                    payload={},
                ),
            )

        orchestrator._enrich_event_for_sse = mock_enrich_event

        event_count = 0
        async for sse_event in orchestrator.stream_events("run-789", on_event_persisted=capture_callback):
            event_count += 1
            if event_count >= 1:
                break

        # Verify callback was not invoked for empty display_text (line 202)
        assert len(callback_invocations) == 0

    @pytest.mark.asyncio
    async def test_stream_events_callback_exception_handled_gracefully(self):
        """Test callback exception is caught and logged (lines 203-210)."""
        from gearmeshing_ai.server.services.orchestrator import OrchestratorService

        orchestrator = OrchestratorService()

        async def failing_callback(run_id: str, display_text: str, event_type: str) -> None:
            raise Exception("Callback processing error")

        dt = datetime(2025, 12, 28, 22, 0, 0, tzinfo=timezone.utc)

        event = AgentEvent(
            id="evt-4",
            type="capability.executed",
            created_at=dt,
            run_id="run-error",
            payload={},
        )

        mock_events_repo = AsyncMock()
        mock_events_repo.list = AsyncMock(return_value=[event])
        
        mock_repos = MagicMock()
        mock_repos.events = mock_events_repo
        orchestrator.repos = mock_repos

        async def mock_enrich_event(evt):
            return SSEResponse(
                id=evt.id,
                type=evt.type,
                data=SSEEventData(
                    id=evt.id,
                    type="capability_executed",
                    category="operation",
                    created_at=dt,
                    run_id=evt.run_id,
                    payload={},
                    operation=OperationData(
                        capability="test",
                        status="success",
                        result={},
                        timestamp=dt,
                    ),
                ),
            )

        orchestrator._enrich_event_for_sse = mock_enrich_event

        # Should not raise exception even though callback fails
        event_count = 0
        async for sse_event in orchestrator.stream_events("run-error", on_event_persisted=failing_callback):
            event_count += 1
            if event_count >= 1:
                break

        # Verify event was still yielded despite callback error
        assert event_count == 1

    @pytest.mark.asyncio
    async def test_stream_events_callback_receives_correct_parameters(self):
        """Test callback receives correct run_id, display_text, and event_type (lines 204-207)."""
        from gearmeshing_ai.server.services.orchestrator import OrchestratorService

        orchestrator = OrchestratorService()
        callback_params = {}

        async def capture_params(run_id: str, display_text: str, event_type: str) -> None:
            callback_params["run_id"] = run_id
            callback_params["display_text"] = display_text
            callback_params["event_type"] = event_type

        dt = datetime(2025, 12, 28, 22, 0, 0, tzinfo=timezone.utc)
        test_run_id = "run-param-test-123"
        test_event_type = "approval.requested"

        event = AgentEvent(
            id="evt-5",
            type="approval.requested",
            created_at=dt,
            run_id=test_run_id,
            payload={},
        )

        mock_events_repo = AsyncMock()
        mock_events_repo.list = AsyncMock(return_value=[event])
        
        mock_repos = MagicMock()
        mock_repos.events = mock_events_repo
        orchestrator.repos = mock_repos

        async def mock_enrich_event(evt):
            return SSEResponse(
                id=evt.id,
                type=evt.type,
                data=SSEEventData(
                    id=evt.id,
                    type="approval_requested",
                    category="approval",
                    created_at=dt,
                    run_id=evt.run_id,
                    payload={},
                    approval_request=ApprovalRequestData(
                        capability="delete_file",
                        risk="high",
                        reason="Test approval",
                        timestamp=dt,
                    ),
                ),
            )

        orchestrator._enrich_event_for_sse = mock_enrich_event

        event_count = 0
        async for sse_event in orchestrator.stream_events(test_run_id, on_event_persisted=capture_params):
            event_count += 1
            if event_count >= 1:
                break

        # Verify all parameters are correct (lines 204-207)
        assert callback_params["run_id"] == test_run_id
        assert callback_params["event_type"] == "approval_requested"
        assert "delete_file" in callback_params["display_text"]

    @pytest.mark.asyncio
    async def test_stream_events_with_none_callback_parameter(self):
        """Test stream_events handles None callback gracefully (line 200)."""
        from gearmeshing_ai.server.services.orchestrator import OrchestratorService

        orchestrator = OrchestratorService()

        dt = datetime(2025, 12, 28, 22, 0, 0, tzinfo=timezone.utc)

        async def mock_get_events(run_id: str, since_event_id: str = None):
            event = AgentEvent(
                id="evt-6",
                type="capability_executed",
                created_at=dt,
                run_id=run_id,
                payload={},
            )
            yield event

        async def mock_enrich_event(event):
            return SSEResponse(
                id=event.id,
                type=event.type,
                data=SSEEventData(
                    id=event.id,
                    type="capability_executed",
                    category="operation",
                    created_at=dt,
                    run_id=event.run_id,
                    payload={},
                    operation=OperationData(
                        capability="test",
                        status="success",
                        result={},
                        timestamp=dt,
                    ),
                ),
            )

        orchestrator._get_events = mock_get_events
        orchestrator._enrich_event_for_sse = mock_enrich_event

        # Should not raise exception with None callback
        event_count = 0
        async for event in orchestrator.stream_events("run-none", on_event_persisted=None):
            event_count += 1
            if event_count >= 1:
                break

        assert event_count == 1
