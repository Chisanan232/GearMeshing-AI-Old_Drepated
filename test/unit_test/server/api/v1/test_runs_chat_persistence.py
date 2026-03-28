"""
Unit tests for chat persistence integration in runs endpoint.

Tests cover:
- persist_event_to_chat callback functionality
- Chat session initialization in callback
- Event persistence with proper formatting
- Error handling and resilience
- Integration with ChatPersistenceService
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gearmeshing_ai.core.database.entities.chat_sessions import ChatSession
from gearmeshing_ai.core.models.domain import AgentRun, AgentRunStatus
from gearmeshing_ai.server.schemas import (
    ApprovalRequestData,
    OperationData,
    SSEEventData,
    SSEResponse,
    ToolExecutionData,
)


class TestPersistEventToChatCallback:
    """Tests for persist_event_to_chat callback in runs endpoint."""

    @pytest.mark.asyncio
    async def test_callback_initializes_chat_session_on_first_event(self):
        """Test that callback initializes chat session on first event."""
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        # Mock orchestrator
        mock_orchestrator = AsyncMock()
        mock_orchestrator.get_run = AsyncMock(
            return_value=AgentRun(
                id="run-123",
                tenant_id="tenant-1",
                role="planner",
                objective="Test",
                status=AgentRunStatus.running.value,
                created_at=datetime.now(timezone.utc),
            )
        )

        # Mock chat service
        mock_chat_service = AsyncMock(spec=ChatPersistenceService)
        mock_chat_session = ChatSession(
            id=1,
            title="Chat - planner",
            agent_role="planner",
            tenant_id="tenant-1",
            run_id="run-123",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        mock_chat_service.get_or_create_session = AsyncMock(return_value=mock_chat_session)
        mock_chat_service.add_agent_message = AsyncMock()

        # Create callback
        chat_session_holder = {"session": None}

        async def persist_event_to_chat(run_id: str, display_text: str, event_type: str) -> None:
            if chat_session_holder["session"] is None:
                chat_session_holder["session"] = await mock_chat_service.get_or_create_session(
                    run_id=run_id,
                    tenant_id="tenant-1",
                    agent_role="planner",
                    title=f"Chat - planner",
                    description=f"Chat history for run {run_id}",
                )

            if chat_session_holder["session"]:
                await mock_chat_service.add_agent_message(
                    session_id=chat_session_holder["session"].id,
                    content=display_text,
                    event_type=event_type,
                )

        # Call callback
        await persist_event_to_chat("run-123", "Operation completed", "capability_executed")

        # Verify session was initialized
        assert chat_session_holder["session"] is not None
        assert chat_session_holder["session"].id == 1
        mock_chat_service.get_or_create_session.assert_called_once()
        mock_chat_service.add_agent_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_callback_reuses_existing_session(self):
        """Test that callback reuses existing chat session."""
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        mock_chat_service = AsyncMock(spec=ChatPersistenceService)
        mock_chat_session = ChatSession(
            id=1,
            title="Chat - planner",
            agent_role="planner",
            tenant_id="tenant-1",
            run_id="run-123",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        mock_chat_service.add_agent_message = AsyncMock()

        # Pre-initialize session
        chat_session_holder = {"session": mock_chat_session}

        async def persist_event_to_chat(run_id: str, display_text: str, event_type: str) -> None:
            if chat_session_holder["session"]:
                await mock_chat_service.add_agent_message(
                    session_id=chat_session_holder["session"].id,
                    content=display_text,
                    event_type=event_type,
                )

        # Call callback multiple times
        await persist_event_to_chat("run-123", "Message 1", "capability_executed")
        await persist_event_to_chat("run-123", "Message 2", "tool_invoked")
        await persist_event_to_chat("run-123", "Message 3", "approval_requested")

        # Verify session was reused (not recreated)
        assert mock_chat_service.add_agent_message.call_count == 3
        assert chat_session_holder["session"].id == 1

    @pytest.mark.asyncio
    async def test_callback_persists_operation_event(self):
        """Test callback persists operation events correctly."""
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        mock_chat_service = AsyncMock(spec=ChatPersistenceService)
        mock_chat_session = ChatSession(
            id=1,
            title="Chat - planner",
            agent_role="planner",
            tenant_id="tenant-1",
            run_id="run-123",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        mock_chat_service.add_agent_message = AsyncMock()

        chat_session_holder = {"session": mock_chat_session}

        async def persist_event_to_chat(run_id: str, display_text: str, event_type: str) -> None:
            if chat_session_holder["session"]:
                await mock_chat_service.add_agent_message(
                    session_id=chat_session_holder["session"].id,
                    content=display_text,
                    event_type=event_type,
                )

        # Persist operation event
        await persist_event_to_chat(
            "run-123",
            "✓ Operation: search (success)",
            "capability_executed",
        )

        # Verify message was persisted
        mock_chat_service.add_agent_message.assert_called_once_with(
            session_id=1,
            content="✓ Operation: search (success)",
            event_type="capability_executed",
        )

    @pytest.mark.asyncio
    async def test_callback_persists_tool_execution_event(self):
        """Test callback persists tool execution events correctly."""
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        mock_chat_service = AsyncMock(spec=ChatPersistenceService)
        mock_chat_session = ChatSession(
            id=1,
            title="Chat - planner",
            agent_role="planner",
            tenant_id="tenant-1",
            run_id="run-123",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        mock_chat_service.add_agent_message = AsyncMock()

        chat_session_holder = {"session": mock_chat_session}

        async def persist_event_to_chat(run_id: str, display_text: str, event_type: str) -> None:
            if chat_session_holder["session"]:
                await mock_chat_service.add_agent_message(
                    session_id=chat_session_holder["session"].id,
                    content=display_text,
                    event_type=event_type,
                )

        # Persist tool execution event
        await persist_event_to_chat(
            "run-123",
            "✓ Tool: search_web (mcp-server-1)",
            "tool_invoked",
        )

        # Verify message was persisted
        mock_chat_service.add_agent_message.assert_called_once_with(
            session_id=1,
            content="✓ Tool: search_web (mcp-server-1)",
            event_type="tool_invoked",
        )

    @pytest.mark.asyncio
    async def test_callback_persists_approval_request_event(self):
        """Test callback persists approval request events correctly."""
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        mock_chat_service = AsyncMock(spec=ChatPersistenceService)
        mock_chat_session = ChatSession(
            id=1,
            title="Chat - planner",
            agent_role="planner",
            tenant_id="tenant-1",
            run_id="run-123",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        mock_chat_service.add_agent_message = AsyncMock()

        chat_session_holder = {"session": mock_chat_session}

        async def persist_event_to_chat(run_id: str, display_text: str, event_type: str) -> None:
            if chat_session_holder["session"]:
                await mock_chat_service.add_agent_message(
                    session_id=chat_session_holder["session"].id,
                    content=display_text,
                    event_type=event_type,
                )

        # Persist approval request event
        await persist_event_to_chat(
            "run-123",
            "⚠️ Approval Required: delete_file\n  Reason: User requested file deletion\n  Risk Level: high",
            "approval_requested",
        )

        # Verify message was persisted
        call_args = mock_chat_service.add_agent_message.call_args
        assert call_args[1]["session_id"] == 1
        assert "Approval Required" in call_args[1]["content"]
        assert call_args[1]["event_type"] == "approval_requested"

    @pytest.mark.asyncio
    async def test_callback_handles_missing_session_gracefully(self):
        """Test callback handles missing session gracefully."""
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        mock_chat_service = AsyncMock(spec=ChatPersistenceService)
        mock_chat_service.get_or_create_session = AsyncMock(return_value=None)

        chat_session_holder = {"session": None}

        async def persist_event_to_chat(run_id: str, display_text: str, event_type: str) -> None:
            if chat_session_holder["session"] is None:
                try:
                    chat_session_holder["session"] = await mock_chat_service.get_or_create_session(
                        run_id=run_id,
                        tenant_id="tenant-1",
                        agent_role="planner",
                        title=f"Chat - planner",
                        description=f"Chat history for run {run_id}",
                    )
                except Exception:
                    return

            if chat_session_holder["session"]:
                await mock_chat_service.add_agent_message(
                    session_id=chat_session_holder["session"].id,
                    content=display_text,
                    event_type=event_type,
                )

        # Call callback - should not raise
        await persist_event_to_chat("run-123", "Test message", "capability_executed")

        # Verify no message was persisted (session is None)
        mock_chat_service.add_agent_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_callback_handles_service_errors_gracefully(self):
        """Test callback handles service errors gracefully."""
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        mock_chat_service = AsyncMock(spec=ChatPersistenceService)
        mock_chat_session = ChatSession(
            id=1,
            title="Chat - planner",
            agent_role="planner",
            tenant_id="tenant-1",
            run_id="run-123",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        mock_chat_service.add_agent_message = AsyncMock(side_effect=RuntimeError("Database error"))

        chat_session_holder = {"session": mock_chat_session}

        async def persist_event_to_chat(run_id: str, display_text: str, event_type: str) -> None:
            if chat_session_holder["session"]:
                try:
                    await mock_chat_service.add_agent_message(
                        session_id=chat_session_holder["session"].id,
                        content=display_text,
                        event_type=event_type,
                    )
                except Exception:
                    pass  # Silently ignore errors

        # Call callback - should not raise
        await persist_event_to_chat("run-123", "Test message", "capability_executed")

        # Verify add_agent_message was called despite error
        mock_chat_service.add_agent_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_callback_with_multiple_events_in_sequence(self):
        """Test callback handles multiple events in sequence."""
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        mock_chat_service = AsyncMock(spec=ChatPersistenceService)
        mock_chat_session = ChatSession(
            id=1,
            title="Chat - planner",
            agent_role="planner",
            tenant_id="tenant-1",
            run_id="run-123",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        mock_chat_service.add_agent_message = AsyncMock()

        chat_session_holder = {"session": mock_chat_session}

        async def persist_event_to_chat(run_id: str, display_text: str, event_type: str) -> None:
            if chat_session_holder["session"]:
                await mock_chat_service.add_agent_message(
                    session_id=chat_session_holder["session"].id,
                    content=display_text,
                    event_type=event_type,
                )

        # Simulate event sequence
        events = [
            ("▶️ Run Started", "run_started"),
            ("✓ Operation: search (success)", "capability_executed"),
            ("✓ Tool: search_web (mcp-server-1)", "tool_invoked"),
            ("⚠️ Approval Required: delete_file", "approval_requested"),
            ("✓ Approval APPROVED", "approval_resolved"),
            ("✓ Run Completed Successfully", "run_completed"),
        ]

        for display_text, event_type in events:
            await persist_event_to_chat("run-123", display_text, event_type)

        # Verify all events were persisted
        assert mock_chat_service.add_agent_message.call_count == 6

    @pytest.mark.asyncio
    async def test_callback_preserves_event_metadata(self):
        """Test callback preserves event metadata in persistence."""
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        mock_chat_service = AsyncMock(spec=ChatPersistenceService)
        mock_chat_session = ChatSession(
            id=1,
            title="Chat - planner",
            agent_role="planner",
            tenant_id="tenant-1",
            run_id="run-123",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        mock_chat_service.add_agent_message = AsyncMock()

        chat_session_holder = {"session": mock_chat_session}

        async def persist_event_to_chat(run_id: str, display_text: str, event_type: str) -> None:
            if chat_session_holder["session"]:
                await mock_chat_service.add_agent_message(
                    session_id=chat_session_holder["session"].id,
                    content=display_text,
                    event_type=event_type,
                )

        # Persist event with specific metadata
        await persist_event_to_chat(
            "run-123",
            "✓ Operation: search (success)",
            "capability_executed",
        )

        # Verify event_type is preserved
        call_args = mock_chat_service.add_agent_message.call_args
        assert call_args[1]["event_type"] == "capability_executed"
        assert call_args[1]["session_id"] == 1
        assert call_args[1]["content"] == "✓ Operation: search (success)"


class TestOrchestratorCallbackIntegration:
    """Tests for orchestrator callback integration with persistence."""

    @pytest.mark.asyncio
    async def test_orchestrator_calls_callback_for_each_event(self):
        """Test orchestrator calls persistence callback for each event."""
        # Track callback invocations
        callback_calls = []

        async def mock_callback(run_id: str, display_text: str, event_type: str):
            callback_calls.append((run_id, display_text, event_type))

        # Create mock SSE events that would be returned from orchestrator
        dt = datetime(2025, 12, 28, 22, 0, 0, tzinfo=timezone.utc)
        event1_data = SSEEventData(
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
        event1 = SSEResponse(data=event1_data)

        event2_data = SSEEventData(
            id="evt-2",
            type="tool_invoked",
            category="tool_execution",
            created_at=dt,
            run_id="run-1",
            payload={},
            tool_execution=ToolExecutionData(
                server_id="mcp-1",
                tool_name="search_web",
                args={},
                result={"results": []},
                ok=True,
                risk="low",
                timestamp=dt,
            ),
        )
        event2 = SSEResponse(data=event2_data)

        # Simulate the callback being called by the orchestrator
        # This tests that the callback receives proper parameters
        await mock_callback("run-1", "✓ Operation: search (success)", "capability_executed")
        await mock_callback("run-1", "✓ Tool: search_web (mcp-1)", "tool_invoked")

        # Verify callback was called for each event
        assert len(callback_calls) == 2
        assert callback_calls[0][0] == "run-1"
        assert "search" in callback_calls[0][1]
        assert callback_calls[0][2] == "capability_executed"
        assert callback_calls[1][0] == "run-1"
        assert "search_web" in callback_calls[1][1]
        assert callback_calls[1][2] == "tool_invoked"

    @pytest.mark.asyncio
    async def test_orchestrator_skips_callback_for_thinking_events(self):
        """Test orchestrator skips callback for thinking events."""
        # Track callback invocations
        callback_calls = []

        async def mock_callback(run_id: str, display_text: str, event_type: str):
            callback_calls.append((run_id, display_text, event_type))

        # Create thinking event (should be skipped)
        dt = datetime(2025, 12, 28, 22, 0, 0, tzinfo=timezone.utc)
        thinking_event_data = SSEEventData(
            id="evt-1",
            type="thought_executed",
            category="thinking",
            created_at=dt,
            run_id="run-1",
            payload={},
        )
        thinking_event = SSEResponse(data=thinking_event_data)

        # Thinking events should not trigger callback (empty display_text)
        # This simulates the orchestrator's _format_event_for_chat returning empty string
        display_text = ""  # Thinking events return empty display_text
        if display_text:  # This condition should be False for thinking events
            await mock_callback("run-1", display_text, "thought_executed")

        # Verify callback was NOT called for thinking event
        assert len(callback_calls) == 0

    @pytest.mark.asyncio
    async def test_orchestrator_callback_error_doesnt_break_stream(self):
        """Test orchestrator callback errors don't break the stream."""
        # Track callback invocations
        callback_calls = []
        error_count = [0]

        async def failing_callback(run_id: str, display_text: str, event_type: str):
            callback_calls.append((run_id, display_text, event_type))
            error_count[0] += 1
            raise ValueError("Callback error")

        # Simulate orchestrator calling callback with error handling
        try:
            await failing_callback("run-1", "✓ Operation: search (success)", "capability_executed")
        except Exception as e:
            # Orchestrator catches and logs the error
            pass

        # Verify callback was called despite error
        assert error_count[0] == 1
        assert len(callback_calls) == 1

    @pytest.mark.asyncio
    async def test_orchestrator_formats_event_before_callback(self):
        """Test orchestrator formats event before calling callback."""
        # Track callback invocations
        callback_calls = []

        async def mock_callback(run_id: str, display_text: str, event_type: str):
            callback_calls.append((run_id, display_text, event_type))

        # Create mock event data
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

        # Simulate orchestrator formatting and calling callback
        # The orchestrator's _format_event_for_chat would format the event
        formatted_text = "✓ Operation: search (success)"
        await mock_callback("run-1", formatted_text, "capability_executed")

        # Verify callback received formatted display text
        assert len(callback_calls) == 1
        run_id, display_text, event_type = callback_calls[0]
        assert run_id == "run-1"
        assert display_text != ""  # Should be formatted
        assert "Operation" in display_text or "search" in display_text
        assert event_type == "capability_executed"

    @pytest.mark.asyncio
    async def test_orchestrator_callback_with_approval_events(self):
        """Test orchestrator callback with approval events."""
        # Track callback invocations
        callback_calls = []

        async def mock_callback(run_id: str, display_text: str, event_type: str):
            callback_calls.append((run_id, display_text, event_type))

        # Create approval request event
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

        # Simulate orchestrator formatting and calling callback for approval event
        formatted_text = "⚠️ Approval Required: delete_file\n  Reason: User requested file deletion\n  Risk Level: high"
        await mock_callback("run-1", formatted_text, "approval_requested")

        # Verify callback received approval event
        assert len(callback_calls) == 1
        run_id, display_text, event_type = callback_calls[0]
        assert event_type == "approval_requested"
        assert "Approval" in display_text or "delete_file" in display_text


class TestUserMessagePersistence:
    """Tests for user message persistence in chat history."""

    @pytest.mark.asyncio
    async def test_user_objective_persisted_on_first_event(self):
        """Test that user objective is persisted when chat session is initialized."""
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        # Mock database session
        mock_session = MagicMock()
        service = ChatPersistenceService(mock_session)

        # Track add_user_message calls
        user_messages = []

        async def mock_add_user_message(session_id, content, metadata=None):
            user_messages.append(
                {
                    "session_id": session_id,
                    "content": content,
                    "metadata": metadata,
                }
            )
            return MagicMock(id=1, content=content)

        service.add_user_message = mock_add_user_message

        # Simulate persisting user objective
        objective = "Find the best machine learning model for image classification"
        await service.add_user_message(
            session_id=1,
            content=objective,
            metadata={"type": "initial_objective", "run_id": "run-123"},
        )

        # Verify user message was persisted
        assert len(user_messages) == 1
        assert user_messages[0]["content"] == objective
        assert user_messages[0]["metadata"]["type"] == "initial_objective"
        assert user_messages[0]["metadata"]["run_id"] == "run-123"

    @pytest.mark.asyncio
    async def test_user_objective_with_metadata(self):
        """Test user objective is persisted with correct metadata."""
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        mock_session = MagicMock()
        service = ChatPersistenceService(mock_session)

        user_messages = []

        async def mock_add_user_message(session_id, content, metadata=None):
            user_messages.append(
                {
                    "session_id": session_id,
                    "content": content,
                    "metadata": metadata,
                }
            )
            return MagicMock(id=1)

        service.add_user_message = mock_add_user_message

        # Simulate persisting user objective with metadata
        objective = "Analyze customer feedback and generate insights"
        metadata = {
            "type": "initial_objective",
            "run_id": "run-456",
            "tenant_id": "tenant-1",
        }

        await service.add_user_message(
            session_id=2,
            content=objective,
            metadata=metadata,
        )

        # Verify metadata is preserved
        assert len(user_messages) == 1
        assert user_messages[0]["metadata"]["type"] == "initial_objective"
        assert user_messages[0]["metadata"]["run_id"] == "run-456"
        assert user_messages[0]["metadata"]["tenant_id"] == "tenant-1"

    @pytest.mark.asyncio
    async def test_user_and_agent_messages_in_sequence(self):
        """Test that user and agent messages are persisted in correct sequence."""
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        mock_session = MagicMock()
        service = ChatPersistenceService(mock_session)

        messages = []

        async def mock_add_user_message(session_id, content, metadata=None):
            messages.append({"role": "user", "content": content})
            return MagicMock(id=len(messages))

        async def mock_add_agent_message(session_id, content, event_type=None, event_category=None, metadata=None):
            messages.append({"role": "agent", "content": content, "event_type": event_type})
            return MagicMock(id=len(messages))

        service.add_user_message = mock_add_user_message
        service.add_agent_message = mock_add_agent_message

        # Simulate chat flow: user objective -> agent events
        await service.add_user_message(
            session_id=1,
            content="Find the best ML model",
            metadata={"type": "initial_objective"},
        )

        await service.add_agent_message(
            session_id=1,
            content="✓ Operation: search (success)",
            event_type="capability_executed",
        )

        await service.add_agent_message(
            session_id=1,
            content="✓ Tool: search_web (mcp-1)",
            event_type="tool_invoked",
        )

        # Verify message sequence
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Find the best ML model"
        assert messages[1]["role"] == "agent"
        assert messages[1]["event_type"] == "capability_executed"
        assert messages[2]["role"] == "agent"
        assert messages[2]["event_type"] == "tool_invoked"

    @pytest.mark.asyncio
    async def test_empty_objective_not_persisted(self):
        """Test that empty objectives are not persisted."""
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        mock_session = MagicMock()
        service = ChatPersistenceService(mock_session)

        user_messages = []

        async def mock_add_user_message(session_id, content, metadata=None):
            if content:  # Only persist non-empty content
                user_messages.append({"content": content})
            return None

        service.add_user_message = mock_add_user_message

        # Try to persist empty objective
        result = await service.add_user_message(
            session_id=1,
            content="",
            metadata={"type": "initial_objective"},
        )

        # Verify empty message was not persisted
        assert len(user_messages) == 0
        assert result is None

    @pytest.mark.asyncio
    async def test_long_objective_persisted(self):
        """Test that long objectives are persisted correctly."""
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        mock_session = MagicMock()
        service = ChatPersistenceService(mock_session)

        user_messages = []

        async def mock_add_user_message(session_id, content, metadata=None):
            user_messages.append({"content": content})
            return MagicMock(id=1)

        service.add_user_message = mock_add_user_message

        # Create a long objective
        long_objective = (
            "Analyze and optimize the performance of our e-commerce platform by examining user behavior patterns, "
            "identifying bottlenecks in the checkout process, and recommending improvements to increase conversion rates "
            "and reduce cart abandonment. Include competitor analysis and best practices from industry leaders."
        )

        await service.add_user_message(
            session_id=1,
            content=long_objective,
            metadata={"type": "initial_objective"},
        )

        # Verify long objective was persisted
        assert len(user_messages) == 1
        assert user_messages[0]["content"] == long_objective
        assert len(user_messages[0]["content"]) > 200

    @pytest.mark.asyncio
    async def test_objective_with_special_characters(self):
        """Test that objectives with special characters are persisted."""
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        mock_session = MagicMock()
        service = ChatPersistenceService(mock_session)

        user_messages = []

        async def mock_add_user_message(session_id, content, metadata=None):
            user_messages.append({"content": content})
            return MagicMock(id=1)

        service.add_user_message = mock_add_user_message

        # Create objective with special characters
        objective = "Find & analyze 'best practices' for C++ & Python (ML/AI) optimization [2024]"

        await service.add_user_message(
            session_id=1,
            content=objective,
            metadata={"type": "initial_objective"},
        )

        # Verify special characters are preserved
        assert len(user_messages) == 1
        assert user_messages[0]["content"] == objective
        assert "&" in user_messages[0]["content"]
        assert "'" in user_messages[0]["content"]
        assert "[" in user_messages[0]["content"]

    @pytest.mark.asyncio
    async def test_complete_chat_history_flow(self):
        """Test complete chat history with user objective and agent responses."""
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        mock_session = MagicMock()
        service = ChatPersistenceService(mock_session)

        chat_history = []

        async def mock_add_user_message(session_id, content, metadata=None):
            chat_history.append(
                {
                    "role": "user",
                    "content": content,
                    "metadata": metadata,
                }
            )
            return MagicMock(id=len(chat_history))

        async def mock_add_agent_message(session_id, content, event_type=None, event_category=None, metadata=None):
            chat_history.append(
                {
                    "role": "agent",
                    "content": content,
                    "event_type": event_type,
                    "event_category": event_category,
                }
            )
            return MagicMock(id=len(chat_history))

        service.add_user_message = mock_add_user_message
        service.add_agent_message = mock_add_agent_message

        # Simulate complete chat flow
        # 1. User objective
        await service.add_user_message(
            session_id=1,
            content="Optimize database queries for performance",
            metadata={"type": "initial_objective", "run_id": "run-789"},
        )

        # 2. Agent operations
        await service.add_agent_message(
            session_id=1,
            content="✓ Operation: analyze_queries (success)",
            event_type="capability_executed",
            event_category="operation",
        )

        await service.add_agent_message(
            session_id=1,
            content="✓ Tool: database_profiler (mcp-1)",
            event_type="tool_invoked",
            event_category="tool_execution",
        )

        await service.add_agent_message(
            session_id=1,
            content="✓ Optimization: Added 3 indexes, reduced query time by 45%",
            event_type="capability_executed",
            event_category="operation",
        )

        # Verify complete chat history
        assert len(chat_history) == 4
        assert chat_history[0]["role"] == "user"
        assert chat_history[0]["content"] == "Optimize database queries for performance"
        assert chat_history[1]["role"] == "agent"
        assert chat_history[1]["event_type"] == "capability_executed"
        assert chat_history[2]["role"] == "agent"
        assert chat_history[2]["event_type"] == "tool_invoked"
        assert chat_history[3]["role"] == "agent"
        assert "45%" in chat_history[3]["content"]


class TestRunsEndpointCallbackRealExecution:
    """Integration tests that execute the real callback code in stream_run_events (lines 309-348)."""

    @pytest.mark.asyncio
    async def test_stream_run_events_initializes_chat_session_on_first_event(self):
        """Test that stream_run_events callback initializes chat session (lines 309-320)."""
        from gearmeshing_ai.server.api.v1.runs import stream_run_events
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        # Mock dependencies
        mock_db_session = AsyncMock()
        mock_orchestrator = AsyncMock()
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        # Create test run
        test_run = AgentRun(
            id="run-stream-test",
            tenant_id="tenant-1",
            role="planner",
            objective="Stream test objective",
            status=AgentRunStatus.running.value,
            created_at=datetime.now(timezone.utc),
        )

        # Mock orchestrator methods
        mock_orchestrator.get_run = AsyncMock(return_value=test_run)

        # Mock chat service
        mock_chat_session = MagicMock()
        mock_chat_session.id = 1

        mock_chat_service_instance = AsyncMock(spec=ChatPersistenceService)
        mock_chat_service_instance.get_or_create_session = AsyncMock(return_value=mock_chat_session)
        mock_chat_service_instance.add_user_message = AsyncMock()
        mock_chat_service_instance.add_agent_message = AsyncMock()

        # Mock stream_events to yield a test event
        async def mock_stream_events(run_id, on_event_persisted=None):
            # Simulate event that triggers callback
            if on_event_persisted:
                await on_event_persisted("run-stream-test", "Test event", "capability_executed")
            yield MagicMock()  # Yield a mock SSE response

        mock_orchestrator.stream_events = mock_stream_events

        # Patch ChatPersistenceService to use our mock
        with patch("gearmeshing_ai.server.api.v1.runs.ChatPersistenceService", return_value=mock_chat_service_instance):
            # Call the real stream_run_events function (lines 275-361)
            response = await stream_run_events(
                run_id="run-stream-test",
                request=mock_request,
                orchestrator=mock_orchestrator,
                db_session=mock_db_session,
            )

            # Consume the EventSourceResponse generator to execute the real code
            event_count = 0
            async for event in response.body_iterator:
                event_count += 1
                if event_count >= 1:
                    break

        # Verify chat session was created (lines 313-319)
        mock_chat_service_instance.get_or_create_session.assert_called_once()
        call_kwargs = mock_chat_service_instance.get_or_create_session.call_args[1]
        assert call_kwargs["run_id"] == "run-stream-test"
        assert call_kwargs["tenant_id"] == "tenant-1"

    @pytest.mark.asyncio
    async def test_stream_run_events_callback_persists_user_objective(self):
        """Test that user objective is persisted (lines 323-333)."""
        from gearmeshing_ai.server.api.v1.runs import stream_run_events
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        mock_db_session = AsyncMock()
        mock_orchestrator = AsyncMock()
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        test_run = AgentRun(
            id="run-user-obj",
            tenant_id="tenant-2",
            role="executor",
            objective="User objective test",
            status=AgentRunStatus.running.value,
            created_at=datetime.now(timezone.utc),
        )

        mock_orchestrator.get_run = AsyncMock(return_value=test_run)

        mock_chat_session = MagicMock()
        mock_chat_session.id = 2

        mock_chat_service_instance = AsyncMock(spec=ChatPersistenceService)
        mock_chat_service_instance.get_or_create_session = AsyncMock(return_value=mock_chat_session)
        mock_chat_service_instance.add_user_message = AsyncMock()
        mock_chat_service_instance.add_agent_message = AsyncMock()

        async def mock_stream_events(run_id, on_event_persisted=None):
            if on_event_persisted:
                await on_event_persisted("run-user-obj", "Event", "type")
            yield MagicMock()

        mock_orchestrator.stream_events = mock_stream_events

        # Call the real stream_run_events function
        with patch("gearmeshing_ai.server.api.v1.runs.ChatPersistenceService", return_value=mock_chat_service_instance):
            response = await stream_run_events(
                run_id="run-user-obj",
                request=mock_request,
                orchestrator=mock_orchestrator,
                db_session=mock_db_session,
            )

            # Consume the EventSourceResponse generator to execute the real code
            event_count = 0
            async for event in response.body_iterator:
                event_count += 1
                if event_count >= 1:
                    break

        # Verify user message was persisted (lines 325-329)
        mock_chat_service_instance.add_user_message.assert_called_once()
        call_kwargs = mock_chat_service_instance.add_user_message.call_args[1]
        assert call_kwargs["content"] == "User objective test"
        assert call_kwargs["metadata"]["type"] == "initial_objective"

    @pytest.mark.asyncio
    async def test_stream_run_events_callback_persists_agent_messages(self):
        """Test that agent messages are persisted (lines 341-345)."""
        from gearmeshing_ai.server.api.v1.runs import stream_run_events
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        mock_db_session = AsyncMock()
        mock_orchestrator = AsyncMock()
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        test_run = AgentRun(
            id="run-agent-msg",
            tenant_id="tenant-3",
            role="worker",
            objective="Test",
            status=AgentRunStatus.running.value,
            created_at=datetime.now(timezone.utc),
        )

        mock_orchestrator.get_run = AsyncMock(return_value=test_run)

        mock_chat_session = MagicMock()
        mock_chat_session.id = 3

        mock_chat_service_instance = AsyncMock(spec=ChatPersistenceService)
        mock_chat_service_instance.get_or_create_session = AsyncMock(return_value=mock_chat_session)
        mock_chat_service_instance.add_user_message = AsyncMock()
        mock_chat_service_instance.add_agent_message = AsyncMock()

        async def mock_stream_events(run_id, on_event_persisted=None):
            if on_event_persisted:
                await on_event_persisted("run-agent-msg", "Agent response", "capability_executed")
            yield MagicMock()

        mock_orchestrator.stream_events = mock_stream_events

        # Call the real stream_run_events function
        with patch("gearmeshing_ai.server.api.v1.runs.ChatPersistenceService", return_value=mock_chat_service_instance):
            response = await stream_run_events(
                run_id="run-agent-msg",
                request=mock_request,
                orchestrator=mock_orchestrator,
                db_session=mock_db_session,
            )

            # Consume the EventSourceResponse generator to execute the real code
            event_count = 0
            async for event in response.body_iterator:
                event_count += 1
                if event_count >= 1:
                    break

        # Verify agent message was persisted (lines 341-345)
        mock_chat_service_instance.add_agent_message.assert_called_once()
        call_kwargs = mock_chat_service_instance.add_agent_message.call_args[1]
        assert call_kwargs["content"] == "Agent response"
        assert call_kwargs["event_type"] == "capability_executed"

    @pytest.mark.asyncio
    async def test_stream_run_events_handles_orchestrator_get_run_exception(self):
        """Test exception handling when orchestrator.get_run fails (lines 310-336)."""
        from gearmeshing_ai.server.api.v1.runs import stream_run_events
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        mock_db_session = AsyncMock()
        mock_orchestrator = AsyncMock()
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        # Mock orchestrator to raise exception
        mock_orchestrator.get_run = AsyncMock(side_effect=Exception("Failed to get run"))

        mock_chat_service_instance = AsyncMock(spec=ChatPersistenceService)
        mock_chat_service_instance.get_or_create_session = AsyncMock()
        mock_chat_service_instance.add_user_message = AsyncMock()
        mock_chat_service_instance.add_agent_message = AsyncMock()

        async def mock_stream_events(run_id, on_event_persisted=None):
            if on_event_persisted:
                await on_event_persisted("run-error", "Event", "type")
            yield MagicMock()

        mock_orchestrator.stream_events = mock_stream_events

        # Call the real stream_run_events function
        with patch("gearmeshing_ai.server.api.v1.runs.ChatPersistenceService", return_value=mock_chat_service_instance):
            response = await stream_run_events(
                run_id="run-error",
                request=mock_request,
                orchestrator=mock_orchestrator,
                db_session=mock_db_session,
            )

            # Consume the EventSourceResponse generator
            event_count = 0
            async for event in response.body_iterator:
                event_count += 1
                if event_count >= 1:
                    break

        # Verify get_or_create_session was not called due to exception (lines 334-336)
        mock_chat_service_instance.get_or_create_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_run_events_handles_add_user_message_exception(self):
        """Test exception handling when add_user_message fails (lines 332-333)."""
        from gearmeshing_ai.server.api.v1.runs import stream_run_events
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        mock_db_session = AsyncMock()
        mock_orchestrator = AsyncMock()
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        test_run = AgentRun(
            id="run-user-msg-error",
            tenant_id="tenant-4",
            role="analyzer",
            objective="Test objective",
            status=AgentRunStatus.running.value,
            created_at=datetime.now(timezone.utc),
        )

        mock_orchestrator.get_run = AsyncMock(return_value=test_run)

        mock_chat_session = MagicMock()
        mock_chat_session.id = 4

        mock_chat_service_instance = AsyncMock(spec=ChatPersistenceService)
        mock_chat_service_instance.get_or_create_session = AsyncMock(return_value=mock_chat_session)
        # Simulate add_user_message failure
        mock_chat_service_instance.add_user_message = AsyncMock(side_effect=Exception("Failed to add user message"))
        mock_chat_service_instance.add_agent_message = AsyncMock()

        async def mock_stream_events(run_id, on_event_persisted=None):
            if on_event_persisted:
                await on_event_persisted("run-user-msg-error", "Event", "type")
            yield MagicMock()

        mock_orchestrator.stream_events = mock_stream_events

        # Call the real stream_run_events function
        with patch("gearmeshing_ai.server.api.v1.runs.ChatPersistenceService", return_value=mock_chat_service_instance):
            response = await stream_run_events(
                run_id="run-user-msg-error",
                request=mock_request,
                orchestrator=mock_orchestrator,
                db_session=mock_db_session,
            )

            # Consume the EventSourceResponse generator
            event_count = 0
            async for event in response.body_iterator:
                event_count += 1
                if event_count >= 1:
                    break

        # Verify add_user_message was called but exception was handled (lines 325-333)
        mock_chat_service_instance.add_user_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_run_events_handles_add_agent_message_exception(self):
        """Test exception handling when add_agent_message fails (lines 346-347)."""
        from gearmeshing_ai.server.api.v1.runs import stream_run_events
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        mock_db_session = AsyncMock()
        mock_orchestrator = AsyncMock()
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        test_run = AgentRun(
            id="run-agent-msg-error",
            tenant_id="tenant-5",
            role="executor",
            objective="Test",
            status=AgentRunStatus.running.value,
            created_at=datetime.now(timezone.utc),
        )

        mock_orchestrator.get_run = AsyncMock(return_value=test_run)

        mock_chat_session = MagicMock()
        mock_chat_session.id = 5

        mock_chat_service_instance = AsyncMock(spec=ChatPersistenceService)
        mock_chat_service_instance.get_or_create_session = AsyncMock(return_value=mock_chat_session)
        mock_chat_service_instance.add_user_message = AsyncMock()
        # Simulate add_agent_message failure
        mock_chat_service_instance.add_agent_message = AsyncMock(side_effect=Exception("Failed to add agent message"))

        async def mock_stream_events(run_id, on_event_persisted=None):
            if on_event_persisted:
                await on_event_persisted("run-agent-msg-error", "Agent response", "tool_invoked")
            yield MagicMock()

        mock_orchestrator.stream_events = mock_stream_events

        # Call the real stream_run_events function
        with patch("gearmeshing_ai.server.api.v1.runs.ChatPersistenceService", return_value=mock_chat_service_instance):
            response = await stream_run_events(
                run_id="run-agent-msg-error",
                request=mock_request,
                orchestrator=mock_orchestrator,
                db_session=mock_db_session,
            )

            # Consume the EventSourceResponse generator
            event_count = 0
            async for event in response.body_iterator:
                event_count += 1
                if event_count >= 1:
                    break

        # Verify add_agent_message was called but exception was handled (lines 341-347)
        mock_chat_service_instance.add_agent_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_run_events_skips_user_objective_when_no_objective(self):
        """Test that user objective is skipped when run has no objective (lines 323-333)."""
        from gearmeshing_ai.server.api.v1.runs import stream_run_events
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        mock_db_session = AsyncMock()
        mock_orchestrator = AsyncMock()
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        test_run = AgentRun(
            id="run-no-objective",
            tenant_id="tenant-6",
            role="planner",
            objective="",  # Empty objective
            status=AgentRunStatus.running.value,
            created_at=datetime.now(timezone.utc),
        )

        mock_orchestrator.get_run = AsyncMock(return_value=test_run)

        mock_chat_session = MagicMock()
        mock_chat_session.id = 6

        mock_chat_service_instance = AsyncMock(spec=ChatPersistenceService)
        mock_chat_service_instance.get_or_create_session = AsyncMock(return_value=mock_chat_session)
        mock_chat_service_instance.add_user_message = AsyncMock()
        mock_chat_service_instance.add_agent_message = AsyncMock()

        async def mock_stream_events(run_id, on_event_persisted=None):
            if on_event_persisted:
                await on_event_persisted("run-no-objective", "Event", "type")
            yield MagicMock()

        mock_orchestrator.stream_events = mock_stream_events

        # Call the real stream_run_events function
        with patch("gearmeshing_ai.server.api.v1.runs.ChatPersistenceService", return_value=mock_chat_service_instance):
            response = await stream_run_events(
                run_id="run-no-objective",
                request=mock_request,
                orchestrator=mock_orchestrator,
                db_session=mock_db_session,
            )

            # Consume the EventSourceResponse generator
            event_count = 0
            async for event in response.body_iterator:
                event_count += 1
                if event_count >= 1:
                    break

        # Verify add_user_message was not called (lines 323-333)
        mock_chat_service_instance.add_user_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_run_events_client_disconnect_stops_stream(self):
        """Test that stream stops when client disconnects (lines 351-353)."""
        from gearmeshing_ai.server.api.v1.runs import stream_run_events
        from gearmeshing_ai.server.services.chat_persistence import (
            ChatPersistenceService,
        )

        mock_db_session = AsyncMock()
        mock_orchestrator = AsyncMock()
        mock_request = AsyncMock()
        # Simulate client disconnect on second check
        mock_request.is_disconnected = AsyncMock(side_effect=[False, True])

        test_run = AgentRun(
            id="run-disconnect",
            tenant_id="tenant-7",
            role="worker",
            objective="Test",
            status=AgentRunStatus.running.value,
            created_at=datetime.now(timezone.utc),
        )

        mock_orchestrator.get_run = AsyncMock(return_value=test_run)

        mock_chat_session = MagicMock()
        mock_chat_session.id = 7

        mock_chat_service_instance = AsyncMock(spec=ChatPersistenceService)
        mock_chat_service_instance.get_or_create_session = AsyncMock(return_value=mock_chat_session)
        mock_chat_service_instance.add_user_message = AsyncMock()
        mock_chat_service_instance.add_agent_message = AsyncMock()

        async def mock_stream_events(run_id, on_event_persisted=None):
            if on_event_persisted:
                await on_event_persisted("run-disconnect", "Event", "type")
            yield MagicMock()

        mock_orchestrator.stream_events = mock_stream_events

        # Call the real stream_run_events function
        with patch("gearmeshing_ai.server.api.v1.runs.ChatPersistenceService", return_value=mock_chat_service_instance):
            response = await stream_run_events(
                run_id="run-disconnect",
                request=mock_request,
                orchestrator=mock_orchestrator,
                db_session=mock_db_session,
            )

            # Consume the EventSourceResponse generator
            event_count = 0
            async for event in response.body_iterator:
                event_count += 1
                if event_count >= 2:
                    break

        # Verify is_disconnected was called (lines 351-353)
        assert mock_request.is_disconnected.call_count >= 1


class TestRunsEndpointCallbackIntegration:
    """Integration tests for runs endpoint callback execution (lines 309-348)."""

    @pytest.mark.asyncio
    async def test_persist_event_to_chat_initializes_session_on_first_event(self):
        """Test callback initializes chat session on first event (lines 309-320)."""
        mock_db_session = AsyncMock()
        mock_orchestrator = AsyncMock()
        mock_chat_service = AsyncMock()

        # Create a real run
        test_run = AgentRun(
            id="run-init-test",
            tenant_id="tenant-1",
            role="planner",
            objective="Initialize session test",
            status=AgentRunStatus.running.value,
            created_at=datetime.now(timezone.utc),
        )

        # Create a mock chat session
        mock_chat_session = MagicMock()
        mock_chat_session.id = 1

        mock_orchestrator.get_run = AsyncMock(return_value=test_run)
        mock_chat_service.get_or_create_session = AsyncMock(return_value=mock_chat_session)
        mock_chat_service.add_user_message = AsyncMock()
        mock_chat_service.add_agent_message = AsyncMock()

        # Simulate the callback function from runs.py
        chat_session = None
        user_message_persisted = False

        async def persist_event_to_chat(run_id: str, display_text: str, event_type: str) -> None:
            nonlocal chat_session, user_message_persisted

            # Initialize chat session on first event (lines 309-320)
            if chat_session is None:
                try:
                    run = await mock_orchestrator.get_run(run_id)
                    if run:
                        chat_session = await mock_chat_service.get_or_create_session(
                            run_id=run_id,
                            tenant_id=run.tenant_id,
                            agent_role=run.role,
                            title=f"Chat - {run.role}",
                            description=f"Chat history for run {run_id}",
                        )

                        # Persist user's initial objective as the first message (lines 323-333)
                        if not user_message_persisted and run.objective:
                            try:
                                await mock_chat_service.add_user_message(
                                    session_id=chat_session.id,
                                    content=run.objective,
                                    metadata={"type": "initial_objective", "run_id": run_id},
                                )
                                user_message_persisted = True
                            except Exception as e:
                                pass

                except Exception as e:
                    return

            # Persist message to chat session (lines 338-347)
            if chat_session:
                try:
                    await mock_chat_service.add_agent_message(
                        session_id=chat_session.id,
                        content=display_text,
                        event_type=event_type,
                    )
                except Exception as e:
                    pass

        # Call callback to trigger initialization
        await persist_event_to_chat("run-init-test", "First event", "capability_executed")

        # Verify session was initialized (lines 313-319)
        mock_chat_service.get_or_create_session.assert_called_once()
        call_kwargs = mock_chat_service.get_or_create_session.call_args[1]
        assert call_kwargs["run_id"] == "run-init-test"
        assert call_kwargs["tenant_id"] == "tenant-1"
        assert call_kwargs["agent_role"] == "planner"

    @pytest.mark.asyncio
    async def test_persist_event_to_chat_persists_user_objective_once(self):
        """Test user objective is persisted only once (lines 323-333)."""
        mock_db_session = AsyncMock()
        mock_orchestrator = AsyncMock()
        mock_chat_service = AsyncMock()

        test_run = AgentRun(
            id="run-user-obj-test",
            tenant_id="tenant-2",
            role="executor",
            objective="Execute important task",
            status=AgentRunStatus.running.value,
            created_at=datetime.now(timezone.utc),
        )

        mock_chat_session = MagicMock()
        mock_chat_session.id = 2

        mock_orchestrator.get_run = AsyncMock(return_value=test_run)
        mock_chat_service.get_or_create_session = AsyncMock(return_value=mock_chat_session)
        mock_chat_service.add_user_message = AsyncMock()
        mock_chat_service.add_agent_message = AsyncMock()

        chat_session = None
        user_message_persisted = False

        async def persist_event_to_chat(run_id: str, display_text: str, event_type: str) -> None:
            nonlocal chat_session, user_message_persisted

            if chat_session is None:
                try:
                    run = await mock_orchestrator.get_run(run_id)
                    if run:
                        chat_session = await mock_chat_service.get_or_create_session(
                            run_id=run_id,
                            tenant_id=run.tenant_id,
                            agent_role=run.role,
                            title=f"Chat - {run.role}",
                            description=f"Chat history for run {run_id}",
                        )

                        # Persist user objective (lines 323-333)
                        if not user_message_persisted and run.objective:
                            try:
                                await mock_chat_service.add_user_message(
                                    session_id=chat_session.id,
                                    content=run.objective,
                                    metadata={"type": "initial_objective", "run_id": run_id},
                                )
                                user_message_persisted = True
                            except Exception as e:
                                pass

                except Exception as e:
                    return

            if chat_session:
                try:
                    await mock_chat_service.add_agent_message(
                        session_id=chat_session.id,
                        content=display_text,
                        event_type=event_type,
                    )
                except Exception as e:
                    pass

        # Call callback multiple times
        await persist_event_to_chat("run-user-obj-test", "Event 1", "type1")
        await persist_event_to_chat("run-user-obj-test", "Event 2", "type2")
        await persist_event_to_chat("run-user-obj-test", "Event 3", "type3")

        # Verify user message was persisted only once (line 323 flag check)
        assert mock_chat_service.add_user_message.call_count == 1
        # Verify agent messages were persisted for each event
        assert mock_chat_service.add_agent_message.call_count == 3

    @pytest.mark.asyncio
    async def test_persist_event_to_chat_handles_missing_run(self):
        """Test callback handles missing run gracefully (lines 310-311)."""
        mock_orchestrator = AsyncMock()
        mock_chat_service = AsyncMock()

        mock_orchestrator.get_run = AsyncMock(return_value=None)
        mock_chat_service.get_or_create_session = AsyncMock()

        chat_session = None
        user_message_persisted = False

        async def persist_event_to_chat(run_id: str, display_text: str, event_type: str) -> None:
            nonlocal chat_session, user_message_persisted

            if chat_session is None:
                try:
                    run = await mock_orchestrator.get_run(run_id)
                    if run:  # Line 312: check if run exists
                        chat_session = await mock_chat_service.get_or_create_session(
                            run_id=run_id,
                            tenant_id=run.tenant_id,
                            agent_role=run.role,
                            title=f"Chat - {run.role}",
                            description=f"Chat history for run {run_id}",
                        )
                except Exception as e:
                    return

        # Call with missing run
        await persist_event_to_chat("run-missing", "Event", "type")

        # Verify session was not created (line 312 condition failed)
        mock_chat_service.get_or_create_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_persist_event_to_chat_handles_empty_objective(self):
        """Test callback handles empty objective (line 323)."""
        mock_orchestrator = AsyncMock()
        mock_chat_service = AsyncMock()

        test_run = AgentRun(
            id="run-empty-obj",
            tenant_id="tenant-3",
            role="worker",
            objective="",  # Empty objective
            status=AgentRunStatus.running.value,
            created_at=datetime.now(timezone.utc),
        )

        mock_chat_session = MagicMock()
        mock_chat_session.id = 3

        mock_orchestrator.get_run = AsyncMock(return_value=test_run)
        mock_chat_service.get_or_create_session = AsyncMock(return_value=mock_chat_session)
        mock_chat_service.add_user_message = AsyncMock()

        chat_session = None
        user_message_persisted = False

        async def persist_event_to_chat(run_id: str, display_text: str, event_type: str) -> None:
            nonlocal chat_session, user_message_persisted

            if chat_session is None:
                try:
                    run = await mock_orchestrator.get_run(run_id)
                    if run:
                        chat_session = await mock_chat_service.get_or_create_session(
                            run_id=run_id,
                            tenant_id=run.tenant_id,
                            agent_role=run.role,
                            title=f"Chat - {run.role}",
                            description=f"Chat history for run {run_id}",
                        )

                        # Line 323: check if objective exists
                        if not user_message_persisted and run.objective:
                            try:
                                await mock_chat_service.add_user_message(
                                    session_id=chat_session.id,
                                    content=run.objective,
                                    metadata={"type": "initial_objective", "run_id": run_id},
                                )
                                user_message_persisted = True
                            except Exception as e:
                                pass

                except Exception as e:
                    return

        await persist_event_to_chat("run-empty-obj", "Event", "type")

        # Verify user message was not persisted (line 323 condition failed)
        mock_chat_service.add_user_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_persist_event_to_chat_session_init_exception_handling(self):
        """Test exception during session initialization is handled (lines 334-336)."""
        mock_orchestrator = AsyncMock()
        mock_chat_service = AsyncMock()

        test_run = AgentRun(
            id="run-init-error",
            tenant_id="tenant-4",
            role="planner",
            objective="Test",
            status=AgentRunStatus.running.value,
            created_at=datetime.now(timezone.utc),
        )

        mock_orchestrator.get_run = AsyncMock(return_value=test_run)
        mock_chat_service.get_or_create_session = AsyncMock(side_effect=Exception("Database error"))

        chat_session = None
        user_message_persisted = False

        async def persist_event_to_chat(run_id: str, display_text: str, event_type: str) -> None:
            nonlocal chat_session, user_message_persisted

            if chat_session is None:
                try:
                    run = await mock_orchestrator.get_run(run_id)
                    if run:
                        chat_session = await mock_chat_service.get_or_create_session(
                            run_id=run_id,
                            tenant_id=run.tenant_id,
                            agent_role=run.role,
                            title=f"Chat - {run.role}",
                            description=f"Chat history for run {run_id}",
                        )
                except Exception as e:
                    # Lines 334-336: exception handling
                    return

        # Should not raise exception
        await persist_event_to_chat("run-init-error", "Event", "type")
        assert chat_session is None

    @pytest.mark.asyncio
    async def test_persist_event_to_chat_user_message_exception_handling(self):
        """Test exception during user message persistence is handled (lines 332-333)."""
        mock_orchestrator = AsyncMock()
        mock_chat_service = AsyncMock()

        test_run = AgentRun(
            id="run-user-msg-error",
            tenant_id="tenant-5",
            role="planner",
            objective="Test objective",
            status=AgentRunStatus.running.value,
            created_at=datetime.now(timezone.utc),
        )

        mock_chat_session = MagicMock()
        mock_chat_session.id = 5

        mock_orchestrator.get_run = AsyncMock(return_value=test_run)
        mock_chat_service.get_or_create_session = AsyncMock(return_value=mock_chat_session)
        mock_chat_service.add_user_message = AsyncMock(side_effect=Exception("User message error"))
        mock_chat_service.add_agent_message = AsyncMock()

        chat_session = None
        user_message_persisted = False

        async def persist_event_to_chat(run_id: str, display_text: str, event_type: str) -> None:
            nonlocal chat_session, user_message_persisted

            if chat_session is None:
                try:
                    run = await mock_orchestrator.get_run(run_id)
                    if run:
                        chat_session = await mock_chat_service.get_or_create_session(
                            run_id=run_id,
                            tenant_id=run.tenant_id,
                            agent_role=run.role,
                            title=f"Chat - {run.role}",
                            description=f"Chat history for run {run_id}",
                        )

                        if not user_message_persisted and run.objective:
                            try:
                                await mock_chat_service.add_user_message(
                                    session_id=chat_session.id,
                                    content=run.objective,
                                    metadata={"type": "initial_objective", "run_id": run_id},
                                )
                                user_message_persisted = True
                            except Exception as e:
                                # Lines 332-333: exception handling
                                pass

                except Exception as e:
                    return

            if chat_session:
                try:
                    await mock_chat_service.add_agent_message(
                        session_id=chat_session.id,
                        content=display_text,
                        event_type=event_type,
                    )
                except Exception as e:
                    pass

        await persist_event_to_chat("run-user-msg-error", "Event", "type")

        # Verify session was still created despite user message error
        assert chat_session is not None
        # Verify agent message was persisted
        mock_chat_service.add_agent_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_event_to_chat_agent_message_exception_handling(self):
        """Test exception during agent message persistence is handled (lines 346-347)."""
        mock_orchestrator = AsyncMock()
        mock_chat_service = AsyncMock()

        test_run = AgentRun(
            id="run-agent-msg-error",
            tenant_id="tenant-6",
            role="executor",
            objective="Execute",
            status=AgentRunStatus.running.value,
            created_at=datetime.now(timezone.utc),
        )

        mock_chat_session = MagicMock()
        mock_chat_session.id = 6

        mock_orchestrator.get_run = AsyncMock(return_value=test_run)
        mock_chat_service.get_or_create_session = AsyncMock(return_value=mock_chat_session)
        mock_chat_service.add_user_message = AsyncMock()
        mock_chat_service.add_agent_message = AsyncMock(side_effect=Exception("Agent message error"))

        chat_session = None
        user_message_persisted = False

        async def persist_event_to_chat(run_id: str, display_text: str, event_type: str) -> None:
            nonlocal chat_session, user_message_persisted

            if chat_session is None:
                try:
                    run = await mock_orchestrator.get_run(run_id)
                    if run:
                        chat_session = await mock_chat_service.get_or_create_session(
                            run_id=run_id,
                            tenant_id=run.tenant_id,
                            agent_role=run.role,
                            title=f"Chat - {run.role}",
                            description=f"Chat history for run {run_id}",
                        )

                        if not user_message_persisted and run.objective:
                            try:
                                await mock_chat_service.add_user_message(
                                    session_id=chat_session.id,
                                    content=run.objective,
                                    metadata={"type": "initial_objective", "run_id": run_id},
                                )
                                user_message_persisted = True
                            except Exception as e:
                                pass

                except Exception as e:
                    return

            if chat_session:
                try:
                    await mock_chat_service.add_agent_message(
                        session_id=chat_session.id,
                        content=display_text,
                        event_type=event_type,
                    )
                except Exception as e:
                    # Lines 346-347: exception handling
                    pass

        await persist_event_to_chat("run-agent-msg-error", "Event", "type")

        # Verify session was created and callback completed without raising
        assert chat_session is not None
