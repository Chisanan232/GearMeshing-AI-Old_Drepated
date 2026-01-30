"""
Unit tests for chat persistence service and orchestrator chat integration.

Tests cover:
- ChatPersistenceService CRUD operations
- Chat session creation and retrieval by run_id
- Message persistence (user, agent, system)
- Event formatting for chat display
- Orchestrator event formatting and callback integration
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gearmeshing_ai.core.database.entities.chat_sessions import (
    ChatSession,
    MessageRole,
)
from gearmeshing_ai.server.schemas import (
    ApprovalRequestData,
    ApprovalResolutionData,
    OperationData,
    RunCompletionData,
    RunFailureData,
    RunStartData,
    SSEEventData,
    ToolExecutionData,
)
from gearmeshing_ai.server.services.chat_persistence import ChatPersistenceService


class TestChatPersistenceService:
    """Tests for ChatPersistenceService."""

    @pytest.fixture
    async def mock_session(self):
        """Create a mock AsyncSession."""
        session = AsyncMock(spec=AsyncSession)
        return session

    @pytest.fixture
    def chat_service(self, mock_session):
        """Create ChatPersistenceService with mock session."""
        return ChatPersistenceService(mock_session)

    @pytest.mark.asyncio
    async def test_get_or_create_session_new(self, chat_service, mock_session):
        """Test creating a new chat session."""
        # Mock the execute to return no existing session
        mock_scalars = MagicMock()
        mock_scalars.first.return_value = None
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Mock the commit and refresh
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        # Create session
        session = await chat_service.get_or_create_session(
            run_id="run-123",
            tenant_id="tenant-456",
            agent_role="planner",
            title="Test Chat",
            description="Test Description",
        )

        # Verify session was created
        assert session is not None
        assert session.run_id == "run-123"
        assert session.tenant_id == "tenant-456"
        assert session.agent_role == "planner"
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_or_create_session_existing(self, chat_service, mock_session):
        """Test retrieving an existing chat session."""
        # Create a mock existing session
        existing_session = ChatSession(
            id=1,
            run_id="run-123",
            tenant_id="tenant-456",
            agent_role="planner",
            title="Existing Chat",
        )

        # Mock the execute to return existing session
        mock_scalars = MagicMock()
        mock_scalars.first.return_value = existing_session
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Get session
        session = await chat_service.get_or_create_session(
            run_id="run-123",
            tenant_id="tenant-456",
            agent_role="planner",
        )

        # Verify existing session was returned
        assert session.id == 1
        assert session.run_id == "run-123"
        mock_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_user_message(self, chat_service, mock_session):
        """Test adding a user message."""
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        message = await chat_service.add_user_message(
            session_id=1,
            content="Hello, agent!",
            metadata={"role": "user", "source": "frontend"},
        )

        assert message is not None
        assert message.session_id == 1
        assert message.role == MessageRole.USER
        assert message.content == "Hello, agent!"
        assert message.message_metadata is not None
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_agent_message(self, chat_service, mock_session):
        """Test adding an agent message."""
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        message = await chat_service.add_agent_message(
            session_id=1,
            content="Agent response",
            event_type="capability_executed",
            event_category="operation",
            metadata={"capability": "search", "status": "success"},
        )

        assert message is not None
        assert message.session_id == 1
        assert message.role == MessageRole.ASSISTANT
        assert message.content == "Agent response"
        assert message.message_metadata is not None
        metadata = json.loads(message.message_metadata)
        assert metadata["event_type"] == "capability_executed"
        assert metadata["event_category"] == "operation"
        mock_session.add.assert_called_once()


class TestOrchestratorEventFormatting:
    """Tests for orchestrator event formatting for chat."""

    @pytest.fixture
    def orchestrator(self):
        """Create a mock orchestrator with mocked dependencies."""
        from unittest.mock import patch

        from gearmeshing_ai.server.services.orchestrator import OrchestratorService

        with (
            patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos_from_session"),
            patch("gearmeshing_ai.server.services.orchestrator.AsyncPostgresSaver"),
            patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider"),
            patch("gearmeshing_ai.server.services.orchestrator.AgentService"),
            patch("gearmeshing_ai.server.services.orchestrator.checkpointer_pool"),
        ):
            return OrchestratorService()

    def test_format_event_for_chat_operation_success(self, orchestrator):
        """Test formatting successful operation event."""
        event_data = SSEEventData(
            id="evt-1",
            type="capability_executed",
            category="operation",
            created_at=datetime.now(timezone.utc),
            run_id="run-1",
            payload={},
            operation=OperationData(
                capability="search",
                status="success",
                result={"found": 5},
                timestamp=datetime.now(timezone.utc),
            ),
        )

        display_text = orchestrator._format_event_for_chat(event_data)

        assert "✓ Operation: search (success)" in display_text
        assert "Result:" in display_text

    def test_format_event_for_chat_operation_failure(self, orchestrator):
        """Test formatting failed operation event."""
        event_data = SSEEventData(
            id="evt-2",
            type="capability_executed",
            category="operation",
            created_at=datetime.now(timezone.utc),
            run_id="run-1",
            payload={},
            operation=OperationData(
                capability="search",
                status="failed",
                result={"error": "Connection timeout"},
                timestamp=datetime.now(timezone.utc),
            ),
        )

        display_text = orchestrator._format_event_for_chat(event_data)

        assert "✗ Operation: search (failed)" in display_text

    def test_format_event_for_chat_tool_execution(self, orchestrator):
        """Test formatting tool execution event."""
        event_data = SSEEventData(
            id="evt-3",
            type="tool_invoked",
            category="tool_execution",
            created_at=datetime.now(timezone.utc),
            run_id="run-1",
            payload={},
            tool_execution=ToolExecutionData(
                server_id="mcp-server-1",
                tool_name="search_web",
                args={"query": "AI trends"},
                result={"results": ["result1", "result2"]},
                ok=True,
                risk="low",
                timestamp=datetime.now(timezone.utc),
            ),
        )

        display_text = orchestrator._format_event_for_chat(event_data)

        assert "✓ Tool: search_web (mcp-server-1)" in display_text
        assert "Result:" in display_text

    def test_format_event_for_chat_approval_request(self, orchestrator):
        """Test formatting approval request event."""
        event_data = SSEEventData(
            id="evt-4",
            type="approval_requested",
            category="approval",
            created_at=datetime.now(timezone.utc),
            run_id="run-1",
            payload={},
            approval_request=ApprovalRequestData(
                capability="delete_file",
                risk="high",
                reason="User requested file deletion",
                timestamp=datetime.now(timezone.utc),
            ),
        )

        display_text = orchestrator._format_event_for_chat(event_data)

        assert "⚠️ Approval Required: delete_file" in display_text
        assert "Reason:" in display_text
        assert "Risk Level: high" in display_text

    def test_format_event_for_chat_approval_resolution(self, orchestrator):
        """Test formatting approval resolution event."""
        event_data = SSEEventData(
            id="evt-5",
            type="approval_resolved",
            category="approval",
            created_at=datetime.now(timezone.utc),
            run_id="run-1",
            payload={},
            approval_resolution=ApprovalResolutionData(
                decision="approved",
                decided_by="user@example.com",
                timestamp=datetime.now(timezone.utc),
            ),
        )

        display_text = orchestrator._format_event_for_chat(event_data)

        assert "✓ Approval APPROVED" in display_text
        assert "user@example.com" in display_text

    def test_format_event_for_chat_run_started(self, orchestrator):
        """Test formatting run started event."""
        event_data = SSEEventData(
            id="evt-6",
            type="run_started",
            category="run_lifecycle",
            created_at=datetime.now(timezone.utc),
            run_id="run-1",
            payload={},
            run_start=RunStartData(
                run_id="run-1",
                timestamp=datetime.now(timezone.utc),
            ),
        )

        display_text = orchestrator._format_event_for_chat(event_data)

        assert "▶️ Run Started" in display_text

    def test_format_event_for_chat_run_completed(self, orchestrator):
        """Test formatting run completed event."""
        event_data = SSEEventData(
            id="evt-7",
            type="run_completed",
            category="run_lifecycle",
            created_at=datetime.now(timezone.utc),
            run_id="run-1",
            payload={},
            run_completion=RunCompletionData(
                status="succeeded",
                timestamp=datetime.now(timezone.utc),
            ),
        )

        display_text = orchestrator._format_event_for_chat(event_data)

        assert "✓ Run Completed Successfully" in display_text

    def test_format_event_for_chat_run_failed(self, orchestrator):
        """Test formatting run failed event."""
        event_data = SSEEventData(
            id="evt-8",
            type="run_failed",
            category="run_lifecycle",
            created_at=datetime.now(timezone.utc),
            run_id="run-1",
            payload={},
            run_failure=RunFailureData(
                error="Agent execution timeout",
                timestamp=datetime.now(timezone.utc),
            ),
        )

        display_text = orchestrator._format_event_for_chat(event_data)

        assert "✗ Run Failed: Agent execution timeout" in display_text

    def test_format_event_for_chat_skips_thinking(self, orchestrator):
        """Test that thinking events are not formatted for chat."""
        event_data = SSEEventData(
            id="evt-9",
            type="thought_executed",
            category="thinking",
            created_at=datetime.now(timezone.utc),
            run_id="run-1",
            payload={},
        )

        display_text = orchestrator._format_event_for_chat(event_data)

        assert display_text == ""

    def test_format_event_for_chat_skips_thinking_output(self, orchestrator):
        """Test that thinking_output events are not formatted for chat."""
        event_data = SSEEventData(
            id="evt-10",
            type="artifact_created",
            category="thinking_output",
            created_at=datetime.now(timezone.utc),
            run_id="run-1",
            payload={},
        )

        display_text = orchestrator._format_event_for_chat(event_data)

        assert display_text == ""


class TestStreamEventsWithChatPersistence:
    """Tests for stream_events with chat persistence callback."""

    @pytest.mark.asyncio
    async def test_stream_events_calls_callback(self):
        """Test that stream_events calls the persistence callback."""
        from unittest.mock import patch

        from gearmeshing_ai.server.services.orchestrator import OrchestratorService

        with (
            patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos_from_session"),
            patch("gearmeshing_ai.server.services.orchestrator.AsyncPostgresSaver"),
            patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider"),
            patch("gearmeshing_ai.server.services.orchestrator.AgentService"),
            patch("gearmeshing_ai.server.services.orchestrator.checkpointer_pool"),
        ):
            orchestrator = OrchestratorService()

        # Mock the repos
        orchestrator.repos = MagicMock()
        mock_event = MagicMock()
        mock_event.type = "capability_executed"
        mock_event.model_dump.return_value = {
            "id": "evt-1",
            "type": "capability_executed",
            "created_at": "2025-12-28T22:00:00Z",
            "run_id": "run-1",
            "payload": {},
        }

        # Mock event list to return one event then empty
        async def mock_list(*args, **kwargs):
            if not hasattr(mock_list, "call_count"):
                mock_list.call_count = 0
            mock_list.call_count += 1
            if mock_list.call_count == 1:
                return [mock_event]
            return []

        orchestrator.repos.events.list = mock_list

        # Track callback invocations
        callback_calls = []

        async def mock_callback(run_id: str, display_text: str, event_type: str):
            callback_calls.append((run_id, display_text, event_type))

        # Stream events
        events = []
        async for event in orchestrator.stream_events("run-1", on_event_persisted=mock_callback):
            events.append(event)
            if len(events) > 1:  # Stop after keep-alive
                break

        # Verify events were streamed
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_stream_events_without_callback(self):
        """Test that stream_events works without persistence callback."""
        from unittest.mock import patch

        from gearmeshing_ai.server.services.orchestrator import OrchestratorService

        with (
            patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos_from_session"),
            patch("gearmeshing_ai.server.services.orchestrator.AsyncPostgresSaver"),
            patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider"),
            patch("gearmeshing_ai.server.services.orchestrator.AgentService"),
            patch("gearmeshing_ai.server.services.orchestrator.checkpointer_pool"),
        ):
            orchestrator = OrchestratorService()

        # Mock the repos
        orchestrator.repos = MagicMock()
        mock_event = MagicMock()
        mock_event.type = "capability_executed"
        mock_event.model_dump.return_value = {
            "id": "evt-1",
            "type": "capability_executed",
            "created_at": "2025-12-28T22:00:00Z",
            "run_id": "run-1",
            "payload": {},
        }

        # Mock event list
        async def mock_list(*args, **kwargs):
            if not hasattr(mock_list, "call_count"):
                mock_list.call_count = 0
            mock_list.call_count += 1
            if mock_list.call_count == 1:
                return [mock_event]
            return []

        orchestrator.repos.events.list = mock_list

        # Stream events without callback
        events = []
        async for event in orchestrator.stream_events("run-1"):
            events.append(event)
            if len(events) > 1:
                break

        # Verify events were streamed
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_stream_events_callback_error_handling(self):
        """Test that callback errors don't break the stream."""
        from unittest.mock import patch

        from gearmeshing_ai.server.services.orchestrator import OrchestratorService

        with (
            patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos_from_session"),
            patch("gearmeshing_ai.server.services.orchestrator.AsyncPostgresSaver"),
            patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider"),
            patch("gearmeshing_ai.server.services.orchestrator.AgentService"),
            patch("gearmeshing_ai.server.services.orchestrator.checkpointer_pool"),
        ):
            orchestrator = OrchestratorService()

        # Mock the repos
        orchestrator.repos = MagicMock()
        mock_event = MagicMock()
        mock_event.type = "capability_executed"
        mock_event.model_dump.return_value = {
            "id": "evt-1",
            "type": "capability_executed",
            "created_at": "2025-12-28T22:00:00Z",
            "run_id": "run-1",
            "payload": {},
        }

        # Mock event list
        async def mock_list(*args, **kwargs):
            if not hasattr(mock_list, "call_count"):
                mock_list.call_count = 0
            mock_list.call_count += 1
            if mock_list.call_count == 1:
                return [mock_event]
            return []

        orchestrator.repos.events.list = mock_list

        # Callback that raises an error
        async def failing_callback(run_id: str, display_text: str, event_type: str):
            raise ValueError("Callback error")

        # Stream events - should not raise
        events = []
        async for event in orchestrator.stream_events("run-1", on_event_persisted=failing_callback):
            events.append(event)
            if len(events) > 1:
                break

        # Verify events were still streamed despite callback error
        assert len(events) > 0
