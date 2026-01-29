"""
Unit tests for Chat Sessions API endpoints.

Tests cover:
- Creating chat sessions
- Retrieving chat session details
- Listing chat sessions with filtering
- Updating chat session metadata
- Deleting chat sessions
- Adding messages to sessions
- Retrieving messages with pagination
- Getting complete chat history
- Deleting specific messages

These tests use direct function calls to ensure proper coverage detection of async code.
See TestDirectFunctionCalls class documentation for why direct calls are necessary.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gearmeshing_ai.core.database.entities.chat_sessions import (
    ChatMessage,
    ChatSession,
)
from gearmeshing_ai.core.models.io.chat_sessions import (
    ChatMessageCreate,
    ChatSessionCreate,
    ChatSessionUpdate,
)

pytestmark = pytest.mark.asyncio


class TestDirectFunctionCalls:
    """Direct function call tests to ensure proper coverage detection of async code.

    IMPORTANT: These tests call the endpoint functions directly with mocked AsyncSession,
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
    - SQLAlchemy async operations may not be properly detected when called through HTTP

    SOLUTION: Direct function calls with mocked AsyncSession allow coverage.py to:
    - Directly instrument the endpoint function code
    - Track all await statements and async operations
    - Properly detect exception handling paths
    - Verify that specific lines like session operations and HTTPException raises execute

    LINES THAT REQUIRE DIRECT CALLS:
    - Lines 62-66: create_chat_session (model_validate, add, commit, refresh)
    - Lines 93-99: get_chat_session (session.get, 404 check)
    - Lines 127-134: list_chat_sessions (query building, filtering, model_validate)
    - Lines 163-177: update_chat_session (get, 404, setattr, commit, refresh)
    - Lines 202-218: delete_chat_session (get, 404, delete messages, delete session, commit)
    - Lines 254-272: add_message (get session, 404, validate, add, commit, refresh)
    - Lines 303-319: get_messages (get session, 404, query, model_validate)
    - Lines 347-363: get_chat_history (get session, 404, get messages, return history)
    - Lines 391-405: delete_message (get message, 404, session check, delete, commit)
    """

    async def test_create_chat_session_direct_call(self, session: AsyncSession):
        """Test create chat session endpoint directly - covers lines 62-66.

        COVERAGE TARGET: Lines 62-66 in chat_sessions.py
            db_session = ChatSession.model_validate(session_data)
            session.add(db_session)
            await session.commit()
            await session.refresh(db_session)
            return ChatSessionRead.model_validate(db_session)

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect await session.commit() execution
        - HTTP layer cannot properly detect await session.refresh() execution
        - Direct function call allows coverage.py to instrument the actual await statements

        VERIFICATION:
        - result is not None: Proves commit and refresh were called
        - result.id is not None: Proves refresh populated the ID
        """
        from gearmeshing_ai.server.api.v1 import chat_sessions

        # Create session data
        session_data = ChatSessionCreate(
            tenant_id="test-tenant",
            title="Test Session",
            description="A test chat session",
            agent_role="planner",
            is_active=True,
        )

        # Call endpoint directly
        result = await chat_sessions.create_chat_session(session_data, session)

        # Verify
        assert result is not None
        assert result.tenant_id == "test-tenant"
        assert result.title == "Test Session"
        assert result.id is not None

    async def test_get_chat_session_success_direct_call(self, session: AsyncSession):
        """Test get chat session endpoint directly - covers line 93.

        COVERAGE TARGET: Line 93 in chat_sessions.py
            chat_session = await session.get(ChatSession, session_id)

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect await session.get() execution
        - Direct function call allows coverage.py to instrument the actual await statement

        VERIFICATION:
        - result is not None: Proves session.get was awaited
        - result.id == session_id: Proves correct session was retrieved
        """
        from gearmeshing_ai.server.api.v1 import chat_sessions

        # Create a session first
        db_session = ChatSession(
            tenant_id="test-tenant", title="Test Session", description="Test", agent_role="planner", is_active=True
        )
        session.add(db_session)
        await session.commit()
        await session.refresh(db_session)
        session_id = db_session.id

        # Call endpoint directly
        result = await chat_sessions.get_chat_session(session_id, session)

        # Verify
        assert result is not None
        assert result.id == session_id
        assert result.title == "Test Session"

    async def test_get_chat_session_not_found_direct_call(self, session: AsyncSession):
        """Test get chat session not found - covers lines 94-98.

        COVERAGE TARGET: Lines 94-98 in chat_sessions.py
            if not chat_session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chat session {session_id} not found",
                )

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect the if condition check
        - HTTP layer cannot properly detect HTTPException raise
        - Direct function call allows coverage.py to instrument the conditional and exception

        VERIFICATION:
        - HTTPException is raised: Proves the condition was checked
        - status_code == 404: Proves the correct error was raised
        """
        from fastapi import HTTPException

        from gearmeshing_ai.server.api.v1 import chat_sessions

        # Call endpoint directly with non-existent session ID
        with pytest.raises(HTTPException) as exc_info:
            await chat_sessions.get_chat_session(99999, session)

        # Verify
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()

    async def test_list_chat_sessions_direct_call(self, session: AsyncSession):
        """Test list chat sessions endpoint directly - covers lines 127-134.

        COVERAGE TARGET: Lines 127-134 in chat_sessions.py
            statement = select(ChatSession)
            if tenant_id:
                statement = statement.where(ChatSession.tenant_id == tenant_id)
            if active_only:
                statement = statement.where(ChatSession.is_active == True)
            result = await session.execute(statement)
            sessions = result.scalars().all()
            return [ChatSessionRead.model_validate(s) for s in sessions]

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect query building and filtering
        - HTTP layer cannot properly detect await session.execute() execution
        - Direct function call allows coverage.py to instrument the actual await statement

        VERIFICATION:
        - result is a list: Proves the query was executed
        - len(result) == 1: Proves filtering worked correctly
        """
        from gearmeshing_ai.server.api.v1 import chat_sessions

        # Create test sessions
        session1 = ChatSession(tenant_id="tenant-1", title="Session 1", agent_role="planner", is_active=True)
        session2 = ChatSession(tenant_id="tenant-2", title="Session 2", agent_role="dev", is_active=False)
        session.add(session1)
        session.add(session2)
        await session.commit()

        # Call endpoint directly with tenant filter
        result = await chat_sessions.list_chat_sessions(tenant_id="tenant-1", active_only=True, session=session)

        # Verify
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].tenant_id == "tenant-1"
        assert result[0].is_active is True

    async def test_update_chat_session_direct_call(self, session: AsyncSession):
        """Test update chat session endpoint directly - covers lines 163-177.

        COVERAGE TARGET: Lines 163-177 in chat_sessions.py
            chat_session = await session.get(ChatSession, session_id)
            if not chat_session:
                raise HTTPException(...)
            update_data = session_update.model_dump(exclude_unset=True)
            for key, value in update_data.items():
                setattr(chat_session, key, value)
            session.add(chat_session)
            await session.commit()
            await session.refresh(chat_session)
            return ChatSessionRead.model_validate(chat_session)

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect setattr() calls
        - HTTP layer cannot properly detect await session.commit() and refresh()
        - Direct function call allows coverage.py to instrument the actual await statements

        VERIFICATION:
        - result is not None: Proves commit and refresh were called
        - result.title == "Updated": Proves setattr worked correctly
        """
        from gearmeshing_ai.server.api.v1 import chat_sessions

        # Create a session first
        db_session = ChatSession(tenant_id="test-tenant", title="Original Title", agent_role="planner", is_active=True)
        session.add(db_session)
        await session.commit()
        await session.refresh(db_session)
        session_id = db_session.id

        # Update the session
        update_data = ChatSessionUpdate(title="Updated Title")
        result = await chat_sessions.update_chat_session(session_id, update_data, session)

        # Verify
        assert result is not None
        assert result.title == "Updated Title"
        assert result.id == session_id

    async def test_delete_chat_session_direct_call(self, session: AsyncSession):
        """Test delete chat session endpoint directly - covers lines 202-218.

        COVERAGE TARGET: Lines 202-218 in chat_sessions.py
            chat_session = await session.get(ChatSession, session_id)
            if not chat_session:
                raise HTTPException(...)
            statement = select(ChatMessage).where(ChatMessage.session_id == session_id)
            result = await session.execute(statement)
            messages = result.scalars().all()
            for message in messages:
                await session.delete(message)
            await session.delete(chat_session)
            await session.commit()

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect multiple await session.delete() calls
        - HTTP layer cannot properly detect the message deletion loop
        - Direct function call allows coverage.py to instrument the actual await statements

        VERIFICATION:
        - No exception raised: Proves deletion succeeded
        - Session no longer exists: Proves delete was committed
        """
        from gearmeshing_ai.server.api.v1 import chat_sessions

        # Create a session with a message
        db_session = ChatSession(tenant_id="test-tenant", title="Test Session", agent_role="planner", is_active=True)
        session.add(db_session)
        await session.commit()
        await session.refresh(db_session)
        session_id = db_session.id

        # Add a message
        message = ChatMessage(session_id=session_id, role="user", content="Test message")
        session.add(message)
        await session.commit()

        # Delete the session
        await chat_sessions.delete_chat_session(session_id, session)

        # Verify deletion
        deleted_session = await session.get(ChatSession, session_id)
        assert deleted_session is None

    async def test_add_message_direct_call(self, session: AsyncSession):
        """Test add message endpoint directly - covers lines 254-272.

        COVERAGE TARGET: Lines 254-272 in chat_sessions.py
            chat_session = await session.get(ChatSession, session_id)
            if not chat_session:
                raise HTTPException(...)
            if message_data.session_id != session_id:
                raise HTTPException(...)
            db_message = ChatMessage.model_validate(message_data)
            session.add(db_message)
            await session.commit()
            await session.refresh(db_message)
            return ChatMessageRead.model_validate(db_message)

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect await session.commit() and refresh()
        - HTTP layer cannot properly detect validation checks
        - Direct function call allows coverage.py to instrument the actual await statements

        VERIFICATION:
        - result is not None: Proves commit and refresh were called
        - result.id is not None: Proves refresh populated the ID
        """
        from gearmeshing_ai.server.api.v1 import chat_sessions

        # Create a session first
        db_session = ChatSession(tenant_id="test-tenant", title="Test Session", agent_role="planner", is_active=True)
        session.add(db_session)
        await session.commit()
        await session.refresh(db_session)
        session_id = db_session.id

        # Add a message
        message_data = ChatMessageCreate(session_id=session_id, role="user", content="Test message")
        result = await chat_sessions.add_message(session_id, message_data, session)

        # Verify
        assert result is not None
        assert result.session_id == session_id
        assert result.role == "user"
        assert result.content == "Test message"
        assert result.id is not None

    async def test_get_messages_direct_call(self, session: AsyncSession):
        """Test get messages endpoint directly - covers lines 303-319.

        COVERAGE TARGET: Lines 303-319 in chat_sessions.py
            chat_session = await session.get(ChatSession, session_id)
            if not chat_session:
                raise HTTPException(...)
            statement = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
                .offset(offset)
                .limit(limit)
            )
            result = await session.execute(statement)
            messages = result.scalars().all()
            return [ChatMessageRead.model_validate(m) for m in messages]

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect query building with pagination
        - HTTP layer cannot properly detect await session.execute() execution
        - Direct function call allows coverage.py to instrument the actual await statement

        VERIFICATION:
        - result is a list: Proves the query was executed
        - len(result) == 2: Proves messages were retrieved
        """
        from gearmeshing_ai.server.api.v1 import chat_sessions

        # Create a session with messages
        db_session = ChatSession(tenant_id="test-tenant", title="Test Session", agent_role="planner", is_active=True)
        session.add(db_session)
        await session.commit()
        await session.refresh(db_session)
        session_id = db_session.id

        # Add messages
        for i in range(3):
            message = ChatMessage(
                session_id=session_id, role="user" if i % 2 == 0 else "assistant", content=f"Message {i}"
            )
            session.add(message)
        await session.commit()

        # Get messages with limit
        result = await chat_sessions.get_messages(session_id, limit=2, offset=0, session=session)

        # Verify
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(m.session_id == session_id for m in result)

    async def test_get_chat_history_direct_call(self, session: AsyncSession):
        """Test get chat history endpoint directly - covers lines 347-363.

        COVERAGE TARGET: Lines 347-363 in chat_sessions.py
            chat_session = await session.get(ChatSession, session_id)
            if not chat_session:
                raise HTTPException(...)
            statement = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
            )
            result = await session.execute(statement)
            messages = result.scalars().all()
            return ChatHistoryRead(...)

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect query building and execution
        - HTTP layer cannot properly detect await session.execute() execution
        - Direct function call allows coverage.py to instrument the actual await statement

        VERIFICATION:
        - result is not None: Proves the history was retrieved
        - result.session is not None: Proves session was included
        - len(result.messages) == 2: Proves messages were included
        """
        from gearmeshing_ai.server.api.v1 import chat_sessions

        # Create a session with messages
        db_session = ChatSession(tenant_id="test-tenant", title="Test Session", agent_role="planner", is_active=True)
        session.add(db_session)
        await session.commit()
        await session.refresh(db_session)
        session_id = db_session.id

        # Add messages
        for i in range(2):
            message = ChatMessage(
                session_id=session_id, role="user" if i % 2 == 0 else "assistant", content=f"Message {i}"
            )
            session.add(message)
        await session.commit()

        # Get chat history
        result = await chat_sessions.get_chat_history(session_id, session)

        # Verify
        assert result is not None
        assert result.session is not None
        assert result.session.id == session_id
        assert len(result.messages) == 2

    async def test_delete_message_direct_call(self, session: AsyncSession):
        """Test delete message endpoint directly - covers lines 391-405.

        COVERAGE TARGET: Lines 391-405 in chat_sessions.py
            message = await session.get(ChatMessage, message_id)
            if not message:
                raise HTTPException(...)
            if message.session_id != session_id:
                raise HTTPException(...)
            await session.delete(message)
            await session.commit()

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect await session.delete() and commit()
        - HTTP layer cannot properly detect session_id validation
        - Direct function call allows coverage.py to instrument the actual await statements

        VERIFICATION:
        - No exception raised: Proves deletion succeeded
        - Message no longer exists: Proves delete was committed
        """
        from gearmeshing_ai.server.api.v1 import chat_sessions

        # Create a session with a message
        db_session = ChatSession(tenant_id="test-tenant", title="Test Session", agent_role="planner", is_active=True)
        session.add(db_session)
        await session.commit()
        await session.refresh(db_session)
        session_id = db_session.id

        # Add a message
        message = ChatMessage(session_id=session_id, role="user", content="Test message")
        session.add(message)
        await session.commit()
        await session.refresh(message)
        message_id = message.id

        # Delete the message
        await chat_sessions.delete_message(session_id, message_id, session)

        # Verify deletion
        deleted_message = await session.get(ChatMessage, message_id)
        assert deleted_message is None

    async def test_add_message_session_mismatch_direct_call(self, session: AsyncSession):
        """Test add message with session mismatch - covers lines 262-266.

        COVERAGE TARGET: Lines 262-266 in chat_sessions.py
            if message_data.session_id != session_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Message session_id does not match URL session_id",
                )

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect the session_id validation check
        - Direct function call allows coverage.py to instrument the conditional and exception

        VERIFICATION:
        - HTTPException is raised: Proves the condition was checked
        - status_code == 400: Proves the correct error was raised
        """
        from fastapi import HTTPException

        from gearmeshing_ai.server.api.v1 import chat_sessions

        # Create a session first
        db_session = ChatSession(tenant_id="test-tenant", title="Test Session", agent_role="planner", is_active=True)
        session.add(db_session)
        await session.commit()
        await session.refresh(db_session)
        session_id = db_session.id

        # Try to add a message with mismatched session_id
        message_data = ChatMessageCreate(session_id=999, role="user", content="Test message")  # Wrong session ID

        with pytest.raises(HTTPException) as exc_info:
            await chat_sessions.add_message(session_id, message_data, session)

        # Verify
        assert exc_info.value.status_code == 400
        assert "does not match" in exc_info.value.detail.lower()

    async def test_delete_message_wrong_session_direct_call(self, session: AsyncSession):
        """Test delete message from wrong session - covers lines 398-402.

        COVERAGE TARGET: Lines 398-402 in chat_sessions.py
            if message.session_id != session_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Message does not belong to this session",
                )

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect the session_id validation check
        - Direct function call allows coverage.py to instrument the conditional and exception

        VERIFICATION:
        - HTTPException is raised: Proves the condition was checked
        - status_code == 400: Proves the correct error was raised
        """
        from fastapi import HTTPException

        from gearmeshing_ai.server.api.v1 import chat_sessions

        # Create two sessions
        session1 = ChatSession(tenant_id="test-tenant", title="Session 1", agent_role="planner", is_active=True)
        session2 = ChatSession(tenant_id="test-tenant", title="Session 2", agent_role="dev", is_active=True)
        session.add(session1)
        session.add(session2)
        await session.commit()
        await session.refresh(session1)
        await session.refresh(session2)

        # Add a message to session1
        message = ChatMessage(session_id=session1.id, role="user", content="Test message")
        session.add(message)
        await session.commit()
        await session.refresh(message)

        # Try to delete the message from session2
        with pytest.raises(HTTPException) as exc_info:
            await chat_sessions.delete_message(session2.id, message.id, session)

        # Verify
        assert exc_info.value.status_code == 400
        assert "does not belong" in exc_info.value.detail.lower()

    async def test_get_chat_session_not_found_exception_direct_call(self, session: AsyncSession):
        """Test get chat session not found exception - covers lines 165-169.

        COVERAGE TARGET: Lines 165-169 in chat_sessions.py
            if not chat_session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chat session {session_id} not found",
                )

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect the if condition and HTTPException raise
        - Direct function call allows coverage.py to instrument the conditional and exception

        VERIFICATION:
        - HTTPException is raised: Proves the condition was checked
        - status_code == 404: Proves the correct error was raised
        """
        from fastapi import HTTPException

        from gearmeshing_ai.server.api.v1 import chat_sessions

        # Call endpoint directly with non-existent session ID
        with pytest.raises(HTTPException) as exc_info:
            await chat_sessions.get_chat_session(99999, session)

        # Verify
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()

    async def test_delete_chat_session_not_found_exception_direct_call(self, session: AsyncSession):
        """Test delete chat session not found exception - covers lines 204-208.

        COVERAGE TARGET: Lines 204-208 in chat_sessions.py
            if not chat_session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chat session {session_id} not found",
                )

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect the if condition and HTTPException raise
        - Direct function call allows coverage.py to instrument the conditional and exception

        VERIFICATION:
        - HTTPException is raised: Proves the condition was checked
        - status_code == 404: Proves the correct error was raised
        """
        from fastapi import HTTPException

        from gearmeshing_ai.server.api.v1 import chat_sessions

        # Call endpoint directly with non-existent session ID
        with pytest.raises(HTTPException) as exc_info:
            await chat_sessions.delete_chat_session(99999, session)

        # Verify
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()

    async def test_update_chat_session_not_found_exception_direct_call(self, session: AsyncSession):
        """Test update chat session not found exception - covers lines 256-260.

        COVERAGE TARGET: Lines 256-260 in chat_sessions.py
            if not chat_session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chat session {session_id} not found",
                )

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect the if condition and HTTPException raise
        - Direct function call allows coverage.py to instrument the conditional and exception

        VERIFICATION:
        - HTTPException is raised: Proves the condition was checked
        - status_code == 404: Proves the correct error was raised
        """
        from fastapi import HTTPException

        from gearmeshing_ai.server.api.v1 import chat_sessions

        # Call endpoint directly with non-existent session ID
        update_data = ChatSessionUpdate(title="Updated")
        with pytest.raises(HTTPException) as exc_info:
            await chat_sessions.update_chat_session(99999, update_data, session)

        # Verify
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()

    async def test_get_messages_session_not_found_exception_direct_call(self, session: AsyncSession):
        """Test get messages session not found exception - covers lines 305-309.

        COVERAGE TARGET: Lines 305-309 in chat_sessions.py
            if not chat_session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chat session {session_id} not found",
                )

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect the if condition and HTTPException raise
        - Direct function call allows coverage.py to instrument the conditional and exception

        VERIFICATION:
        - HTTPException is raised: Proves the condition was checked
        - status_code == 404: Proves the correct error was raised
        """
        from fastapi import HTTPException

        from gearmeshing_ai.server.api.v1 import chat_sessions

        # Call endpoint directly with non-existent session ID
        with pytest.raises(HTTPException) as exc_info:
            await chat_sessions.get_messages(99999, limit=100, offset=0, session=session)

        # Verify
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()

    async def test_get_chat_history_session_not_found_exception_direct_call(self, session: AsyncSession):
        """Test get chat history session not found exception - covers lines 349-353.

        COVERAGE TARGET: Lines 349-353 in chat_sessions.py
            if not chat_session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chat session {session_id} not found",
                )

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect the if condition and HTTPException raise
        - Direct function call allows coverage.py to instrument the conditional and exception

        VERIFICATION:
        - HTTPException is raised: Proves the condition was checked
        - status_code == 404: Proves the correct error was raised
        """
        from fastapi import HTTPException

        from gearmeshing_ai.server.api.v1 import chat_sessions

        # Call endpoint directly with non-existent session ID
        with pytest.raises(HTTPException) as exc_info:
            await chat_sessions.get_chat_history(99999, session)

        # Verify
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()

    async def test_delete_message_not_found_exception_direct_call(self, session: AsyncSession):
        """Test delete message not found exception - covers lines 393-397.

        COVERAGE TARGET: Lines 393-397 in chat_sessions.py
            if not message:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Message {message_id} not found",
                )

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect the if condition and HTTPException raise
        - Direct function call allows coverage.py to instrument the conditional and exception

        VERIFICATION:
        - HTTPException is raised: Proves the condition was checked
        - status_code == 404: Proves the correct error was raised
        """
        from fastapi import HTTPException

        from gearmeshing_ai.server.api.v1 import chat_sessions

        # Create a session first
        db_session = ChatSession(tenant_id="test-tenant", title="Test Session", agent_role="planner", is_active=True)
        session.add(db_session)
        await session.commit()
        await session.refresh(db_session)
        session_id = db_session.id

        # Call endpoint directly with non-existent message ID
        with pytest.raises(HTTPException) as exc_info:
            await chat_sessions.delete_message(session_id, 99999, session)

        # Verify
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()

    async def test_update_chat_session_with_multiple_fields_direct_call(self, session: AsyncSession):
        """Test update chat session with multiple fields - covers lines 170-177.

        COVERAGE TARGET: Lines 170-177 in chat_sessions.py
            update_data = session_update.model_dump(exclude_unset=True)
            for key, value in update_data.items():
                setattr(chat_session, key, value)
            session.add(chat_session)
            await session.commit()
            await session.refresh(chat_session)
            return ChatSessionRead.model_validate(chat_session)

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect the setattr loop and multiple field updates
        - HTTP layer cannot properly detect await session.commit() and refresh()
        - Direct function call allows coverage.py to instrument the actual await statements

        VERIFICATION:
        - result.title == "New Title": Proves setattr was called for title
        - result.description == "New Desc": Proves setattr was called for description
        - result.is_active == False: Proves setattr was called for is_active
        """
        from gearmeshing_ai.server.api.v1 import chat_sessions

        # Create a session first
        db_session = ChatSession(
            tenant_id="test-tenant",
            title="Original Title",
            description="Original Desc",
            agent_role="planner",
            is_active=True,
        )
        session.add(db_session)
        await session.commit()
        await session.refresh(db_session)
        session_id = db_session.id

        # Update multiple fields
        update_data = ChatSessionUpdate(title="New Title", description="New Desc", is_active=False)
        result = await chat_sessions.update_chat_session(session_id, update_data, session)

        # Verify all fields were updated
        assert result is not None
        assert result.title == "New Title"
        assert result.description == "New Desc"
        assert result.is_active is False
        assert result.id == session_id

    async def test_update_chat_session_single_field_direct_call(self, session: AsyncSession):
        """Test update chat session with single field - covers lines 170-177.

        COVERAGE TARGET: Lines 170-177 in chat_sessions.py (with single field update)
            update_data = session_update.model_dump(exclude_unset=True)
            for key, value in update_data.items():
                setattr(chat_session, key, value)
            session.add(chat_session)
            await session.commit()
            await session.refresh(chat_session)

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect the setattr loop with single field
        - Direct function call allows coverage.py to track single field update path

        VERIFICATION:
        - result.agent_role == "dev": Proves setattr was called for single field
        - Other fields unchanged: Proves only specified field was updated
        """
        from gearmeshing_ai.server.api.v1 import chat_sessions

        # Create a session first
        db_session = ChatSession(
            tenant_id="test-tenant",
            title="Original Title",
            description="Original Desc",
            agent_role="planner",
            is_active=True,
        )
        session.add(db_session)
        await session.commit()
        await session.refresh(db_session)
        session_id = db_session.id

        # Update single field
        update_data = ChatSessionUpdate(agent_role="dev")
        result = await chat_sessions.update_chat_session(session_id, update_data, session)

        # Verify only the specified field was updated
        assert result is not None
        assert result.agent_role == "dev"
        assert result.title == "Original Title"  # Unchanged
        assert result.description == "Original Desc"  # Unchanged
        assert result.is_active is True  # Unchanged

    async def test_add_message_session_not_found_exception_direct_call(self, session: AsyncSession):
        """Test add message session not found exception - covers lines 256-260.

        COVERAGE TARGET: Lines 256-260 in chat_sessions.py
            if not chat_session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chat session {session_id} not found",
                )

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect the if condition and HTTPException raise
        - Direct function call allows coverage.py to instrument the conditional and exception

        VERIFICATION:
        - HTTPException is raised: Proves the condition was checked
        - status_code == 404: Proves the correct error was raised
        """
        from fastapi import HTTPException

        from gearmeshing_ai.server.api.v1 import chat_sessions

        # Try to add a message to non-existent session
        message_data = ChatMessageCreate(session_id=99999, role="user", content="Test message")

        # Call endpoint directly with non-existent session ID
        with pytest.raises(HTTPException) as exc_info:
            await chat_sessions.add_message(99999, message_data, session)

        # Verify
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()
