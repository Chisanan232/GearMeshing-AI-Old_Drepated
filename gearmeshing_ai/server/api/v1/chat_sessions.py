"""
API endpoints for managing chat sessions and messages.

Provides operations for creating chat sessions (chat rooms/zoom),
managing conversation history, and persisting chat data.

Chat sessions represent individual conversation contexts where users and agents
exchange messages. Each session maintains its own message history and metadata.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from gearmeshing_ai.server.core.database import get_session
from gearmeshing_ai.server.models.chat_session import (
    ChatHistoryRead,
    ChatMessage,
    ChatMessageCreate,
    ChatMessageRead,
    ChatSession,
    ChatSessionCreate,
    ChatSessionRead,
    ChatSessionUpdate,
)

router = APIRouter(tags=["chat-sessions"])


@router.post(
    "",
    response_model=ChatSessionRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create Chat Session",
    description="Create a new chat session (conversation room) for a tenant. Each session maintains its own message history.",
    response_description="The created chat session object with auto-generated ID.",
    responses={
        201: {"description": "Chat session created successfully"},
        400: {"description": "Invalid session data"},
    },
)
async def create_chat_session(
    session_data: ChatSessionCreate,
    session: AsyncSession = Depends(get_session),
) -> ChatSessionRead:
    """
    Create a new chat session.

    Initializes a new chat session (conversation room) for a specific tenant.
    Sessions serve as containers for messages exchanged between users and agents.
    Each session has its own metadata and message history.

    - **tenant_id**: The tenant identifier for this session.
    - **title**: A human-readable title for the session.
    - **description**: Optional description of the session purpose.
    - **is_active**: Whether the session is currently active (default: True).
    """
    db_session = ChatSession.model_validate(session_data)
    session.add(db_session)
    await session.commit()
    await session.refresh(db_session)
    return ChatSessionRead.model_validate(db_session)


@router.get(
    "/{session_id}",
    response_model=ChatSessionRead,
    summary="Get Chat Session",
    description="Retrieve a specific chat session by its unique identifier.",
    response_description="The chat session object.",
    responses={
        200: {"description": "Chat session found"},
        404: {"description": "Chat session not found"},
    },
)
async def get_chat_session(
    session_id: int,
    session: AsyncSession = Depends(get_session),
) -> ChatSessionRead:
    """
    Get chat session by ID.

    Retrieves the metadata and details of a specific chat session.
    Use this endpoint to get session information without retrieving all messages.
    To get the full conversation history, use the /history endpoint.

    - **session_id**: The unique identifier of the chat session.
    """
    chat_session = await session.get(ChatSession, session_id)
    if not chat_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session {session_id} not found",
        )
    return ChatSessionRead.model_validate(chat_session)


@router.get(
    "",
    response_model=list[ChatSessionRead],
    summary="List Chat Sessions",
    description="Retrieve a list of chat sessions, optionally filtered by tenant, run_id, and active status.",
    response_description="A list of chat session objects.",
    responses={
        200: {"description": "List of sessions retrieved successfully"},
    },
)
async def list_chat_sessions(
    tenant_id: Optional[str] = None,
    run_id: Optional[str] = None,
    active_only: bool = True,
    session: AsyncSession = Depends(get_session),
) -> list[ChatSessionRead]:
    """
    List chat sessions.

    Retrieves all chat sessions with optional filtering. By default, only active
    sessions are returned. You can filter by tenant_id or run_id to get sessions for a specific
    tenant or agent run, or include inactive sessions.

    - **tenant_id**: Optional filter for a specific tenant. If not provided, returns all sessions.
    - **run_id**: Optional filter for a specific agent run ID. Useful for retrieving chat history for a specific run.
    - **active_only**: If True (default), only returns active sessions. Set to False to include inactive ones.
    """
    statement = select(ChatSession)
    if tenant_id:
        statement = statement.where(ChatSession.tenant_id == tenant_id)
    if run_id:
        statement = statement.where(ChatSession.run_id == run_id)
    if active_only:
        statement = statement.where(ChatSession.is_active == True)
    result = await session.execute(statement)
    sessions = result.scalars().all()
    return [ChatSessionRead.model_validate(s) for s in sessions]


@router.post(
    "/by-run/{run_id}",
    response_model=ChatSessionRead,
    status_code=status.HTTP_200_OK,
    summary="Get or Create Chat Session by Run ID",
    description="Get an existing chat session for a run, or create one if it doesn't exist. Useful for persisting chat history with specific agent runs.",
    response_description="The chat session object (existing or newly created).",
    responses={
        200: {"description": "Chat session retrieved or created successfully"},
        400: {"description": "Invalid session data"},
    },
)
async def get_or_create_chat_session_by_run(
    run_id: str,
    session_data: ChatSessionCreate,
    session: AsyncSession = Depends(get_session),
) -> ChatSessionRead:
    """
    Get or create chat session by run ID.

    Retrieves an existing chat session for a specific agent run, or creates a new one
    if it doesn't exist. This is useful for maintaining persistent chat history across
    multiple interactions with the same agent run.

    - **run_id**: The unique identifier of the agent run.
    - **session_data**: Chat session metadata (title, description, agent_role, tenant_id).
      The run_id will be automatically set to the URL parameter.
    """
    # Check if session already exists for this run
    statement = select(ChatSession).where(ChatSession.run_id == run_id)
    result = await session.execute(statement)
    existing_session = result.scalars().first()
    
    if existing_session:
        return ChatSessionRead.model_validate(existing_session)
    
    # Create new session with run_id
    session_data_dict = session_data.model_dump()
    session_data_dict['run_id'] = run_id
    db_session = ChatSession.model_validate(session_data_dict)
    session.add(db_session)
    await session.commit()
    await session.refresh(db_session)
    return ChatSessionRead.model_validate(db_session)


@router.get(
    "/by-run/{run_id}",
    response_model=ChatSessionRead,
    summary="Get Chat Session by Run ID",
    description="Retrieve a chat session associated with a specific agent run.",
    response_description="The chat session object.",
    responses={
        200: {"description": "Chat session found"},
        404: {"description": "Chat session not found for this run"},
    },
)
async def get_chat_session_by_run(
    run_id: str,
    session: AsyncSession = Depends(get_session),
) -> ChatSessionRead:
    """
    Get chat session by run ID.

    Retrieves the chat session associated with a specific agent run.
    This allows you to fetch the conversation history for a particular run.

    - **run_id**: The unique identifier of the agent run.
    """
    statement = select(ChatSession).where(ChatSession.run_id == run_id)
    result = await session.execute(statement)
    chat_session = result.scalars().first()
    
    if not chat_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session not found for run {run_id}",
        )
    return ChatSessionRead.model_validate(chat_session)


@router.patch(
    "/{session_id}",
    response_model=ChatSessionRead,
    summary="Update Chat Session",
    description="Partially update an existing chat session. Only provided fields are updated.",
    response_description="The updated chat session object.",
    responses={
        200: {"description": "Chat session updated successfully"},
        404: {"description": "Chat session not found"},
    },
)
async def update_chat_session(
    session_id: int,
    session_update: ChatSessionUpdate,
    session: AsyncSession = Depends(get_session),
) -> ChatSessionRead:
    """
    Update chat session.

    Performs a partial update on an existing chat session. Only the fields
    provided in the request body are updated; other fields remain unchanged.
    This is useful for updating session metadata like title, description, or status.

    - **session_id**: The unique identifier of the session to update.
    - **session_update**: The fields to update (all fields are optional).
    """
    chat_session = await session.get(ChatSession, session_id)
    if not chat_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session {session_id} not found",
        )

    update_data = session_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(chat_session, key, value)

    session.add(chat_session)
    await session.commit()
    await session.refresh(chat_session)
    return ChatSessionRead.model_validate(chat_session)


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Chat Session",
    description="Permanently delete a chat session and all its associated messages.",
    responses={
        204: {"description": "Chat session deleted successfully"},
        404: {"description": "Chat session not found"},
    },
)
async def delete_chat_session(
    session_id: int,
    session: AsyncSession = Depends(get_session),
) -> None:
    """
    Delete chat session.

    Permanently removes a chat session and all associated messages from the database.
    This operation cannot be undone. All conversation history for this session will be lost.

    - **session_id**: The unique identifier of the session to delete.
    """
    chat_session = await session.get(ChatSession, session_id)
    if not chat_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session {session_id} not found",
        )

    # Delete all messages in the session
    statement = select(ChatMessage).where(ChatMessage.session_id == session_id)
    result = await session.execute(statement)
    messages = result.scalars().all()
    for message in messages:
        await session.delete(message)

    # Delete the session
    await session.delete(chat_session)
    await session.commit()


# Chat message endpoints
@router.post(
    "/{session_id}/messages",
    response_model=ChatMessageRead,
    status_code=status.HTTP_201_CREATED,
    summary="Add Message to Session",
    description="Add a new message to a chat session. Messages represent exchanges between users and agents.",
    response_description="The created message object with auto-generated ID and timestamp.",
    responses={
        201: {"description": "Message added successfully"},
        400: {"description": "Invalid message data or session mismatch"},
        404: {"description": "Chat session not found"},
    },
)
async def add_message(
    session_id: int,
    message_data: ChatMessageCreate,
    session: AsyncSession = Depends(get_session),
) -> ChatMessageRead:
    """
    Add message to session.

    Adds a new message to a chat session. Messages can be from users or agents
    and are stored with timestamps for conversation history tracking.

    - **session_id**: The session to add the message to.
    - **message_data**: The message content and metadata.
      - **session_id**: Must match the URL session_id.
      - **role**: The sender role ('user', 'agent', 'system').
      - **content**: The message text content.
      - **metadata**: Optional metadata (e.g., tool calls, reasoning).
    """
    # Verify session exists
    chat_session = await session.get(ChatSession, session_id)
    if not chat_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session {session_id} not found",
        )

    # Ensure message belongs to this session
    if message_data.session_id != session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message session_id does not match URL session_id",
        )

    db_message = ChatMessage.model_validate(message_data)
    session.add(db_message)
    await session.commit()
    await session.refresh(db_message)
    return ChatMessageRead.model_validate(db_message)


@router.get(
    "/{session_id}/messages",
    response_model=list[ChatMessageRead],
    summary="Get Session Messages",
    description="Retrieve messages from a chat session with pagination support.",
    response_description="A list of message objects ordered by creation time.",
    responses={
        200: {"description": "Messages retrieved successfully"},
        404: {"description": "Chat session not found"},
    },
)
async def get_messages(
    session_id: int,
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
) -> list[ChatMessageRead]:
    """
    Get session messages.

    Retrieves messages from a specific chat session, ordered by creation time.
    Supports pagination through limit and offset parameters.

    - **session_id**: The session to retrieve messages from.
    - **limit**: Maximum number of messages to return (default: 100).
    - **offset**: Number of messages to skip for pagination (default: 0).
    """
    # Verify session exists
    chat_session = await session.get(ChatSession, session_id)
    if not chat_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session {session_id} not found",
        )

    statement = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())  # type: ignore[attr-defined]
        .offset(offset)
        .limit(limit)
    )
    result = await session.execute(statement)
    messages = result.scalars().all()
    return [ChatMessageRead.model_validate(m) for m in messages]


@router.get(
    "/{session_id}/history",
    response_model=ChatHistoryRead,
    summary="Get Chat History",
    description="Retrieve the complete chat history including session metadata and all messages.",
    response_description="A ChatHistoryRead object containing the session and all its messages.",
    responses={
        200: {"description": "Chat history retrieved successfully"},
        404: {"description": "Chat session not found"},
    },
)
async def get_chat_history(
    session_id: int,
    session: AsyncSession = Depends(get_session),
) -> ChatHistoryRead:
    """
    Get chat history.

    Retrieves the complete conversation history for a session, including the session
    metadata and all messages in chronological order. This is useful for displaying
    the full conversation context to users or for analysis.

    - **session_id**: The session to retrieve history for.
    """
    # Get session
    chat_session = await session.get(ChatSession, session_id)
    if not chat_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session {session_id} not found",
        )

    # Get all messages
    statement = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())  # type: ignore[attr-defined]
    )
    result = await session.execute(statement)
    messages = result.scalars().all()

    return ChatHistoryRead(
        session=ChatSessionRead.model_validate(chat_session),
        messages=[ChatMessageRead.model_validate(m) for m in messages],
    )


@router.delete(
    "/{session_id}/messages/{message_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Message",
    description="Permanently delete a specific message from a chat session.",
    responses={
        204: {"description": "Message deleted successfully"},
        400: {"description": "Message does not belong to this session"},
        404: {"description": "Message not found"},
    },
)
async def delete_message(
    session_id: int,
    message_id: int,
    session: AsyncSession = Depends(get_session),
) -> None:
    """
    Delete message.

    Permanently removes a specific message from a chat session. This operation
    cannot be undone. The message will be removed from the conversation history.

    - **session_id**: The session containing the message.
    - **message_id**: The unique identifier of the message to delete.
    """
    message = await session.get(ChatMessage, message_id)
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Message {message_id} not found",
        )

    if message.session_id != session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message does not belong to this session",
        )

    await session.delete(message)
    await session.commit()
