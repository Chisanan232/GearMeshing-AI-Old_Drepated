"""
I/O models for API requests and responses.

This package contains Pydantic-based I/O schemas that define the contract
between API endpoints and clients. These models are separate from database
entities to allow independent evolution of API contracts.

Modules:
- agent_configs: Agent configuration I/O models
- chat_sessions: Chat session and message I/O models
"""

from .agent_configs import (
    AgentConfigCreate,
    AgentConfigRead,
    AgentConfigUpdate,
)
from .chat_sessions import (
    ChatHistoryRead,
    ChatMessageCreate,
    ChatMessageRead,
    ChatSessionCreate,
    ChatSessionRead,
    ChatSessionUpdate,
    MessageRole,
)

__all__ = [
    "AgentConfigCreate",
    "AgentConfigRead",
    "AgentConfigUpdate",
    "ChatHistoryRead",
    "ChatMessageCreate",
    "ChatMessageRead",
    "ChatSessionCreate",
    "ChatSessionRead",
    "ChatSessionUpdate",
    "MessageRole",
]
