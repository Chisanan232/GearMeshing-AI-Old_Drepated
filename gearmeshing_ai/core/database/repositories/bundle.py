"""
Repository bundle for dependency injection.

This module provides a convenience bundle of all repository instances
for easy dependency injection in services and application components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .agent_configs import AgentConfigRepository
from .agent_events import AgentEventRepository
from .agent_runs import AgentRunRepository
from .approvals import ApprovalRepository
from .chat_sessions import ChatSessionRepository
from .checkpoints import CheckpointRepository
from .policies import PolicyRepository
from .tool_invocations import ToolInvocationRepository
from .usage_ledger import UsageLedgerRepository


@dataclass(frozen=True)
class SqlRepoBundle:
    """Convenience bundle of all SQL repositories for dependency injection."""

    runs: AgentRunRepository
    events: AgentEventRepository
    approvals: ApprovalRepository
    checkpoints: CheckpointRepository
    tool_invocations: ToolInvocationRepository
    usage: UsageLedgerRepository
    policies: PolicyRepository
    chat_sessions: ChatSessionRepository
    agent_configs: Optional[AgentConfigRepository] = None


async def build_sql_repos(*, session_factory: async_sessionmaker[AsyncSession]) -> SqlRepoBundle:
    """Build a SqlRepoBundle from a session factory.

    Args:
        session_factory: Async session factory for creating sessions

    Returns:
        Bundle containing all repository instances
    """
    # Create a session to initialize repositories
    async with session_factory() as session:
        return SqlRepoBundle(
            runs=AgentRunRepository(session),
            events=AgentEventRepository(session),
            approvals=ApprovalRepository(session),
            checkpoints=CheckpointRepository(session),
            tool_invocations=ToolInvocationRepository(session),
            usage=UsageLedgerRepository(session),
            policies=PolicyRepository(session),
            chat_sessions=ChatSessionRepository(session),
            agent_configs=(
                AgentConfigRepository(session)
                if hasattr(session, "bind") and "agent_configs" in str(session.bind.url)
                else None
            ),
        )


def build_sql_repos_from_session(*, session: AsyncSession) -> SqlRepoBundle:
    """Build a SqlRepoBundle from an existing session.

    Args:
        session: Existing async session

    Returns:
        Bundle containing all repository instances
    """
    return SqlRepoBundle(
        runs=AgentRunRepository(session),
        events=AgentEventRepository(session),
        approvals=ApprovalRepository(session),
        checkpoints=CheckpointRepository(session),
        tool_invocations=ToolInvocationRepository(session),
        usage=UsageLedgerRepository(session),
        policies=PolicyRepository(session),
        chat_sessions=ChatSessionRepository(session),
        agent_configs=(
            AgentConfigRepository(session)
            if hasattr(session, "bind") and "agent_configs" in str(session.bind.url)
            else None
        ),
    )
