"""
Complete async repository bundle matching the old agent_core.repos design.

This module provides all async repository implementations that match the interfaces
and functionality of the original agent_core.repos.sql module, but using the
new centralized database architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import select

from .async_base import AsyncBaseRepository, AsyncQueryBuilder, _utc_now_naive
from .entities.agent_events import AgentEvent
from .entities.agent_runs import AgentRun
from .entities.approvals import Approval
from .entities.checkpoints import Checkpoint
from .entities.policies import Policy
from .entities.tool_invocations import ToolInvocation
from .entities.usage_ledger import UsageLedger
from .async_repositories import AsyncAgentRunRepository


class AsyncAgentEventRepository(AsyncBaseRepository[AgentEvent]):
    """Async repository for agent event data access operations."""
    
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, AgentEvent)
    
    async def create(self, event: AgentEvent) -> AgentEvent:
        self.session.add(event)
        await self.session.commit()
        await self.session.refresh(event)
        return event
    
    async def get_by_id(self, event_id: str | int) -> Optional[AgentEvent]:
        event_id_str = str(event_id)
        stmt = select(AgentEvent).where(AgentEvent.id == event_id_str)
        result = await self.session.exec(stmt)
        return result.one_or_none()
    
    async def update(self, event: AgentEvent) -> AgentEvent:
        self.session.add(event)
        await self.session.commit()
        await self.session.refresh(event)
        return event
    
    async def delete(self, event_id: str | int) -> bool:
        event = await self.get_by_id(event_id)
        if event:
            await self.session.delete(event)
            await self.session.commit()
            return True
        return False
    
    async def list(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[dict] = None
    ) -> list[AgentEvent]:
        stmt = select(AgentEvent).order_by(AgentEvent.created_at.asc())
        
        if filters:
            stmt = AsyncQueryBuilder.apply_filters(stmt, AgentEvent, filters)
        
        stmt = AsyncQueryBuilder.apply_pagination(stmt, limit, offset)
        
        result = await self.session.exec(stmt)
        return list(result)
    
    # Methods to match old interface
    async def append(self, event: AgentEvent) -> None:
        """Append a new event to the store."""
        await self.create(event)
    
    async def list_by_run(self, run_id: str, limit: int = 100) -> list[AgentEvent]:
        """List events for a specific run."""
        filters = {"run_id": run_id}
        return await self.list(limit=limit, filters=filters)


class AsyncApprovalRepository(AsyncBaseRepository[Approval]):
    """Async repository for approval data access operations."""
    
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Approval)
    
    async def create(self, approval: Approval) -> Approval:
        self.session.add(approval)
        await self.session.commit()
        await self.session.refresh(approval)
        return approval
    
    async def get_by_id(self, approval_id: str | int) -> Optional[Approval]:
        approval_id_str = str(approval_id)
        stmt = select(Approval).where(Approval.id == approval_id_str)
        result = await self.session.exec(stmt)
        return result.one_or_none()
    
    async def update(self, approval: Approval) -> Approval:
        self.session.add(approval)
        await self.session.commit()
        await self.session.refresh(approval)
        return approval
    
    async def delete(self, approval_id: str | int) -> bool:
        approval = await self.get_by_id(approval_id)
        if approval:
            await self.session.delete(approval)
            await self.session.commit()
            return True
        return False
    
    async def list(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[dict] = None
    ) -> list[Approval]:
        stmt = select(Approval).order_by(Approval.requested_at.asc())
        
        if filters:
            stmt = AsyncQueryBuilder.apply_filters(stmt, Approval, filters)
        
        stmt = AsyncQueryBuilder.apply_pagination(stmt, limit, offset)
        
        result = await self.session.exec(stmt)
        return list(result)
    
    # Methods to match old interface
    async def get(self, approval_id: str) -> Optional[Approval]:
        """Retrieve an approval by ID."""
        return await self.get_by_id(approval_id)
    
    async def resolve(self, approval_id: str, *, decision: str, decided_by: str | None) -> None:
        """Update an approval with a decision."""
        approval = await self.get_by_id(approval_id)
        if approval:
            approval.decision = decision
            approval.decided_by = decided_by
            approval.decided_at = _utc_now_naive()
            await self.session.commit()
    
    async def list_by_run(self, run_id: str, pending_only: bool = True) -> list[Approval]:
        """List approvals for a run."""
        filters = {"run_id": run_id}
        if pending_only:
            filters["decision"] = None  # Only pending approvals
        return await self.list(filters=filters)


class AsyncCheckpointRepository(AsyncBaseRepository[Checkpoint]):
    """Async repository for checkpoint data access operations."""
    
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Checkpoint)
    
    async def create(self, checkpoint: Checkpoint) -> Checkpoint:
        self.session.add(checkpoint)
        await self.session.commit()
        await self.session.refresh(checkpoint)
        return checkpoint
    
    async def get_by_id(self, checkpoint_id: str | int) -> Optional[Checkpoint]:
        checkpoint_id_str = str(checkpoint_id)
        stmt = select(Checkpoint).where(Checkpoint.id == checkpoint_id_str)
        result = await self.session.exec(stmt)
        return result.one_or_none()
    
    async def update(self, checkpoint: Checkpoint) -> Checkpoint:
        self.session.add(checkpoint)
        await self.session.commit()
        await self.session.refresh(checkpoint)
        return checkpoint
    
    async def delete(self, checkpoint_id: str | int) -> bool:
        checkpoint = await self.get_by_id(checkpoint_id)
        if checkpoint:
            await self.session.delete(checkpoint)
            await self.session.commit()
            return True
        return False
    
    async def list(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[dict] = None
    ) -> list[Checkpoint]:
        stmt = select(Checkpoint).order_by(Checkpoint.created_at.desc())
        
        if filters:
            stmt = AsyncQueryBuilder.apply_filters(stmt, Checkpoint, filters)
        
        stmt = AsyncQueryBuilder.apply_pagination(stmt, limit, offset)
        
        result = await self.session.exec(stmt)
        return list(result)
    
    # Methods to match old interface
    async def save(self, checkpoint: Checkpoint) -> None:
        """Persist a checkpoint state."""
        await self.create(checkpoint)
    
    async def latest(self, run_id: str) -> Optional[Checkpoint]:
        """Fetch the most recent checkpoint for a run."""
        stmt = (
            select(Checkpoint)
            .where(Checkpoint.run_id == run_id)
            .order_by(Checkpoint.created_at.desc())
            .limit(1)
        )
        result = await self.session.exec(stmt)
        return result.one_or_none()


class AsyncToolInvocationRepository(AsyncBaseRepository[ToolInvocation]):
    """Async repository for tool invocation data access operations."""
    
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, ToolInvocation)
    
    async def create(self, invocation: ToolInvocation) -> ToolInvocation:
        self.session.add(invocation)
        await self.session.commit()
        await self.session.refresh(invocation)
        return invocation
    
    async def get_by_id(self, invocation_id: str | int) -> Optional[ToolInvocation]:
        invocation_id_str = str(invocation_id)
        stmt = select(ToolInvocation).where(ToolInvocation.id == invocation_id_str)
        result = await self.session.exec(stmt)
        return result.one_or_none()
    
    async def update(self, invocation: ToolInvocation) -> ToolInvocation:
        self.session.add(invocation)
        await self.session.commit()
        await self.session.refresh(invocation)
        return invocation
    
    async def delete(self, invocation_id: str | int) -> bool:
        invocation = await self.get_by_id(invocation_id)
        if invocation:
            await self.session.delete(invocation)
            await self.session.commit()
            return True
        return False
    
    async def list(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[dict] = None
    ) -> list[ToolInvocation]:
        stmt = select(ToolInvocation).order_by(ToolInvocation.created_at.desc())
        
        if filters:
            stmt = AsyncQueryBuilder.apply_filters(stmt, ToolInvocation, filters)
        
        stmt = AsyncQueryBuilder.apply_pagination(stmt, limit, offset)
        
        result = await self.session.exec(stmt)
        return list(result)
    
    # Methods to match old interface
    async def append(self, invocation: ToolInvocation) -> None:
        """Log a tool invocation."""
        await self.create(invocation)


class AsyncUsageRepository(AsyncBaseRepository[UsageLedger]):
    """Async repository for usage ledger data access operations."""
    
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, UsageLedger)
    
    async def create(self, usage: UsageLedger) -> UsageLedger:
        self.session.add(usage)
        await self.session.commit()
        await self.session.refresh(usage)
        return usage
    
    async def get_by_id(self, usage_id: str | int) -> Optional[UsageLedger]:
        usage_id_str = str(usage_id)
        stmt = select(UsageLedger).where(UsageLedger.id == usage_id_str)
        result = await self.session.exec(stmt)
        return result.one_or_none()
    
    async def update(self, usage: UsageLedger) -> UsageLedger:
        self.session.add(usage)
        await self.session.commit()
        await self.session.refresh(usage)
        return usage
    
    async def delete(self, usage_id: str | int) -> bool:
        usage = await self.get_by_id(usage_id)
        if usage:
            await self.session.delete(usage)
            await self.session.commit()
            return True
        return False
    
    async def list(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[dict] = None
    ) -> list[UsageLedger]:
        stmt = select(UsageLedger).order_by(UsageLedger.created_at.desc())
        
        if filters:
            stmt = AsyncQueryBuilder.apply_filters(stmt, UsageLedger, filters)
        
        stmt = AsyncQueryBuilder.apply_pagination(stmt, limit, offset)
        
        result = await self.session.exec(stmt)
        return list(result)
    
    # Methods to match old interface
    async def append(self, usage: UsageLedger) -> None:
        """Record a usage entry."""
        await self.create(usage)
    
    async def list_by_tenant(
        self, 
        tenant_id: str, 
        from_date: Optional[datetime] = None, 
        to_date: Optional[datetime] = None
    ) -> list[UsageLedger]:
        """List usage entries for a tenant within a date range."""
        # Need to join with runs to filter by tenant_id since UsageLedger doesn't have tenant_id
        stmt = (
            select(UsageLedger)
            .join(AgentRun, UsageLedger.run_id == AgentRun.id)
            .where(AgentRun.tenant_id == tenant_id)
        )
        
        if from_date:
            stmt = stmt.where(UsageLedger.created_at >= from_date)
        if to_date:
            stmt = stmt.where(UsageLedger.created_at <= to_date)
        
        stmt = stmt.order_by(UsageLedger.created_at.desc())
        
        result = await self.session.exec(stmt)
        return list(result)


class AsyncPolicyRepository(AsyncBaseRepository[Policy]):
    """Async repository for policy data access operations."""
    
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Policy)
    
    async def create(self, policy: Policy) -> Policy:
        self.session.add(policy)
        await self.session.commit()
        await self.session.refresh(policy)
        return policy
    
    async def get_by_id(self, policy_id: str | int) -> Optional[Policy]:
        policy_id_str = str(policy_id)
        stmt = select(Policy).where(Policy.id == policy_id_str)
        result = await self.session.exec(stmt)
        return result.one_or_none()
    
    async def update(self, policy: Policy) -> Policy:
        policy.updated_at = _utc_now_naive()
        self.session.add(policy)
        await self.session.commit()
        await self.session.refresh(policy)
        return policy
    
    async def delete(self, policy_id: str | int) -> bool:
        policy = await self.get_by_id(policy_id)
        if policy:
            await self.session.delete(policy)
            await self.session.commit()
            return True
        return False
    
    async def list(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[dict] = None
    ) -> list[Policy]:
        stmt = select(Policy).order_by(Policy.tenant_id)
        
        if filters:
            stmt = AsyncQueryBuilder.apply_filters(stmt, Policy, filters)
        
        stmt = AsyncQueryBuilder.apply_pagination(stmt, limit, offset)
        
        result = await self.session.exec(stmt)
        return list(result)
    
    # Methods to match old interface
    async def get_by_tenant(self, tenant_id: str) -> Optional[Policy]:
        """Get policy configuration for a specific tenant."""
        stmt = select(Policy).where(Policy.tenant_id == tenant_id)
        result = await self.session.exec(stmt)
        return result.one_or_none()


@dataclass(frozen=True)
class AsyncSqlRepoBundle:
    """Convenience bundle of all async SQL repositories for dependency injection."""

    runs: AsyncAgentRunRepository
    events: AsyncAgentEventRepository
    approvals: AsyncApprovalRepository
    checkpoints: AsyncCheckpointRepository
    tool_invocations: AsyncToolInvocationRepository
    usage: AsyncUsageRepository
    policies: AsyncPolicyRepository


async def build_async_sql_repos(*, session_factory: async_sessionmaker[AsyncSession]) -> AsyncSqlRepoBundle:
    """Build an ``AsyncSqlRepoBundle`` from a session factory.
    
    Args:
        session_factory: Async session factory for creating sessions
        
    Returns:
        Bundle containing all async repository instances
    """
    # Create a session to initialize repositories
    async with session_factory() as session:
        return AsyncSqlRepoBundle(
            runs=AsyncAgentRunRepository(session),
            events=AsyncAgentEventRepository(session),
            approvals=AsyncApprovalRepository(session),
            checkpoints=AsyncCheckpointRepository(session),
            tool_invocations=AsyncToolInvocationRepository(session),
            usage=AsyncUsageRepository(session),
            policies=AsyncPolicyRepository(session),
        )
