from __future__ import annotations

"""SQLAlchemy ORM models for agent persistence.

These ORM models define the SQL schema used by the SQL repository
implementation in ``gearmeshing_ai.agent_core.repos.sql``.

Design
------

The schema is optimized for auditability and pause/resume:

- Runs store coarse-grained run metadata and status.
- Events form an append-only timeline.
- Approvals and checkpoints allow pausing and resuming a run.
- Tool invocations provide an auditable record of side-effecting operations.
- Usage ledger records token/cost accounting.

Table names are prefixed with ``gm_`` to avoid collisions in shared databases.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class RunRow(Base):
    """Row model for ``gm_agent_runs``.

    Stores the lifecycle and metadata of an agent run.

    Key fields:

    - ``status``: coarse runtime status (running/paused/succeeded/failed).
    - ``role``/``autonomy_profile``: behavior and approval posture context.
    - ``objective``: the user objective being solved.
    """

    __tablename__ = "gm_agent_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    workspace_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    role: Mapped[str] = mapped_column(String(64))
    autonomy_profile: Mapped[str] = mapped_column(String(32))

    objective: Mapped[str] = mapped_column(Text)
    done_when: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    prompt_provider_version: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    status: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class EventRow(Base):
    """Row model for ``gm_agent_events``.

    Append-only event stream for a run.

    ``payload`` is stored as JSONB to capture structured details for auditing
    and debugging.
    """

    __tablename__ = "gm_agent_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)

    type: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    correlation_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)


class ToolInvocationRow(Base):
    """Row model for ``gm_tool_invocations``.

    Records side-effecting invocations made by Action steps.
    """

    __tablename__ = "gm_tool_invocations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)

    server_id: Mapped[str] = mapped_column(String(128))
    tool_name: Mapped[str] = mapped_column(String(256))

    args: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    ok: Mapped[bool] = mapped_column(Boolean)
    result: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)

    risk: Mapped[str] = mapped_column(String(16))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ApprovalRow(Base):
    """Row model for ``gm_approvals``.

    Stores approval requests and resolutions.
    """

    __tablename__ = "gm_approvals"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)

    risk: Mapped[str] = mapped_column(String(16))
    capability: Mapped[str] = mapped_column(String(64))

    reason: Mapped[str] = mapped_column(Text)
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    decision: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    decided_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)


class CheckpointRow(Base):
    """Row model for ``gm_checkpoints``.

    Checkpoints store serialized LangGraph state required for pause/resume.
    """

    __tablename__ = "gm_checkpoints"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)

    node: Mapped[str] = mapped_column(String(128))
    state: Mapped[Dict[str, Any]] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class UsageRow(Base):
    """Row model for ``gm_usage_ledger``.

    Append-only token/cost accounting per run.
    """

    __tablename__ = "gm_usage_ledger"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)

    provider: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    prompt_tokens: Mapped[int] = mapped_column(Integer)
    completion_tokens: Mapped[int] = mapped_column(Integer)
    total_tokens: Mapped[int] = mapped_column(Integer)

    cost_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
