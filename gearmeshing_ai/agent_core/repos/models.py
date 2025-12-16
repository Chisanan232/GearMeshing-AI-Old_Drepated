from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import String, Text, DateTime, Boolean, Integer, Float
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RunRow(Base):
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
    __tablename__ = "gm_agent_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)

    type: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    correlation_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)


class ToolInvocationRow(Base):
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
    __tablename__ = "gm_checkpoints"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)

    node: Mapped[str] = mapped_column(String(128))
    state: Mapped[Dict[str, Any]] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class UsageRow(Base):
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
