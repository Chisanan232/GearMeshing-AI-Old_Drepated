"""Initial schema and seed data for GearMeshing-AI

Revision ID: 20251224_000000
Revises: None
Create Date: 2025-12-24 00:00:00.000000

This is the initial migration that creates all necessary tables and seeds default data
for the GearMeshing-AI service. This includes:
- Agent core tables (runs, events, tool invocations, approvals, checkpoints, policies, usage)
- Server tables (agent configs, chat sessions)
- Default AI agent roles and configurations
- Default policies and model settings

Revision format: YYYYMMDD_HHMMSS_description

"""

import json
from datetime import datetime
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, ENUM

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20251224_000000"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all tables and seed initial data."""

    # Create gm_agent_runs table
    op.create_table(
        "gm_agent_runs",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("tenant_id", sa.String(128), nullable=True),
        sa.Column("workspace_id", sa.String(128), nullable=True),
        sa.Column("role", sa.String(64), nullable=False),
        sa.Column("autonomy_profile", sa.String(32), nullable=False),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("done_when", sa.Text(), nullable=True),
        sa.Column("prompt_provider_version", sa.String(128), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.Index("ix_gm_agent_runs_tenant_id", "tenant_id"),
    )

    # Create gm_agent_events table
    op.create_table(
        "gm_agent_events",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("type", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("correlation_id", sa.String(128), nullable=True),
        sa.Column("payload", JSONB(), nullable=False, server_default="{}"),
        sa.PrimaryKeyConstraint("id"),
        sa.Index("ix_gm_agent_events_run_id", "run_id"),
    )

    # Create gm_tool_invocations table
    op.create_table(
        "gm_tool_invocations",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("server_id", sa.String(128), nullable=False),
        sa.Column("tool_name", sa.String(256), nullable=False),
        sa.Column("args", JSONB(), nullable=False, server_default="{}"),
        sa.Column("ok", sa.Boolean(), nullable=False),
        sa.Column("result", JSONB(), nullable=False, server_default="{}"),
        sa.Column("risk", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.Index("ix_gm_tool_invocations_run_id", "run_id"),
    )

    # Create gm_approvals table
    op.create_table(
        "gm_approvals",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("risk", sa.String(16), nullable=False),
        sa.Column("capability", sa.String(64), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decision", sa.String(16), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decided_by", sa.String(128), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.Index("ix_gm_approvals_run_id", "run_id"),
    )

    # Create gm_checkpoints table
    op.create_table(
        "gm_checkpoints",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("node", sa.String(128), nullable=False),
        sa.Column("state", JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.Index("ix_gm_checkpoints_run_id", "run_id"),
        sa.Index("ix_gm_checkpoints_created_at", "created_at"),
    )

    # Create gm_policies table
    op.create_table(
        "gm_policies",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("config", JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id"),
        sa.Index("ix_gm_policies_tenant_id", "tenant_id"),
    )

    # Create gm_usage_ledger table
    op.create_table(
        "gm_usage_ledger",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("tenant_id", sa.String(128), nullable=True),
        sa.Column("provider", sa.String(64), nullable=True),
        sa.Column("model", sa.String(128), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("total_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.Index("ix_gm_usage_ledger_run_id", "run_id"),
        sa.Index("ix_gm_usage_ledger_tenant_id", "tenant_id"),
    )

    # Create agent_configs table
    op.create_table(
        "agent_configs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("role_name", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("system_prompt_key", sa.String(), nullable=False),
        sa.Column("model_provider", sa.String(), nullable=False),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=False),
        sa.Column("max_tokens", sa.Integer(), nullable=False),
        sa.Column("top_p", sa.Float(), nullable=False),
        sa.Column("capabilities", sa.String(), nullable=False),
        sa.Column("tools", sa.String(), nullable=False),
        sa.Column("autonomy_profiles", sa.String(), nullable=False),
        sa.Column("done_when", sa.String(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.Index("ix_agent_configs_role_name", "role_name"),
        sa.Index("ix_agent_configs_tenant_id", "tenant_id"),
    )

    # Create chat_sessions table
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=True, index=True),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("agent_role", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=True, index=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create chat_messages table
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        # Create MessageRole enum type for chat_messages
        sa.Column("role", ENUM('user', 'assistant', 'system', name='messagerole', create_type=True)),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("message_metadata", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"]),
    )

    # Seed default agent configurations
    now = datetime.utcnow()

    # Define default agent configurations
    default_configs = [
        {
            "role_name": "planner",
            "display_name": "Planner",
            "description": "Strategic planning and task decomposition agent",
            "system_prompt_key": "planner_system_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 0.9,
            "capabilities": json.dumps(["planning", "analysis", "reasoning"]),
            "tools": json.dumps(["search", "calculator", "code_executor"]),
            "autonomy_profiles": json.dumps(["strict", "balanced", "autonomous"]),
            "done_when": "Task completed and verified",
            "is_active": True,
            "tenant_id": None,
            "created_at": now,
            "updated_at": now,
        },
        {
            "role_name": "dev",
            "display_name": "Developer",
            "description": "Code development and implementation agent",
            "system_prompt_key": "dev_system_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "temperature": 0.5,
            "max_tokens": 8192,
            "top_p": 0.9,
            "capabilities": json.dumps(["coding", "debugging", "testing"]),
            "tools": json.dumps(["code_executor", "git", "package_manager"]),
            "autonomy_profiles": json.dumps(["strict", "balanced"]),
            "done_when": "Code implemented, tested, and committed",
            "is_active": True,
            "tenant_id": None,
            "created_at": now,
            "updated_at": now,
        },
        {
            "role_name": "analyst",
            "display_name": "Analyst",
            "description": "Data analysis and insight generation agent",
            "system_prompt_key": "analyst_system_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "temperature": 0.6,
            "max_tokens": 4096,
            "top_p": 0.9,
            "capabilities": json.dumps(["analysis", "visualization", "reporting"]),
            "tools": json.dumps(["data_query", "visualization", "statistical_analysis"]),
            "autonomy_profiles": json.dumps(["strict", "balanced", "autonomous"]),
            "done_when": "Analysis complete with insights and recommendations",
            "is_active": True,
            "tenant_id": None,
            "created_at": now,
            "updated_at": now,
        },
    ]

    # Insert agent configurations in batch using raw SQL
    agent_configs_columns = [
        "role_name",
        "display_name",
        "description",
        "system_prompt_key",
        "model_provider",
        "model_name",
        "temperature",
        "max_tokens",
        "top_p",
        "capabilities",
        "tools",
        "autonomy_profiles",
        "done_when",
        "is_active",
        "tenant_id",
        "created_at",
        "updated_at",
    ]

    for config in default_configs:
        values = ", ".join(
            [
                f"'{config['role_name']}'",
                f"'{config['display_name']}'",
                f"'{config['description']}'",
                f"'{config['system_prompt_key']}'",
                f"'{config['model_provider']}'",
                f"'{config['model_name']}'",
                f"{config['temperature']}",
                f"{config['max_tokens']}",
                f"{config['top_p']}",
                f"'{config['capabilities']}'",
                f"'{config['tools']}'",
                f"'{config['autonomy_profiles']}'",
                f"'{config['done_when']}'",
                f"{'true' if config['is_active'] else 'false'}",
                f"NULL" if config["tenant_id"] is None else f"'{config['tenant_id']}'",
                f"'{config['created_at']}'",
                f"'{config['updated_at']}'",
            ]
        )
        op.execute(f"INSERT INTO agent_configs ({', '.join(agent_configs_columns)}) " f"VALUES ({values})")

    # Seed default policy for default tenant
    default_policies = [
        {
            "id": "policy_default_tenant",
            "tenant_id": "default-tenant",
            "config": json.dumps(
                {
                    "autonomy": "balanced",
                    "allowed_tools": ["search", "calculator", "code_executor"],
                    "max_budget": 100.0,
                    "approval_required_for": ["high_risk_operations"],
                    "rate_limits": {
                        "requests_per_minute": 60,
                        "tokens_per_day": 1000000,
                    },
                }
            ),
            "created_at": now,
            "updated_at": now,
        }
    ]

    gm_policies_columns = ["id", "tenant_id", "config", "created_at", "updated_at"]

    for policy in default_policies:
        values = ", ".join(
            [
                f"'{policy['id']}'",
                f"'{policy['tenant_id']}'",
                f"'{policy['config']}'",
                f"'{policy['created_at']}'",
                f"'{policy['updated_at']}'",
            ]
        )
        op.execute(f"INSERT INTO gm_policies ({', '.join(gm_policies_columns)}) " f"VALUES ({values})")


def downgrade() -> None:
    """Drop all tables created in upgrade."""
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
    op.drop_table("agent_configs")
    op.drop_table("gm_usage_ledger")
    op.drop_table("gm_policies")
    op.drop_table("gm_checkpoints")
    op.drop_table("gm_approvals")
    op.drop_table("gm_tool_invocations")
    op.drop_table("gm_agent_events")
    op.drop_table("gm_agent_runs")

    # Drop the enum type
    op.execute("DROP TYPE IF EXISTS messagerole")
