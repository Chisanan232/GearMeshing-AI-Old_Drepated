"""Test configuration for database unit tests.

This module provides common fixtures and utilities for testing the
centralized database layer with in-memory SQLite and mocked dependencies.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import AsyncGenerator, Generator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from gearmeshing_ai.core.database.base import Base


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def in_memory_engine() -> AsyncGenerator:
    """Create in-memory SQLite engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True,
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture(scope="function")
async def in_memory_session(
    in_memory_engine
) -> AsyncGenerator[AsyncSession, None]:
    """Create in-memory SQLite session for testing."""
    async_session = sessionmaker(
        bind=in_memory_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session() as session:
        yield session


@pytest.fixture(scope="function")
def sample_agent_run_data() -> dict:
    """Sample agent run data for testing."""
    return {
        "id": "test_run_123",
        "tenant_id": "tenant_456",
        "workspace_id": "workspace_789",
        "role": "developer",
        "autonomy_profile": "balanced",
        "objective": "Build a new feature",
        "done_when": "Feature is complete and tested",
        "prompt_provider_version": "v1.0.0",
        "status": "running",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }


@pytest.fixture(scope="function")
def sample_agent_event_data() -> dict:
    """Sample agent event data for testing."""
    return {
        "id": "test_event_123",
        "run_id": "test_run_123",
        "type": "step_completed",
        "correlation_id": "correlation_456",
        "payload": {"step": "analysis", "result": "success"},
        "created_at": datetime.utcnow(),
    }


@pytest.fixture(scope="function")
def sample_agent_config_data() -> dict:
    """Sample agent configuration data for testing."""
    return {
        "role_name": "developer",
        "display_name": "Software Developer",
        "description": "Handles software development tasks",
        "system_prompt_key": "developer_prompt",
        "model_provider": "openai",
        "model_name": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 4096,
        "top_p": 0.9,
        "capabilities": '["code_generation", "debugging", "testing"]',
        "tools": '["git", "python", "docker"]',
        "autonomy_profiles": '["balanced", "conservative"]',
        "done_when": "All tasks completed successfully",
        "is_active": True,
        "tenant_id": None,
    }


@pytest.fixture(scope="function")
def sample_chat_session_data() -> dict:
    """Sample chat session data for testing."""
    return {
        "title": "Project Discussion",
        "description": "Discussion about new project features",
        "agent_role": "developer",
        "tenant_id": "tenant_456",
        "run_id": "test_run_123",
        "is_active": True,
    }


@pytest.fixture(scope="function")
def sample_policy_data() -> dict:
    """Sample policy data for testing."""
    return {
        "id": "policy_123",
        "tenant_id": "tenant_456",
        "config": {
            "risk_threshold": "medium",
            "allowed_capabilities": ["code_generation", "analysis"],
            "approval_required": True,
            "max_tokens_per_request": 4096,
            "allowed_models": ["gpt-4o", "claude-3-sonnet"]
        },
    }


@pytest.fixture(scope="function")
def sample_approval_data() -> dict:
    """Sample approval data for testing."""
    return {
        "id": "approval_123",
        "run_id": "run_456",
        "risk": "high",
        "capability": "code_execution",
        "reason": "Attempting to execute potentially risky code",
        "requested_at": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(hours=1),
        "decision": None,
        "decided_at": None,
        "decided_by": None,
    }


@pytest.fixture(scope="function")
def sample_usage_ledger_data() -> dict:
    """Sample usage ledger data for testing."""
    return {
        "id": "usage_123",
        "run_id": "run_456",
        "tenant_id": "tenant_789",
        "provider": "openai",
        "model": "gpt-4o",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "cost_usd": 0.0045,
    }
