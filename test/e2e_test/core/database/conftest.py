"""Test configuration for database e2e tests.

This module provides fixtures and utilities for testing the
centralized database layer with real PostgreSQL via Docker compose.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from test.settings import test_settings
from typing import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import Session
from testcontainers.compose import DockerCompose

from gearmeshing_ai.core.database.base import Base

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@pytest.fixture(scope="session")
def _compose_env():
    """Set up environment variables for docker-compose.yml."""
    prev = {}

    def _set(k: str, v: str) -> None:
        if k in os.environ:
            prev[k] = os.environ[k]
        os.environ[k] = v

    # PostgreSQL - use test settings
    postgres_config = test_settings.database.postgres
    _set("DATABASE__POSTGRES__DB", postgres_config.db)
    _set("DATABASE__POSTGRES__USER", postgres_config.user)
    _set("DATABASE__POSTGRES__PASSWORD", postgres_config.password.get_secret_value())

    try:
        yield
    finally:
        for k in list(
            {
                "DATABASE__POSTGRES__DB",
                "DATABASE__POSTGRES__USER",
                "DATABASE__POSTGRES__PASSWORD",
            }
        ):
            if k in prev:
                os.environ[k] = prev[k]
            else:
                os.environ.pop(k, None)


@pytest.fixture(scope="session")
def compose_stack(_compose_env):
    """Start docker-compose stack for testing."""
    project_test_root = Path("./test")
    compose = DockerCompose(str(project_test_root))
    compose.start()

    # Wait for PostgreSQL to be ready
    time.sleep(10)

    try:
        yield compose
    finally:
        compose.stop()


@pytest.fixture(scope="session")
def database_url(compose_stack) -> str:
    """Get synchronous test database URL from test settings."""
    postgres_config = test_settings.database.postgres
    return f"postgresql://{postgres_config.user}:{postgres_config.password.get_secret_value()}@127.0.0.1:{postgres_config.port}/{postgres_config.db}"


@pytest.fixture(scope="function")
def postgres_engine(database_url: str) -> Generator:
    """Create PostgreSQL engine for testing."""
    engine = create_engine(database_url, echo=False)

    # Create all tables
    Base.metadata.create_all(engine)

    try:
        yield engine
    finally:
        # Drop all tables after test
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture(scope="function")
def postgres_session(postgres_engine) -> Generator[Session, None, None]:
    """Create PostgreSQL session for testing."""
    session_factory = sessionmaker(
        bind=postgres_engine,
        class_=Session,
        expire_on_commit=False,
    )

    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def sample_agent_run_data() -> dict:
    """Sample agent run data for testing."""
    from datetime import datetime

    return {
        "id": "e2e_test_run_123",
        "tenant_id": "e2e_tenant_456",
        "workspace_id": "e2e_workspace_789",
        "role": "developer",
        "autonomy_profile": "balanced",
        "objective": "Build and test e2e features",
        "done_when": "All e2e tests pass",
        "prompt_provider_version": "v1.0.0",
        "status": "running",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }


@pytest.fixture(scope="function")
def sample_agent_config_data() -> dict:
    """Sample agent configuration data for testing."""
    return {
        "role_name": "e2e_developer",
        "display_name": "E2E Test Developer",
        "description": "Handles e2e testing tasks",
        "system_prompt_key": "e2e_developer_prompt",
        "model_provider": "openai",
        "model_name": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 4096,
        "top_p": 0.9,
        "capabilities": '["testing", "debugging", "automation"]',
        "tools": '["pytest", "docker", "selenium"]',
        "autonomy_profiles": '["balanced", "thorough"]',
        "done_when": "All tests are green",
        "is_active": True,
        "tenant_id": None,
    }


@pytest.fixture(scope="function")
def sample_chat_session_data() -> dict:
    """Sample chat session data for testing."""
    return {
        "title": "E2E Test Discussion",
        "description": "Discussion about e2e test implementation",
        "agent_role": "e2e_developer",
        "tenant_id": "e2e_tenant_456",
        "run_id": "e2e_test_run_123",
        "is_active": True,
    }
