"""
Shared fixtures for runtime engine smoke tests.

This module provides common fixtures used across all runtime test files.
Uses testcontainers with Docker Compose for real services integration.
"""

from __future__ import annotations

from test.settings import test_settings
from typing import Any, Dict, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gearmeshing_ai.agent_core.capabilities.base import CapabilityResult
from gearmeshing_ai.agent_core.policy.global_policy import GlobalPolicy
from gearmeshing_ai.core.models.domain import RiskLevel


@pytest.fixture(scope="session")
def agent_configs_setup(compose_stack):
    """Set up agent configurations in the database for runtime tests."""
    from sqlalchemy import create_engine
    from sqlmodel import Session

    # Use the PostgreSQL database from testcontainers
    db_url = "postgresql+asyncpg://ai_dev:changeme@localhost:5432/ai_dev"
    # Convert to sync URL for SQLAlchemy
    sync_db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

    engine = create_engine(sync_db_url)

    # Import the SQLModel tables
    from gearmeshing_ai.server.models.agent_config import AgentConfig

    # Insert test agent configurations
    with Session(engine) as session:
        # Check if assistant config already exists
        existing = (
            session.query(AgentConfig)
            .filter(AgentConfig.role_name == "assistant", AgentConfig.model_provider == "openai")
            .first()
        )

        if not existing:
            # OpenAI agent config
            openai_config = AgentConfig(
                role_name="assistant",
                display_name="Assistant",
                description="General assistant role",
                system_prompt_key="assistant",
                model_provider="openai",
                model_name="gpt-4o",
                temperature=0.7,
                max_tokens=4096,
                top_p=0.9,
                tenant_id=None,  # No tenant for general lookup
            )
            session.add(openai_config)

            # Anthropic agent config
            anthropic_config = AgentConfig(
                role_name="assistant",
                display_name="Assistant",
                description="General assistant role",
                system_prompt_key="assistant",
                model_provider="anthropic",
                model_name="claude-3-opus-20240229",
                temperature=0.7,
                max_tokens=4096,
                top_p=0.9,
                tenant_id=None,  # No tenant for general lookup
            )
            session.add(anthropic_config)

            # Google agent config
            google_config = AgentConfig(
                role_name="assistant",
                display_name="Assistant",
                description="General assistant role",
                system_prompt_key="assistant",
                model_provider="google",
                model_name="gemini-pro",
                temperature=0.7,
                max_tokens=4096,
                top_p=0.9,
                tenant_id=None,  # No tenant for general lookup
            )
            session.add(google_config)

            session.commit()

    yield


@pytest.fixture(scope="function")
def patched_settings() -> Generator[Any, None, None]:
    """Patch the main settings to use test settings for smoke tests."""
    # Create a mock settings that has the right structure for the model provider
    from gearmeshing_ai.server.core.config import DatabaseConfig

    mock_settings = MagicMock()
    # Copy AI provider settings from test_settings
    mock_settings.ai_provider = test_settings.ai_provider
    # Create a proper DatabaseConfig with url attribute
    # Use the PostgreSQL database from testcontainers
    mock_settings.database = DatabaseConfig()
    mock_settings.database.url = "postgresql+asyncpg://ai_dev:changeme@localhost:5432/ai_dev"

    # Patch the main config settings (this affects the import in model_provider)
    with patch("gearmeshing_ai.server.core.config.settings", mock_settings):
        yield test_settings


@pytest.fixture
def mock_repositories() -> Dict[str, AsyncMock]:
    """Mock all repository dependencies."""
    return {
        "runs": AsyncMock(),
        "events": AsyncMock(),
        "approvals": AsyncMock(),
        "checkpoints": AsyncMock(),
        "tool_invocations": AsyncMock(),
        "usage": AsyncMock(),
    }


@pytest.fixture
def mock_capabilities() -> MagicMock:
    """Mock capabilities registry with realistic capabilities."""
    capabilities = MagicMock()
    capabilities.list_all.return_value = [
        {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {"query": "string", "max_results": "integer"},
        },
        {
            "name": "web_fetch",
            "description": "Fetch content from a URL",
            "parameters": {"url": "string"},
        },
        {
            "name": "docs_read",
            "description": "Read documentation or files",
            "parameters": {"path": "string"},
        },
        {
            "name": "summarize",
            "description": "Summarize content",
            "parameters": {"content": "string", "focus": "string"},
        },
        {
            "name": "shell_exec",
            "description": "Execute shell commands",
            "parameters": {"command": "string", "working_dir": "string"},
        },
    ]

    # Mock the get method to return async mock capability
    async_mock_capability = AsyncMock()
    async_mock_capability.execute.return_value = CapabilityResult(
        ok=True, output={"status": "success", "data": "mock result"}
    )
    capabilities.get.return_value = async_mock_capability

    return capabilities


@pytest.fixture
def mock_policy() -> GlobalPolicy:
    """Mock global policy for testing."""
    policy = MagicMock(spec=GlobalPolicy)

    # Default policy: allow everything
    mock_decision = MagicMock()
    mock_decision.block = False
    mock_decision.block_reason = None
    mock_decision.require_approval = False
    mock_decision.risk = RiskLevel.low

    # Setup all required methods
    policy.decide.return_value = mock_decision
    policy.validate_tool_args.return_value = None
    policy.classify_risk.return_value = RiskLevel.low

    return policy


@pytest.fixture
def sample_agent_run() -> Any:
    """Sample agent run for testing."""
    import uuid
    from datetime import datetime, timezone

    from gearmeshing_ai.core.models.domain import AgentRun, AgentRunStatus

    return AgentRun(
        id=str(uuid.uuid4()),
        role="assistant",
        objective="Test AI agent workflow execution",
        status=AgentRunStatus.pending,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tenant_id="test-tenant",
    )


@pytest.fixture
def engine_deps(
    mock_repositories: Dict[str, AsyncMock],
    mock_capabilities: MagicMock,
) -> Any:
    """Engine dependencies fixture."""
    from langgraph.checkpoint.memory import MemorySaver

    from gearmeshing_ai.agent_core.runtime.models import EngineDeps

    return EngineDeps(
        runs=mock_repositories["runs"],
        events=mock_repositories["events"],
        approvals=mock_repositories["approvals"],
        checkpoints=mock_repositories["checkpoints"],
        tool_invocations=mock_repositories["tool_invocations"],
        capabilities=mock_capabilities,
        usage=mock_repositories["usage"],
        checkpointer=MemorySaver(),
        thought_model=None,
        prompt_provider=None,
        role_provider=None,
        mcp_info_provider=None,
        mcp_call=None,
    )
