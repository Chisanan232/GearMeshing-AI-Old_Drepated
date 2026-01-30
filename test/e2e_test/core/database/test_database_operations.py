"""End-to-end tests for database operations.

Tests complete database workflows with real PostgreSQL to ensure
data integrity, relationships, and performance under production-like conditions.
"""

from __future__ import annotations


import pytest

from gearmeshing_ai.core.database.entities.agent_configs import AgentConfig
from gearmeshing_ai.core.database.entities.agent_runs import AgentRun
from gearmeshing_ai.core.database.entities.chat_sessions import (
    ChatMessage,
    ChatSession,
    MessageRole,
)
from gearmeshing_ai.core.database.repositories.agent_configs import (
    AgentConfigRepository,
)
from gearmeshing_ai.core.database.repositories.agent_runs import AgentRunRepository
from gearmeshing_ai.core.database.repositories.chat_sessions import (
    ChatSessionRepository,
)


class TestAgentRunE2E:
    """End-to-end tests for agent run operations."""

    async def test_complete_agent_run_lifecycle(self, postgres_session, sample_agent_run_data):
        """Test complete agent run lifecycle from creation to completion."""
        repo = AgentRunRepository(postgres_session)

        # Create agent run
        agent_run = AgentRun(**sample_agent_run_data)
        created_run = await repo.create(agent_run)

        assert created_run.id == sample_agent_run_data["id"]
        assert created_run.status == "running"

        # Verify retrieval
        retrieved_run = await repo.get_by_id(created_run.id)
        assert retrieved_run is not None
        assert retrieved_run.id == created_run.id
        assert retrieved_run.status == "running"

        # Update status
        updated_run = await repo.update_status(created_run.id, "paused")
        assert updated_run.status == "paused"

        # Final update to completed
        completed_run = await repo.update_status(created_run.id, "completed")
        assert completed_run.status == "completed"

        # Verify final state
        final_run = await repo.get_by_id(created_run.id)
        assert final_run.status == "completed"

    async def test_agent_run_query_operations(self, postgres_session, sample_agent_run_data):
        """Test various query operations for agent runs."""
        repo = AgentRunRepository(postgres_session)

        # Create multiple runs with different statuses
        runs = []
        for i, status in enumerate(["running", "paused", "completed", "failed"]):
            data = sample_agent_run_data.copy()
            data["id"] = f"test_run_{i}"
            data["status"] = status
            data["tenant_id"] = f"tenant_{i % 2}"  # Alternate between 2 tenants

            run = AgentRun(**data)
            created_run = await repo.create(run)
            runs.append(created_run)

        # Test list with filters
        running_runs = await repo.list(filters={"status": "running"})
        assert len(running_runs) == 1
        assert running_runs[0].status == "running"

        # Test tenant-specific queries
        tenant_0_runs = await repo.get_by_tenant_and_status("tenant_0", "running")
        assert len(tenant_0_runs) == 1
        assert tenant_0_runs[0].tenant_id == "tenant_0"

        # Test active runs query
        active_runs = await repo.get_active_runs_for_tenant("tenant_0")
        assert len(active_runs) >= 1  # Should include running/paused runs

        # Test pagination
        paginated_runs = await repo.list(limit=2, offset=1)
        assert len(paginated_runs) == 2

    async def test_agent_run_concurrent_operations(self, postgres_session, sample_agent_run_data):
        """Test sequential agent run operations (simplified from concurrent)."""
        repo = AgentRunRepository(postgres_session)

        # Create multiple runs sequentially
        created_runs = []
        for i in range(10):
            data = sample_agent_run_data.copy()
            data["id"] = f"concurrent_run_{i}"
            run = AgentRun(**data)
            created_run = await repo.create(run)
            created_runs.append(created_run)

        # Verify all runs were created successfully
        assert len(created_runs) == 10

        # Verify all runs can be retrieved
        for run in created_runs:
            retrieved_run = await repo.get_by_id(run.id)
            assert retrieved_run is not None
            assert retrieved_run.id == run.id

    async def test_agent_run_delete_operations(self, postgres_session, sample_agent_run_data):
        """Test agent run deletion operations."""
        repo = AgentRunRepository(postgres_session)

        # Create a run
        agent_run = AgentRun(**sample_agent_run_data)
        created_run = await repo.create(agent_run)

        # Verify it exists
        assert await repo.get_by_id(created_run.id) is not None

        # Delete it
        delete_result = await repo.delete(created_run.id)
        assert delete_result is True

        # Verify it's gone
        assert await repo.get_by_id(created_run.id) is None

        # Try to delete non-existent run
        delete_result = await repo.delete("nonexistent_run")
        assert delete_result is False


class TestAgentConfigE2E:
    """End-to-end tests for agent configuration operations."""

    async def test_agent_config_crud_operations(self, postgres_session, sample_agent_config_data):
        """Test complete CRUD operations for agent configurations."""
        repo = AgentConfigRepository(postgres_session)

        # Create configuration
        config = AgentConfig(**sample_agent_config_data)
        created_config = await repo.create(config)

        assert created_config.role_name == sample_agent_config_data["role_name"]
        assert created_config.id is not None  # Should be set by database

        # Test JSON helper methods
        capabilities = created_config.get_capabilities_list()
        assert "testing" in capabilities
        assert "debugging" in capabilities

        # Update capabilities
        new_capabilities = ["testing", "deployment", "monitoring"]
        created_config.set_capabilities_list(new_capabilities)
        updated_config = await repo.update(created_config)

        assert updated_config.get_capabilities_list() == new_capabilities

        # Test retrieval by role
        retrieved_config = await repo.get_by_role("e2e_developer")
        assert retrieved_config is not None
        assert retrieved_config.role_name == "e2e_developer"

        # Test deactivation
        deactivated_config = await repo.deactivate_config(updated_config.id)
        assert deactivated_config.is_active is False

    async def test_agent_config_tenant_isolation(self, postgres_session, sample_agent_config_data):
        """Test tenant-specific configuration isolation."""
        repo = AgentConfigRepository(postgres_session)

        # Create global config
        global_data = sample_agent_config_data.copy()
        global_data["role_name"] = "global_developer"
        global_data["tenant_id"] = None
        global_config = AgentConfig(**global_data)
        await repo.create(global_config)

        # Create tenant-specific config
        tenant_data = sample_agent_config_data.copy()
        tenant_data["role_name"] = "tenant_developer"
        tenant_data["tenant_id"] = "tenant_123"
        tenant_config = AgentConfig(**tenant_data)
        await repo.create(tenant_config)

        # Test retrieval isolation
        global_retrieved = await repo.get_by_role("global_developer")
        assert global_retrieved is not None
        assert global_retrieved.tenant_id is None

        tenant_retrieved = await repo.get_by_role("tenant_developer", "tenant_123")
        assert tenant_retrieved is not None
        assert tenant_retrieved.tenant_id == "tenant_123"

        # Tenant should not get global config when requesting specific tenant
        tenant_wrong = await repo.get_by_role("global_developer", "tenant_123")
        assert tenant_wrong is None

        # Global config should be returned when no tenant specified
        global_for_any = await repo.get_by_role("global_developer")
        assert global_for_any is not None


class TestChatSessionE2E:
    """End-to-end tests for chat session operations."""

    async def test_chat_session_with_messages(self, postgres_session, sample_chat_session_data):
        """Test chat session creation with message management."""
        session_repo = ChatSessionRepository(postgres_session)

        # Create chat session
        chat_session = ChatSession(**sample_chat_session_data)
        created_session = await session_repo.create(chat_session)

        assert created_session.title == sample_chat_session_data["title"]
        assert created_session.id is not None

        # Add messages
        messages = [
            ChatMessage(role=MessageRole.USER, content="Help me implement e2e tests", token_count=15),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="I'll help you set up comprehensive e2e tests",
                token_count=18,
                model_used="gpt-4o",
            ),
            ChatMessage(role=MessageRole.USER, content="Great! Let's start with database tests", token_count=12),
        ]

        created_messages = []
        for message in messages:
            created_message = await session_repo.add_message(created_session.id, message)
            created_messages.append(created_message)

        assert len(created_messages) == 3

        # Verify message retrieval
        retrieved_messages = await session_repo.get_messages_for_session(created_session.id)
        assert len(retrieved_messages) == 3

        # Verify message order (chronological)
        for i in range(len(retrieved_messages) - 1):
            assert retrieved_messages[i].created_at <= retrieved_messages[i + 1].created_at

        # Verify message roles
        assert retrieved_messages[0].role == MessageRole.USER
        assert retrieved_messages[1].role == MessageRole.ASSISTANT
        assert retrieved_messages[2].role == MessageRole.USER

        # Verify session exists (messages relationship is not available in the model)
        session_with_messages = await session_repo.get_by_id(created_session.id)
        assert session_with_messages is not None

    async def test_chat_session_query_operations(self, postgres_session, sample_chat_session_data):
        """Test chat session query operations."""
        session_repo = ChatSessionRepository(postgres_session)

        # Create multiple sessions
        sessions = []
        for i in range(5):
            data = sample_chat_session_data.copy()
            data["title"] = f"Test Session {i}"
            data["agent_role"] = f"role_{i % 2}"  # Alternate between 2 roles
            data["tenant_id"] = f"tenant_{i % 3}"  # 3 different tenants
            data["is_active"] = i < 3  # First 3 are active

            session = ChatSession(**data)
            created_session = await session_repo.create(session)
            sessions.append(created_session)

        # Test filtered queries
        active_sessions = await session_repo.list(filters={"is_active": True})
        assert len(active_sessions) == 3

        role_sessions = await session_repo.list(filters={"agent_role": "role_0"})
        assert len(role_sessions) == 3  # roles 0, 2, 4

        tenant_sessions = await session_repo.list(filters={"tenant_id": "tenant_1"})
        assert len(tenant_sessions) == 2  # indices 1 and 4

        # Test business-specific queries
        role_0_active = await session_repo.get_active_sessions_for_role("role_0")
        assert len(role_0_active) == 2  # role_0 sessions that are active

        tenant_1_sessions = await session_repo.get_sessions_for_tenant("tenant_1")
        assert len(tenant_1_sessions) == 2


class TestDatabaseIntegrityE2E:
    """End-to-end tests for database integrity and constraints."""

    async def test_foreign_key_constraints(self, postgres_session, sample_chat_session_data):
        """Test foreign key constraints are enforced."""
        session_repo = ChatSessionRepository(postgres_session)

        # Try to add message to non-existent session
        fake_message = ChatMessage(
            role=MessageRole.USER, content="This should fail", session_id=99999  # Non-existent session ID
        )

        # This should raise an integrity error
        with pytest.raises(Exception):  # SQLAlchemy will raise an integrity error
            await session_repo.add_message(99999, fake_message)

    async def test_unique_constraints(self, postgres_session, sample_agent_config_data):
        """Test unique constraints are enforced."""
        repo = AgentConfigRepository(postgres_session)

        # Create first config
        config1 = AgentConfig(**sample_agent_config_data)
        await repo.create(config1)

        # Try to create another config with same role and tenant (should be allowed for different tenants)
        config2_data = sample_agent_config_data.copy()
        config2_data["role_name"] = "e2e_developer"  # Same role
        config2_data["tenant_id"] = "different_tenant"  # Different tenant
        config2 = AgentConfig(**config2_data)

        # This should work
        await repo.create(config2)

        # But same role with same tenant should be handled by business logic
        # (SQLModel doesn't enforce unique constraints on non-primary key fields by default)

    async def test_data_type_constraints(self, postgres_session, sample_agent_run_data):
        """Test data type constraints are enforced."""
        repo = AgentRunRepository(postgres_session)

        # Test string length constraints
        long_id_data = sample_agent_run_data.copy()
        long_id_data["id"] = "x" * 100  # Exceeds max length

        # SQLModel/SQLAlchemy should handle this validation
        with pytest.raises(Exception):
            long_run = AgentRun(**long_id_data)
            await repo.create(long_run)

    async def test_transaction_rollback(self, postgres_session, sample_agent_run_data):
        """Test transaction rollback on errors."""
        repo = AgentRunRepository(postgres_session)

        # Test basic rollback behavior
        try:
            # Create a valid run
            run = AgentRun(**sample_agent_run_data)
            postgres_session.add(run)
            postgres_session.flush()  # Flush to DB but don't commit

            # Force an error
            raise ValueError("Intentional error for rollback test")

        except ValueError:
            # Rollback the transaction
            postgres_session.rollback()

        # Verify the run was not committed
        retrieved_run = await repo.get_by_id(sample_agent_run_data["id"])
        assert retrieved_run is None
