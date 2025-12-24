"""End-to-end tests for Alembic migration script.

Tests verify that the initial migration script:
1. Creates all required tables with correct schema
2. Creates all required indexes
3. Seeds default data correctly
4. Can be run multiple times without errors
5. Can be downgraded and upgraded again
"""

import json
import os
import subprocess
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@pytest.fixture
async def async_engine(database_url: str):
    """Create async SQLAlchemy engine."""
    engine = create_async_engine(database_url, echo=False)
    yield engine
    await engine.dispose()


@pytest.fixture
async def async_session(async_session_factory):
    """Create async session instance."""
    async with async_session_factory() as session:
        yield session


class TestMigrationDryRun:
    """Test migration script with dry-run (SQL generation)."""

    def test_migration_dry_run_generates_valid_sql(self):
        """Test that migration script generates valid SQL without errors."""
        result = subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head", "--sql"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Migration dry-run failed: {result.stderr}"
        assert "CREATE TABLE" in result.stdout
        assert "INSERT INTO" in result.stdout
        assert "agent_configs" in result.stdout
        assert "gm_policies" in result.stdout

    def test_migration_dry_run_creates_all_tables(self):
        """Test that migration creates all required tables."""
        result = subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head", "--sql"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        required_tables = [
            "gm_agent_runs",
            "gm_agent_events",
            "gm_tool_invocations",
            "gm_approvals",
            "gm_checkpoints",
            "gm_policies",
            "gm_usage_ledger",
            "agent_configs",
            "chat_sessions",
        ]

        for table in required_tables:
            assert f"CREATE TABLE {table}" in result.stdout, f"Table {table} not created in migration"

    def test_migration_dry_run_creates_all_indexes(self):
        """Test that migration creates all required indexes."""
        result = subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head", "--sql"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        required_indexes = [
            "ix_gm_agent_runs_tenant_id",
            "ix_gm_agent_events_run_id",
            "ix_gm_tool_invocations_run_id",
            "ix_gm_approvals_run_id",
            "ix_gm_checkpoints_run_id",
            "ix_gm_checkpoints_created_at",
            "ix_gm_policies_tenant_id",
            "ix_gm_usage_ledger_tenant_id",
            "ix_gm_usage_ledger_run_id",
            "ix_agent_configs_tenant_id",
            "ix_agent_configs_role_name",
            "ix_chat_sessions_tenant_id",
        ]

        for index in required_indexes:
            assert f"CREATE INDEX {index}" in result.stdout, f"Index {index} not created in migration"

    def test_migration_dry_run_seeds_default_data(self):
        """Test that migration seeds default agent configs and policies."""
        result = subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head", "--sql"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        # Check agent configs are seeded
        assert "INSERT INTO agent_configs" in result.stdout
        assert "'planner'" in result.stdout
        assert "'dev'" in result.stdout
        assert "'analyst'" in result.stdout

        # Check policies are seeded
        assert "INSERT INTO gm_policies" in result.stdout
        assert "'policy_default_tenant'" in result.stdout


class TestMigrationExecution:
    """Test actual migration execution against database."""

    @pytest.mark.requires_db
    def test_migration_creates_tables(self, database_url: str):
        """Test that migration creates all tables in database."""
        # Run migration
        result = subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "DATABASE_URL": database_url},
        )

        assert result.returncode == 0, f"Migration failed: {result.stderr}"

        # Verify tables exist using sync connection
        sync_url = database_url.replace("+asyncpg", "")
        engine = sa.create_engine(sync_url)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        engine.dispose()

        required_tables = [
            "gm_agent_runs",
            "gm_agent_events",
            "gm_tool_invocations",
            "gm_approvals",
            "gm_checkpoints",
            "gm_policies",
            "gm_usage_ledger",
            "agent_configs",
            "chat_sessions",
        ]

        for table in required_tables:
            assert table in tables, f"Table {table} not found in database"

    @pytest.mark.requires_db
    def test_migration_creates_indexes(self, database_url: str):
        """Test that migration creates all required indexes."""
        # Run migration
        result = subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "DATABASE_URL": database_url},
        )

        assert result.returncode == 0, f"Migration failed: {result.stderr}"

        # Verify indexes exist using sync connection
        sync_url = database_url.replace("+asyncpg", "")
        engine = sa.create_engine(sync_url)
        inspector = inspect(engine)

        # Check indexes for each table
        indexes_by_table = {
            "gm_agent_runs": ["ix_gm_agent_runs_tenant_id"],
            "gm_agent_events": ["ix_gm_agent_events_run_id"],
            "gm_tool_invocations": ["ix_gm_tool_invocations_run_id"],
            "gm_approvals": ["ix_gm_approvals_run_id"],
            "gm_checkpoints": ["ix_gm_checkpoints_run_id", "ix_gm_checkpoints_created_at"],
            "gm_policies": ["ix_gm_policies_tenant_id"],
            "gm_usage_ledger": ["ix_gm_usage_ledger_tenant_id", "ix_gm_usage_ledger_run_id"],
            "agent_configs": ["ix_agent_configs_tenant_id", "ix_agent_configs_role_name"],
            "chat_sessions": ["ix_chat_sessions_tenant_id"],
        }

        for table, expected_indexes in indexes_by_table.items():
            indexes = inspector.get_indexes(table)
            index_names = [idx["name"] for idx in indexes]

            for expected_index in expected_indexes:
                assert expected_index in index_names, f"Index {expected_index} not found on table {table}"

        engine.dispose()

    @pytest.mark.requires_db
    def test_migration_seeds_agent_configs(self, database_url: str):
        """Test that migration seeds default agent configurations."""
        # Run migration
        result = subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "DATABASE_URL": database_url},
        )

        assert result.returncode == 0, f"Migration failed: {result.stderr}"

        # Query agent configs using sync connection
        sync_url = database_url.replace("+asyncpg", "")
        engine = sa.create_engine(sync_url)

        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT role_name, display_name, model_name FROM agent_configs ORDER BY role_name")
            )
            configs = result.fetchall()

            assert len(configs) == 3, f"Expected 3 agent configs, got {len(configs)}"

            config_dict = {row[0]: row for row in configs}

            # Verify planner config
            assert "planner" in config_dict
            planner = config_dict["planner"]
            assert planner[1] == "Planner"
            assert planner[2] == "gpt-4o"

            # Verify dev config
            assert "dev" in config_dict
            dev = config_dict["dev"]
            assert dev[1] == "Developer"
            assert dev[2] == "gpt-4o"

            # Verify analyst config
            assert "analyst" in config_dict
            analyst = config_dict["analyst"]
            assert analyst[1] == "Analyst"
            assert analyst[2] == "gpt-4o"

        engine.dispose()

    @pytest.mark.requires_db
    def test_migration_seeds_agent_config_details(self, database_url: str):
        """Test that agent config details are correctly seeded."""
        # Run migration
        result = subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "DATABASE_URL": database_url},
        )

        assert result.returncode == 0, f"Migration failed: {result.stderr}"

        # Query planner config details using sync connection
        sync_url = database_url.replace("+asyncpg", "")
        engine = sa.create_engine(sync_url)

        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT role_name, temperature, max_tokens, top_p,
                           capabilities, tools, autonomy_profiles, is_active
                    FROM agent_configs
                    WHERE role_name = 'planner'
                """
                )
            )
            row = result.fetchone()

            assert row is not None
            role_name, temp, max_tokens, top_p, caps, tools, profiles, is_active = row

            assert role_name == "planner"
            assert temp == 0.7
            assert max_tokens == 4096
            assert top_p == 0.9
            assert is_active is True

            # Verify JSON fields
            capabilities = json.loads(caps)
            assert "planning" in capabilities
            assert "analysis" in capabilities

            tools_list = json.loads(tools)
            assert "search" in tools_list
            assert "calculator" in tools_list

            profiles_list = json.loads(profiles)
            assert "strict" in profiles_list
            assert "balanced" in profiles_list

        engine.dispose()

    @pytest.mark.requires_db
    def test_migration_seeds_default_policy(self, database_url: str):
        """Test that migration seeds default policy."""
        # Run migration
        result = subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "DATABASE_URL": database_url},
        )

        assert result.returncode == 0, f"Migration failed: {result.stderr}"

        # Query default policy using sync connection
        sync_url = database_url.replace("+asyncpg", "")
        engine = sa.create_engine(sync_url)

        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT id, tenant_id, config
                    FROM gm_policies
                    WHERE id = 'policy_default_tenant'
                """
                )
            )
            row = result.fetchone()

            assert row is not None
            policy_id, tenant_id, config_data = row

            assert policy_id == "policy_default_tenant"
            assert tenant_id == "default-tenant"

            # Verify policy config (handle both dict and string formats)
            if isinstance(config_data, str):
                config = json.loads(config_data)
            else:
                config = config_data

            assert config["autonomy"] == "balanced"
            assert "search" in config["allowed_tools"]
            assert config["max_budget"] == 100.0
            assert config["rate_limits"]["requests_per_minute"] == 60

        engine.dispose()

    @pytest.mark.requires_db
    def test_migration_table_schema(self, database_url: str):
        """Test that tables have correct schema and column types."""
        # Run migration
        result = subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "DATABASE_URL": database_url},
        )

        assert result.returncode == 0, f"Migration failed: {result.stderr}"

        # Verify agent_configs table schema using sync connection
        sync_url = database_url.replace("+asyncpg", "")
        engine = sa.create_engine(sync_url)
        inspector = inspect(engine)
        columns = inspector.get_columns("agent_configs")

        column_dict = {col["name"]: col for col in columns}

        # Verify required columns exist
        required_columns = [
            "id",
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

        for col_name in required_columns:
            assert col_name in column_dict, f"Column {col_name} not found in agent_configs"

        engine.dispose()

    @pytest.mark.requires_db
    def test_migration_idempotent(self, database_url: str):
        """Test that migration can be run multiple times without errors."""
        # First run
        result1 = subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "DATABASE_URL": database_url},
        )

        assert result1.returncode == 0, f"First migration failed: {result1.stderr}"

        # Second run (should be idempotent)
        result2 = subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "DATABASE_URL": database_url},
        )

        # Should succeed or indicate already at head
        assert (
            result2.returncode == 0 or "already at head" in result2.stdout.lower()
        ), f"Second migration failed: {result2.stderr}"

    @pytest.mark.requires_db
    def test_migration_downgrade_upgrade(self, database_url: str):
        """Test that migration can be downgraded and upgraded again."""
        env = {**os.environ, "DATABASE_URL": database_url}

        # Upgrade
        result_up = subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"], cwd=PROJECT_ROOT, capture_output=True, text=True, env=env
        )

        assert result_up.returncode == 0, f"Upgrade failed: {result_up.stderr}"

        # Downgrade
        result_down = subprocess.run(
            ["uv", "run", "alembic", "downgrade", "base"], cwd=PROJECT_ROOT, capture_output=True, text=True, env=env
        )

        assert result_down.returncode == 0, f"Downgrade failed: {result_down.stderr}"

        # Upgrade again
        result_up_again = subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"], cwd=PROJECT_ROOT, capture_output=True, text=True, env=env
        )

        assert result_up_again.returncode == 0, f"Re-upgrade failed: {result_up_again.stderr}"


class TestMigrationDataIntegrity:
    """Test data integrity and constraints after migration."""

    @pytest.mark.requires_db
    def test_agent_configs_primary_key(self, database_url: str):
        """Test that agent_configs has proper primary key."""
        # Run migration
        subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            env={**os.environ, "DATABASE_URL": database_url},
        )

        sync_url = database_url.replace("+asyncpg", "")
        engine = sa.create_engine(sync_url)
        inspector = inspect(engine)
        pk = inspector.get_pk_constraint("agent_configs")
        engine.dispose()

        assert pk is not None
        assert "id" in pk["constrained_columns"]

    @pytest.mark.requires_db
    def test_gm_policies_unique_constraint(self, database_url: str):
        """Test that gm_policies has unique constraint on tenant_id."""
        # Run migration
        subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            env={**os.environ, "DATABASE_URL": database_url},
        )

        sync_url = database_url.replace("+asyncpg", "")
        engine = sa.create_engine(sync_url)
        inspector = inspect(engine)
        constraints = inspector.get_unique_constraints("gm_policies")
        engine.dispose()

        # Check for unique constraint on tenant_id
        has_tenant_unique = any("tenant_id" in constraint["column_names"] for constraint in constraints)

        assert has_tenant_unique, "Unique constraint on tenant_id not found"

    @pytest.mark.requires_db
    def test_timestamps_are_not_null(self, database_url: str):
        """Test that timestamp columns are not null."""
        # Run migration
        subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            env={**os.environ, "DATABASE_URL": database_url},
        )

        sync_url = database_url.replace("+asyncpg", "")
        engine = sa.create_engine(sync_url)

        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT created_at, updated_at
                    FROM agent_configs
                    WHERE role_name = 'planner'
                """
                )
            )
            row = result.fetchone()

            assert row is not None
            created_at, updated_at = row

            assert created_at is not None, "created_at is null"
            assert updated_at is not None, "updated_at is null"

        engine.dispose()
