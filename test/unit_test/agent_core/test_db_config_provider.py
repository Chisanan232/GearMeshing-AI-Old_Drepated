"""
Unit tests for DatabaseConfigProvider persistence layer.

Tests the database configuration provider that loads agent configurations
from the database and converts them to domain models.
"""

from __future__ import annotations


import pytest
from sqlmodel import Session, create_engine

from gearmeshing_ai.agent_core.db_config_provider import (
    DatabaseConfigProvider,
    get_db_config_provider,
)
from gearmeshing_ai.agent_core.schemas.config import ModelConfig, RoleConfig
from gearmeshing_ai.server.models.agent_config import AgentConfig


@pytest.fixture
def in_memory_db() -> Session:
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    AgentConfig.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def sample_configs(in_memory_db: Session) -> None:
    """Populate database with sample configurations."""
    configs = [
        AgentConfig(
            role_name="dev",
            display_name="Developer",
            description="AI developer assistant",
            system_prompt_key="dev_system_prompt",
            model_provider="openai",
            model_name="gpt-4o",
            temperature=0.7,
            max_tokens=4096,
            top_p=0.9,
            capabilities='["code_generation", "debugging"]',
            tools='["file_editor", "terminal"]',
            autonomy_profiles='["supervised"]',
            done_when="Code review passed",
            is_active=True,
            tenant_id=None,
        ),
        AgentConfig(
            role_name="qa",
            display_name="QA Engineer",
            description="AI QA assistant",
            system_prompt_key="qa_system_prompt",
            model_provider="anthropic",
            model_name="claude-3-5-sonnet",
            temperature=0.5,
            max_tokens=2048,
            top_p=0.8,
            capabilities='["test_generation", "bug_detection"]',
            tools='["test_runner", "bug_tracker"]',
            autonomy_profiles='["semi_autonomous"]',
            done_when="All tests passed",
            is_active=True,
            tenant_id=None,
        ),
        AgentConfig(
            role_name="dev",
            display_name="Developer (Acme Corp)",
            description="Tenant-specific developer assistant",
            system_prompt_key="dev_system_prompt_acme",
            model_provider="google",
            model_name="gemini-2.0-flash",
            temperature=0.6,
            max_tokens=3000,
            top_p=0.85,
            capabilities='["code_generation"]',
            tools='["file_editor"]',
            autonomy_profiles='["supervised"]',
            done_when="Code review passed",
            is_active=True,
            tenant_id="acme-corp",
        ),
        AgentConfig(
            role_name="planner",
            display_name="Planner",
            description="AI planning agent",
            system_prompt_key="planner_system_prompt",
            model_provider="openai",
            model_name="gpt-4o",
            temperature=0.3,
            max_tokens=1024,
            top_p=0.7,
            capabilities='["planning", "task_decomposition"]',
            tools='["task_manager"]',
            autonomy_profiles='["autonomous"]',
            done_when="Plan created",
            is_active=False,
            tenant_id=None,
        ),
    ]
    for config in configs:
        in_memory_db.add(config)
    in_memory_db.commit()


class TestDatabaseConfigProviderGetModelConfig:
    """Tests for DatabaseConfigProvider.get_model_config() method."""

    def test_get_model_config_default_role(self, in_memory_db: Session, sample_configs: None) -> None:
        """Test retrieving model config for default (non-tenant) role."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        model_config: ModelConfig = provider.get_model_config("dev")

        assert isinstance(model_config, ModelConfig)
        assert model_config.provider == "openai"
        assert model_config.model == "gpt-4o"
        assert model_config.temperature == 0.7
        assert model_config.max_tokens == 4096
        assert model_config.top_p == 0.9

    def test_get_model_config_tenant_specific(self, in_memory_db: Session, sample_configs: None) -> None:
        """Test retrieving tenant-specific model config."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        model_config: ModelConfig = provider.get_model_config("dev", tenant_id="acme-corp")

        assert model_config.provider == "google"
        assert model_config.model == "gemini-2.0-flash"
        assert model_config.temperature == 0.6

    def test_get_model_config_tenant_fallback_to_default(self, in_memory_db: Session, sample_configs: None) -> None:
        """Test that missing tenant config falls back to default."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        model_config: ModelConfig = provider.get_model_config("qa", tenant_id="unknown-tenant")

        assert model_config.provider == "anthropic"
        assert model_config.model == "claude-3-5-sonnet"

    def test_get_model_config_different_providers(self, in_memory_db: Session, sample_configs: None) -> None:
        """Test retrieving configs for different LLM providers."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        dev_config: ModelConfig = provider.get_model_config("dev")
        qa_config: ModelConfig = provider.get_model_config("qa")

        assert dev_config.provider == "openai"
        assert qa_config.provider == "anthropic"

    def test_get_model_config_not_found(self, in_memory_db: Session) -> None:
        """Test error handling for non-existent role."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        with pytest.raises(ValueError, match="Role 'nonexistent' not found"):
            provider.get_model_config("nonexistent")

    def test_get_model_config_inactive_role(self, in_memory_db: Session) -> None:
        """Test that inactive roles are not returned."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        with pytest.raises(ValueError, match="Role 'planner' not found"):
            provider.get_model_config("planner")

    def test_get_model_config_returns_domain_model(self, in_memory_db: Session, sample_configs: None) -> None:
        """Test that returned object is a ModelConfig domain model."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        model_config: ModelConfig = provider.get_model_config("dev")

        assert isinstance(model_config, ModelConfig)
        assert hasattr(model_config, "provider")
        assert hasattr(model_config, "model")
        assert hasattr(model_config, "temperature")


class TestDatabaseConfigProviderGetRoleConfig:
    """Tests for DatabaseConfigProvider.get_role_config() method."""

    def test_get_role_config_default_role(self, in_memory_db: Session, sample_configs: None) -> None:
        """Test retrieving complete role config for default role."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        role_config: RoleConfig = provider.get_role_config("dev")

        assert isinstance(role_config, RoleConfig)
        assert role_config.role_name == "dev"
        assert role_config.display_name == "Developer"
        assert role_config.description == "AI developer assistant"
        assert role_config.system_prompt_key == "dev_system_prompt"

    def test_get_role_config_includes_model_config(self, in_memory_db: Session, sample_configs: None) -> None:
        """Test that RoleConfig includes ModelConfig."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        role_config: RoleConfig = provider.get_role_config("dev")

        assert isinstance(role_config.model, ModelConfig)
        assert role_config.model.provider == "openai"
        assert role_config.model.model == "gpt-4o"

    def test_get_role_config_includes_capabilities(self, in_memory_db: Session, sample_configs: None) -> None:
        """Test that RoleConfig includes parsed capabilities."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        role_config: RoleConfig = provider.get_role_config("dev")

        assert isinstance(role_config.capabilities, list)
        assert "code_generation" in role_config.capabilities
        assert "debugging" in role_config.capabilities

    def test_get_role_config_includes_tools(self, in_memory_db: Session, sample_configs: None) -> None:
        """Test that RoleConfig includes parsed tools."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        role_config: RoleConfig = provider.get_role_config("dev")

        assert isinstance(role_config.tools, list)
        assert "file_editor" in role_config.tools
        assert "terminal" in role_config.tools

    def test_get_role_config_includes_autonomy_profiles(self, in_memory_db: Session, sample_configs: None) -> None:
        """Test that RoleConfig includes parsed autonomy profiles."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        role_config: RoleConfig = provider.get_role_config("dev")

        assert isinstance(role_config.autonomy_profiles, list)
        assert "supervised" in role_config.autonomy_profiles

    def test_get_role_config_tenant_specific(self, in_memory_db: Session, sample_configs: None) -> None:
        """Test retrieving tenant-specific role config."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        role_config: RoleConfig = provider.get_role_config("dev", tenant_id="acme-corp")

        assert role_config.display_name == "Developer (Acme Corp)"
        assert role_config.model.provider == "google"

    def test_get_role_config_tenant_fallback(self, in_memory_db: Session, sample_configs: None) -> None:
        """Test that missing tenant config falls back to default."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        role_config: RoleConfig = provider.get_role_config("qa", tenant_id="unknown-tenant")

        assert role_config.display_name == "QA Engineer"
        assert role_config.model.provider == "anthropic"

    def test_get_role_config_not_found(self, in_memory_db: Session) -> None:
        """Test error handling for non-existent role."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        with pytest.raises(ValueError, match="Role 'nonexistent' not found"):
            provider.get_role_config("nonexistent")

    def test_get_role_config_returns_domain_model(self, in_memory_db: Session, sample_configs: None) -> None:
        """Test that returned object is a RoleConfig domain model."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        role_config: RoleConfig = provider.get_role_config("dev")

        assert isinstance(role_config, RoleConfig)
        assert hasattr(role_config, "role_name")
        assert hasattr(role_config, "model")
        assert hasattr(role_config, "capabilities")


class TestDatabaseConfigProviderListRoles:
    """Tests for DatabaseConfigProvider.list_roles() method."""

    def test_list_roles_all_active(self, in_memory_db: Session, sample_configs: None) -> None:
        """Test listing all active roles."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        roles: list[str] = provider.list_roles()

        assert isinstance(roles, list)
        assert len(roles) == 2
        assert "dev" in roles
        assert "qa" in roles
        assert "planner" not in roles

    def test_list_roles_excludes_inactive(self, in_memory_db: Session, sample_configs: None) -> None:
        """Test that inactive roles are excluded."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        roles: list[str] = provider.list_roles()

        assert "planner" not in roles

    def test_list_roles_tenant_specific(self, in_memory_db: Session, sample_configs: None) -> None:
        """Test listing roles for a specific tenant."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        roles: list[str] = provider.list_roles(tenant_id="acme-corp")

        assert len(roles) == 1
        assert "dev" in roles

    def test_list_roles_no_tenant_configs(self, in_memory_db: Session, sample_configs: None) -> None:
        """Test listing roles for tenant with no specific configs."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        roles: list[str] = provider.list_roles(tenant_id="unknown-tenant")

        assert roles == []

    def test_list_roles_empty_database(self, in_memory_db: Session) -> None:
        """Test listing roles from empty database."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        roles: list[str] = provider.list_roles()

        assert roles == []

    def test_list_roles_returns_list_of_strings(self, in_memory_db: Session, sample_configs: None) -> None:
        """Test that list_roles returns list of strings."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        roles: list[str] = provider.list_roles()

        assert all(isinstance(role, str) for role in roles)


class TestDatabaseConfigProviderFactory:
    """Tests for get_db_config_provider() factory function."""

    def test_get_db_config_provider_returns_provider(self, in_memory_db: Session) -> None:
        """Test that factory function returns DatabaseConfigProvider."""
        provider: DatabaseConfigProvider = get_db_config_provider(in_memory_db)

        assert isinstance(provider, DatabaseConfigProvider)

    def test_get_db_config_provider_with_session(self, in_memory_db: Session, sample_configs: None) -> None:
        """Test that returned provider can access database."""
        provider: DatabaseConfigProvider = get_db_config_provider(in_memory_db)

        roles: list[str] = provider.list_roles()

        assert len(roles) == 2


class TestDatabaseConfigProviderIntegration:
    """Integration tests for DatabaseConfigProvider."""

    def test_get_model_config_then_role_config(self, in_memory_db: Session, sample_configs: None) -> None:
        """Test getting both model and role config for same role."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        model_config: ModelConfig = provider.get_model_config("dev")
        role_config: RoleConfig = provider.get_role_config("dev")

        assert model_config.provider == role_config.model.provider
        assert model_config.model == role_config.model.model
        assert model_config.temperature == role_config.model.temperature

    def test_multiple_roles_different_providers(self, in_memory_db: Session, sample_configs: None) -> None:
        """Test retrieving configs for multiple roles with different providers."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        dev_config: ModelConfig = provider.get_model_config("dev")
        qa_config: ModelConfig = provider.get_model_config("qa")

        assert dev_config.provider != qa_config.provider
        assert dev_config.model != qa_config.model

    def test_tenant_override_priority(self, in_memory_db: Session, sample_configs: None) -> None:
        """Test that tenant-specific configs take priority over defaults."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        default_config: ModelConfig = provider.get_model_config("dev")
        tenant_config: ModelConfig = provider.get_model_config("dev", tenant_id="acme-corp")

        assert default_config.provider != tenant_config.provider
        assert default_config.model != tenant_config.model

    def test_list_and_get_consistency(self, in_memory_db: Session, sample_configs: None) -> None:
        """Test that listed roles can be successfully retrieved."""
        provider: DatabaseConfigProvider = DatabaseConfigProvider(in_memory_db)

        roles: list[str] = provider.list_roles()

        for role in roles:
            model_config: ModelConfig = provider.get_model_config(role)
            assert model_config is not None
            assert isinstance(model_config, ModelConfig)
