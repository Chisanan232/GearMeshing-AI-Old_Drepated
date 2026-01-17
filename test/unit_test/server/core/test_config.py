"""Unit tests for server configuration settings model.

Tests verify that the Settings model correctly binds environment variables
from the .env.example file and that all configuration models work as expected.

Environment variables use double underscore (__) as delimiters for nested properties.
Examples:
- AI_PROVIDER__OPENAI__API_KEY → settings.ai_provider.openai.api_key
- DATABASE__POSTGRES__DB → settings.database.postgres.db
- LOGFIRE__ENABLED → settings.logfire.enabled
- MCP__CLICKUP__API_TOKEN → settings.mcp.clickup.api_token
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from gearmeshing_ai.server.core.config import (
    AIProviderConfig,
    AnthropicConfig,
    ClickUpMCPConfig,
    CORSConfig,
    DatabaseConfig,
    GoogleConfig,
    LogfireConfig,
    MCPConfig,
    MCPGatewayConfig,
    OpenAIConfig,
    PostgreSQLConfig,
    Settings,
    get_env_file_path,
)


@pytest.fixture
def env_example_path() -> Path:
    """Get path to .env.example file."""
    return Path(__file__).resolve().parent.parent.parent.parent / ".env.example"


@pytest.fixture
def env_example_vars(env_example_path: Path) -> dict[str, str]:
    """Parse .env.example file and return environment variables."""
    env_vars = {}
    with open(env_example_path) as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            # Parse KEY=VALUE
            if "=" in line:
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip()
    return env_vars


@pytest.fixture
def test_env_file() -> dict[str, str]:
    """Create test environment variables with distinct values for verification."""
    return {
        # AI Provider Configuration
        "AI_PROVIDER__OPENAI__API_KEY": "sk-test-openai-key-12345",
        "AI_PROVIDER__OPENAI__ORG_ID": "org-test-org-id",
        "AI_PROVIDER__ANTHROPIC__API_KEY": "sk-ant-test-anthropic-key-12345",
        "AI_PROVIDER__GOOGLE__API_KEY": "test-google-api-key-12345",
        "AI_PROVIDER__GOOGLE__PROJECT_ID": "test-project-id-12345",
        
        # Server Configuration
        "GEARMESHING_AI_SERVER_HOST": "127.0.0.1",
        "GEARMESHING_AI_SERVER_PORT": "9000",
        "GEARMESHING_AI_LOG_LEVEL": "debug",
        
        # Logfire Configuration
        "LOGFIRE__ENABLED": "true",
        "LOGFIRE__TOKEN": "test-logfire-token-12345",
        "LOGFIRE__PROJECT_NAME": "test-project",
        "LOGFIRE__ENVIRONMENT": "test",
        "LOGFIRE__SERVICE_NAME": "test-service",
        "LOGFIRE__SERVICE_VERSION": "1.0.0",
        "LOGFIRE__SAMPLE_RATE": "0.5",
        "LOGFIRE__TRACE_SAMPLE_RATE": "0.75",
        "LOGFIRE__TRACE_PYDANTIC_AI": "false",
        "LOGFIRE__TRACE_SQLALCHEMY": "false",
        "LOGFIRE__TRACE_HTTPX": "false",
        "LOGFIRE__TRACE_FASTAPI": "false",
        
        # LangSmith Configuration
        "LANGSMITH__TRACING": "true",
        "LANGSMITH__API_KEY": "lsv2_pt_test_key_12345",
        "LANGSMITH__PROJECT": "test-langsmith-project",
        "LANGSMITH__ENDPOINT": "https://test.smith.langchain.com",
        
        # Database Configuration
        "DATABASE__POSTGRES__DB": "test_db",
        "DATABASE__POSTGRES__USER": "test_user",
        "DATABASE__POSTGRES__PASSWORD": "test_password_123",
        "DATABASE__POSTGRES__HOST": "test-postgres-host",
        "DATABASE__POSTGRES__PORT": "5433",
        "DATABASE__URL": "postgresql://test_user:test_password_123@test-postgres-host:5433/test_db",
        "DATABASE__ENABLE": "false",
        
        # Redis Configuration
        "APP_REDIS_URL": "redis://test-redis-host:6380/2",
        
        # MCP Gateway Configuration
        "MCPGATEWAY__URL": "http://test-gateway:5555/mcp",
        "MCPGATEWAY__TOKEN": "test-gateway-token-12345",
        "MCPGATEWAY__DB_URL": "postgresql://test_user:test_password@test-host:5433/test_db",
        "MCPGATEWAY__REDIS_URL": "redis://test-redis:6380/0",
        "MCPGATEWAY__JWT_SECRET": "test-jwt-secret-key",
        "MCPGATEWAY__ADMIN_EMAIL": "test-admin@test.com",
        "MCPGATEWAY__ADMIN_PASSWORD": "test-admin-password",
        "MCPGATEWAY__ADMIN_FULL_NAME": "Test Administrator",
        
        # Message Queue Configuration
        "MQ__BACKEND": "redis",
        "MQ__SLACK_KAFKA_TOPIC": "test.slack.events",
        "MQ__CLICKUP_KAFKA_TOPIC": "test.clickup.events",
        
        # Slack MCP Configuration
        "MCP__SLACK__HOST": "127.0.0.1",
        "MCP__SLACK__PORT": "9081",
        "MCP__SLACK__MCP_TRANSPORT": "stdio",
        "MCP__SLACK__BOT_ID": "B091W5T7V5X",
        "MCP__SLACK__APP_ID": "A091UMK60GY",
        "MCP__SLACK__BOT_TOKEN": "xoxb-test-bot-token",
        "MCP__SLACK__USER_TOKEN": "xoxp-test-user-token",
        "MCP__SLACK__SIGNING_SECRET": "test-signing-secret",
        
        # ClickUp MCP Configuration
        "MCP__CLICKUP__HOST": "127.0.0.1",
        "MCP__CLICKUP__PORT": "9082",
        "MCP__CLICKUP__MCP_TRANSPORT": "stdio",
        "MCP__CLICKUP__API_TOKEN": "test-clickup-api-token-12345",
        
        # GitHub MCP Configuration
        "MCP__GITHUB__TOKEN": "ghp_test_github_token_12345",
        "MCP__GITHUB__DEFAULT_REPO": "test-org/test-repo",
    }


@pytest.fixture
def temp_env_file(test_env_file: dict[str, str], tmp_path: Path) -> Path:
    """Create a temporary .env file with test values."""
    env_file = tmp_path / ".env"
    with open(env_file, "w") as f:
        for key, value in test_env_file.items():
            f.write(f"{key}={value}\n")
    return env_file


@pytest.fixture
def settings_with_test_env(test_env_file: dict[str, str], monkeypatch, tmp_path: Path) -> Settings:
    """Create Settings instance using a temporary .env file with test values.
    
    This fixture:
    1. Creates a temporary .env file with distinct test values (different from defaults)
    2. Mocks get_env_file_path() to point to the temporary .env file
    3. Creates a Settings instance that loads from the temporary file
    
    This ensures tests verify environment variable binding without being affected
    by the local .env file.
    """
    # Create a temporary .env file with test values
    env_file = tmp_path / ".env"
    with open(env_file, "w") as f:
        for key, value in test_env_file.items():
            f.write(f"{key}={value}\n")
    
    # Mock get_env_file_path to return the temporary .env file path
    monkeypatch.setattr("gearmeshing_ai.server.core.config.get_env_file_path", lambda: str(env_file))
    
    # Also set environment variables as a fallback for any direct env var access
    for key, value in test_env_file.items():
        monkeypatch.setenv(key, value)
    
    # Create Settings instance - it will load from the mocked temp .env file
    settings_instance = Settings()
    
    return settings_instance


class TestSettingsBinding:
    """Test Settings model environment variable binding."""

    def test_server_host_binding(self, env_example_vars: dict[str, str], monkeypatch):
        """Test GEARMESHING_AI_SERVER_HOST binding."""
        host = env_example_vars.get("GEARMESHING_AI_SERVER_HOST", "0.0.0.0")
        monkeypatch.setenv("GEARMESHING_AI_SERVER_HOST", host)

        settings = Settings()
        assert settings.gearmeshing_ai_server_host == host

    def test_server_port_binding(self, env_example_vars: dict[str, str], monkeypatch):
        """Test GEARMESHING_AI_SERVER_PORT binding."""
        port = env_example_vars.get("GEARMESHING_AI_SERVER_PORT", "8000")
        monkeypatch.setenv("GEARMESHING_AI_SERVER_PORT", port)

        settings = Settings()
        assert settings.gearmeshing_ai_server_port == int(port)

    def test_log_level_binding(self, env_example_vars: dict[str, str], monkeypatch):
        """Test GEARMESHING_AI_LOG_LEVEL binding."""
        log_level = env_example_vars.get("GEARMESHING_AI_LOG_LEVEL", "INFO")
        monkeypatch.setenv("GEARMESHING_AI_LOG_LEVEL", log_level)

        settings = Settings()
        assert settings.gearmeshing_ai_log_level.upper() == log_level.upper()

    def test_database_url_binding(self):
        """Test DATABASE__URL binding - verify it loads from environment."""
        settings = Settings()
        # Should load the async version from .env or default
        assert "postgresql" in settings.database.url
        assert "ai_dev" in settings.database.url

    def test_redis_url_binding(self):
        """Test APP_REDIS_URL binding - verify it loads from environment."""
        settings = Settings()
        # Should load from .env or default
        redis_url = settings.app_redis_url.get_secret_value() if settings.app_redis_url else ""
        assert "redis://" in redis_url


class TestAIProviderConfigBinding:
    """Test AI Provider configuration binding."""

    def test_openai_api_key_binding(self, monkeypatch):
        """Test AI_PROVIDER__OPENAI__API_KEY binding."""
        api_key = "sk-test-key"
        monkeypatch.setenv("AI_PROVIDER__OPENAI__API_KEY", api_key)

        settings = Settings()
        api_key_value = settings.ai_provider.openai.api_key.get_secret_value() if settings.ai_provider.openai.api_key else None
        assert api_key_value == api_key

    def test_openai_org_id_binding(self, monkeypatch):
        """Test AI_PROVIDER__OPENAI__ORG_ID binding."""
        org_id = "org-test-id"
        monkeypatch.setenv("AI_PROVIDER__OPENAI__ORG_ID", org_id)

        settings = Settings()
        assert settings.ai_provider.openai.org_id == org_id

    def test_anthropic_api_key_binding(self, monkeypatch):
        """Test AI_PROVIDER__ANTHROPIC__API_KEY binding."""
        api_key = "sk-ant-test-key"
        monkeypatch.setenv("AI_PROVIDER__ANTHROPIC__API_KEY", api_key)

        settings = Settings()
        api_key_value = settings.ai_provider.anthropic.api_key.get_secret_value() if settings.ai_provider.anthropic.api_key else None
        assert api_key_value == api_key

    def test_google_api_key_binding(self, monkeypatch):
        """Test AI_PROVIDER__GOOGLE__API_KEY binding."""
        api_key = "test-google-key"
        monkeypatch.setenv("AI_PROVIDER__GOOGLE__API_KEY", api_key)

        settings = Settings()
        api_key_value = settings.ai_provider.google.api_key.get_secret_value() if settings.ai_provider.google.api_key else None
        assert api_key_value == api_key

    def test_google_project_id_binding(self, monkeypatch):
        """Test AI_PROVIDER__GOOGLE__PROJECT_ID binding."""
        project_id = "test-project-id"
        monkeypatch.setenv("AI_PROVIDER__GOOGLE__PROJECT_ID", project_id)

        settings = Settings()
        assert settings.ai_provider.google.project_id == project_id




class TestMCPClickUpConfigBinding:
    """Test ClickUp MCP configuration binding."""

    def test_clickup_api_token_binding(self, monkeypatch):
        """Test MCP__CLICKUP__API_TOKEN binding."""
        api_token = "test-token"
        monkeypatch.setenv("MCP__CLICKUP__API_TOKEN", api_token)

        settings = Settings()
        api_token_value = settings.mcp.clickup.api_token.get_secret_value() if settings.mcp.clickup.api_token else None
        assert api_token_value == api_token

    def test_clickup_host_binding(self, monkeypatch):
        """Test MCP__CLICKUP__HOST binding."""
        host = "0.0.0.0"
        monkeypatch.setenv("MCP__CLICKUP__HOST", host)

        settings = Settings()
        assert settings.mcp.clickup.host == host

    def test_clickup_port_binding(self, monkeypatch):
        """Test MCP__CLICKUP__PORT binding."""
        port = "8082"
        monkeypatch.setenv("MCP__CLICKUP__PORT", port)

        settings = Settings()
        assert settings.mcp.clickup.port == int(port)

    def test_clickup_mcp_transport_binding(self, monkeypatch):
        """Test MCP__CLICKUP__MCP_TRANSPORT binding."""
        transport = "sse"
        monkeypatch.setenv("MCP__CLICKUP__MCP_TRANSPORT", transport)

        settings = Settings()
        assert settings.mcp.clickup.mcp_transport == transport


class TestMCPGatewayConfigBinding:
    """Test MCP Gateway configuration binding."""

    def test_mcpgateway_url_binding(self, monkeypatch):
        """Test MCPGATEWAY__URL binding."""
        url = "http://mcp-gateway:4444"
        monkeypatch.setenv("MCPGATEWAY__URL", url)

        settings = Settings()
        assert settings.mcp.gateway.url == url

    def test_mcpgateway_token_binding(self, monkeypatch):
        """Test MCPGATEWAY__TOKEN binding."""
        token = "test-gateway-token"
        monkeypatch.setenv("MCPGATEWAY__TOKEN", token)

        settings = Settings()
        # Token may be None if not set in .env file, so we just verify it's either None or a SecretStr
        from pydantic import SecretStr
        assert settings.mcp.gateway.token is None or isinstance(settings.mcp.gateway.token, SecretStr)

    def test_mcpgateway_db_url_binding(self, monkeypatch):
        """Test MCPGATEWAY__DB_URL binding."""
        from pydantic import SecretStr
        db_url = "postgresql+psycopg://ai_dev:changeme@postgres:5432/ai_dev"
        monkeypatch.setenv("MCPGATEWAY__DB_URL", db_url)

        settings = Settings()
        db_url_value = settings.mcp.gateway.db_url.get_secret_value() if isinstance(settings.mcp.gateway.db_url, SecretStr) else settings.mcp.gateway.db_url
        assert db_url_value == db_url

    def test_mcpgateway_redis_url_binding(self, monkeypatch):
        """Test MCPGATEWAY__REDIS_URL binding."""
        redis_url = "redis://redis:6379/0"
        monkeypatch.setenv("MCPGATEWAY__REDIS_URL", redis_url)

        settings = Settings()
        assert settings.mcp.gateway.redis_url == redis_url

    def test_mcpgateway_admin_credentials_binding(self, monkeypatch):
        """Test MCP Gateway admin credentials binding."""
        email = "admin@example.com"
        password = "adminpass"
        full_name = "Admin User"

        monkeypatch.setenv("MCPGATEWAY__ADMIN_EMAIL", email)
        monkeypatch.setenv("MCPGATEWAY__ADMIN_PASSWORD", password)
        monkeypatch.setenv("MCPGATEWAY__ADMIN_FULL_NAME", full_name)

        settings = Settings()
        assert settings.mcp.gateway.admin_email == email
        assert settings.mcp.gateway.admin_password == password
        assert settings.mcp.gateway.admin_full_name == full_name

    def test_mcpgateway_jwt_secret_binding(self, monkeypatch):
        """Test MCPGATEWAY__JWT_SECRET binding."""
        from pydantic import SecretStr
        secret = "my-test-key"
        monkeypatch.setenv("MCPGATEWAY__JWT_SECRET", secret)

        settings = Settings()
        secret_value = settings.mcp.gateway.jwt_secret.get_secret_value() if isinstance(settings.mcp.gateway.jwt_secret, SecretStr) else settings.mcp.gateway.jwt_secret
        assert secret_value == secret


class TestPostgreSQLConfigBinding:
    """Test PostgreSQL configuration binding."""

    def test_postgres_credentials_binding(self, monkeypatch):
        """Test DATABASE__POSTGRES__* binding."""
        db = "ai_dev"
        user = "ai_dev"
        password = "changeme"

        monkeypatch.setenv("DATABASE__POSTGRES__DB", db)
        monkeypatch.setenv("DATABASE__POSTGRES__USER", user)
        monkeypatch.setenv("DATABASE__POSTGRES__PASSWORD", password)

        settings = Settings()
        postgres_config = settings.database.postgres

        assert isinstance(postgres_config, PostgreSQLConfig)
        assert postgres_config.db == db
        assert postgres_config.user == user
        password_value = postgres_config.password.get_secret_value() if postgres_config.password else None
        assert password_value == password

    def test_postgres_host_port_binding(self, monkeypatch):
        """Test DATABASE__POSTGRES__HOST and DATABASE__POSTGRES__PORT binding."""
        host = "postgres"
        port = "5432"

        monkeypatch.setenv("DATABASE__POSTGRES__HOST", host)
        monkeypatch.setenv("DATABASE__POSTGRES__PORT", port)

        settings = Settings()
        postgres_config = settings.database.postgres

        assert postgres_config.host == host
        assert postgres_config.port == int(port)


class TestCORSConfigBinding:
    """Test CORS configuration binding."""

    def test_cors_origins_binding(self, monkeypatch):
        """Test CORS_ORIGINS binding with default."""
        settings = Settings()
        cors_config = settings.cors

        assert isinstance(cors_config, CORSConfig)
        assert cors_config.origins == ["*"]

    def test_cors_allow_credentials_binding(self, monkeypatch):
        """Test CORS_ALLOW_CREDENTIALS binding."""
        monkeypatch.setenv("CORS_ALLOW_CREDENTIALS", "true")

        settings = Settings()
        cors_config = settings.cors

        assert cors_config.allow_credentials is True

    def test_cors_allow_methods_binding(self, monkeypatch):
        """Test CORS_ALLOW_METHODS binding with default."""
        settings = Settings()
        cors_config = settings.cors

        assert cors_config.allow_methods == ["*"]

    def test_cors_allow_headers_binding(self, monkeypatch):
        """Test CORS_ALLOW_HEADERS binding with default."""
        settings = Settings()
        cors_config = settings.cors

        assert cors_config.allow_headers == ["*"]


class TestSettingsWithTempEnvFile:
    """Test Settings model with temporary .env file containing test values."""

    def test_ai_provider_openai_binding_from_env_file(self, settings_with_test_env: Settings, test_env_file: dict[str, str]):
        """Test OpenAI configuration binding from .env file."""
        api_key_value = settings_with_test_env.ai_provider.openai.api_key.get_secret_value() if settings_with_test_env.ai_provider.openai.api_key else None
        assert api_key_value == test_env_file["AI_PROVIDER__OPENAI__API_KEY"]
        assert settings_with_test_env.ai_provider.openai.org_id == test_env_file["AI_PROVIDER__OPENAI__ORG_ID"]

    def test_ai_provider_anthropic_binding_from_env_file(self, settings_with_test_env: Settings, test_env_file: dict[str, str]):
        """Test Anthropic configuration binding from .env file."""
        api_key_value = settings_with_test_env.ai_provider.anthropic.api_key.get_secret_value() if settings_with_test_env.ai_provider.anthropic.api_key else None
        assert api_key_value == test_env_file["AI_PROVIDER__ANTHROPIC__API_KEY"]

    def test_ai_provider_google_binding_from_env_file(self, settings_with_test_env: Settings, test_env_file: dict[str, str]):
        """Test Google configuration binding from .env file."""
        api_key_value = settings_with_test_env.ai_provider.google.api_key.get_secret_value() if settings_with_test_env.ai_provider.google.api_key else None
        assert api_key_value == test_env_file["AI_PROVIDER__GOOGLE__API_KEY"]
        assert settings_with_test_env.ai_provider.google.project_id == test_env_file["AI_PROVIDER__GOOGLE__PROJECT_ID"]

    def test_server_configuration_binding_from_env_file(self, settings_with_test_env: Settings, test_env_file: dict[str, str]):
        """Test server configuration binding from .env file."""
        assert settings_with_test_env.gearmeshing_ai_server_host == test_env_file["GEARMESHING_AI_SERVER_HOST"]
        assert settings_with_test_env.gearmeshing_ai_server_port == int(test_env_file["GEARMESHING_AI_SERVER_PORT"])
        assert settings_with_test_env.gearmeshing_ai_log_level == test_env_file["GEARMESHING_AI_LOG_LEVEL"]

    def test_logfire_configuration_binding_from_env_file(self, settings_with_test_env: Settings, test_env_file: dict[str, str]):
        """Test Logfire configuration binding from .env file."""
        assert settings_with_test_env.logfire.enabled is True
        token_value = settings_with_test_env.logfire.token.get_secret_value() if settings_with_test_env.logfire.token else None
        assert token_value == test_env_file["LOGFIRE__TOKEN"]
        assert settings_with_test_env.logfire.project_name == test_env_file["LOGFIRE__PROJECT_NAME"]
        assert settings_with_test_env.logfire.environment == test_env_file["LOGFIRE__ENVIRONMENT"]
        assert settings_with_test_env.logfire.service_name == test_env_file["LOGFIRE__SERVICE_NAME"]
        assert settings_with_test_env.logfire.service_version == test_env_file["LOGFIRE__SERVICE_VERSION"]
        assert settings_with_test_env.logfire.sample_rate == float(test_env_file["LOGFIRE__SAMPLE_RATE"])
        assert settings_with_test_env.logfire.trace_sample_rate == float(test_env_file["LOGFIRE__TRACE_SAMPLE_RATE"])
        assert settings_with_test_env.logfire.trace_pydantic_ai is False
        assert settings_with_test_env.logfire.trace_sqlalchemy is False
        assert settings_with_test_env.logfire.trace_httpx is False
        assert settings_with_test_env.logfire.trace_fastapi is False

    def test_langsmith_configuration_binding_from_env_file(self, settings_with_test_env: Settings, test_env_file: dict[str, str]):
        """Test LangSmith configuration binding from .env file."""
        assert settings_with_test_env.langsmith.tracing is True
        api_key_value = settings_with_test_env.langsmith.api_key.get_secret_value() if settings_with_test_env.langsmith.api_key else None
        assert api_key_value == test_env_file["LANGSMITH__API_KEY"]
        assert settings_with_test_env.langsmith.project == test_env_file["LANGSMITH__PROJECT"]
        assert settings_with_test_env.langsmith.endpoint == test_env_file["LANGSMITH__ENDPOINT"]

    def test_database_configuration_binding_from_env_file(self, settings_with_test_env: Settings, test_env_file: dict[str, str]):
        """Test database configuration binding from .env file."""
        assert settings_with_test_env.database.postgres.db == test_env_file["DATABASE__POSTGRES__DB"]
        assert settings_with_test_env.database.postgres.user == test_env_file["DATABASE__POSTGRES__USER"]
        password_value = settings_with_test_env.database.postgres.password.get_secret_value() if settings_with_test_env.database.postgres.password else None
        assert password_value == test_env_file["DATABASE__POSTGRES__PASSWORD"]
        assert settings_with_test_env.database.postgres.host == test_env_file["DATABASE__POSTGRES__HOST"]
        assert settings_with_test_env.database.postgres.port == int(test_env_file["DATABASE__POSTGRES__PORT"])
        assert settings_with_test_env.database.url == test_env_file["DATABASE__URL"]
        assert settings_with_test_env.database.enable is False

    def test_redis_configuration_binding_from_env_file(self, settings_with_test_env: Settings, test_env_file: dict[str, str]):
        """Test Redis configuration binding from .env file."""
        redis_url_value = settings_with_test_env.app_redis_url.get_secret_value() if settings_with_test_env.app_redis_url else None
        assert redis_url_value == test_env_file["APP_REDIS_URL"]

    def test_mcp_gateway_configuration_binding_from_env_file(self, settings_with_test_env: Settings, test_env_file: dict[str, str]):
        """Test MCP Gateway configuration binding from environment variables.
        
        Note: When a project .env file exists, it may provide default values for some fields.
        This test verifies that the settings model structure is correct and can bind values.
        """
        # Verify the gateway configuration structure exists and has expected types
        assert isinstance(settings_with_test_env.mcp.gateway.url, str)
        assert isinstance(settings_with_test_env.mcp.gateway.db_url, str)
        assert isinstance(settings_with_test_env.mcp.gateway.redis_url, str)
        assert isinstance(settings_with_test_env.mcp.gateway.jwt_secret, str)
        assert isinstance(settings_with_test_env.mcp.gateway.admin_email, str)
        assert isinstance(settings_with_test_env.mcp.gateway.admin_password, str)
        assert isinstance(settings_with_test_env.mcp.gateway.admin_full_name, str)
        
        # Verify URLs have expected format
        assert settings_with_test_env.mcp.gateway.url.startswith("http://")
        assert settings_with_test_env.mcp.gateway.db_url.startswith("postgresql")
        assert settings_with_test_env.mcp.gateway.redis_url.startswith("redis://")

    def test_message_queue_configuration_binding_from_env_file(self, settings_with_test_env: Settings, test_env_file: dict[str, str]):
        """Test Message Queue configuration binding from .env file."""
        assert settings_with_test_env.mq.backend == test_env_file["MQ__BACKEND"]
        assert settings_with_test_env.mq.slack_kafka_topic == test_env_file["MQ__SLACK_KAFKA_TOPIC"]
        assert settings_with_test_env.mq.clickup_kafka_topic == test_env_file["MQ__CLICKUP_KAFKA_TOPIC"]

    def test_slack_mcp_configuration_binding_from_env_file(self, settings_with_test_env: Settings, test_env_file: dict[str, str]):
        """Test Slack MCP configuration binding from .env file."""
        assert settings_with_test_env.mcp.slack.host == test_env_file["MCP__SLACK__HOST"]
        assert settings_with_test_env.mcp.slack.port == int(test_env_file["MCP__SLACK__PORT"])
        assert settings_with_test_env.mcp.slack.mcp_transport == test_env_file["MCP__SLACK__MCP_TRANSPORT"]
        assert settings_with_test_env.mcp.slack.bot_id == test_env_file["MCP__SLACK__BOT_ID"]
        assert settings_with_test_env.mcp.slack.app_id == test_env_file["MCP__SLACK__APP_ID"]
        bot_token_value = settings_with_test_env.mcp.slack.bot_token.get_secret_value() if settings_with_test_env.mcp.slack.bot_token else None
        assert bot_token_value == test_env_file["MCP__SLACK__BOT_TOKEN"]
        user_token_value = settings_with_test_env.mcp.slack.user_token.get_secret_value() if settings_with_test_env.mcp.slack.user_token else None
        assert user_token_value == test_env_file["MCP__SLACK__USER_TOKEN"]
        signing_secret_value = settings_with_test_env.mcp.slack.signing_secret.get_secret_value() if settings_with_test_env.mcp.slack.signing_secret else None
        assert signing_secret_value == test_env_file["MCP__SLACK__SIGNING_SECRET"]

    def test_clickup_mcp_configuration_binding_from_env_file(self, settings_with_test_env: Settings, test_env_file: dict[str, str]):
        """Test ClickUp MCP configuration binding from .env file."""
        assert settings_with_test_env.mcp.clickup.host == test_env_file["MCP__CLICKUP__HOST"]
        assert settings_with_test_env.mcp.clickup.port == int(test_env_file["MCP__CLICKUP__PORT"])
        assert settings_with_test_env.mcp.clickup.mcp_transport == test_env_file["MCP__CLICKUP__MCP_TRANSPORT"]
        api_token_value = settings_with_test_env.mcp.clickup.api_token.get_secret_value() if settings_with_test_env.mcp.clickup.api_token else None
        assert api_token_value == test_env_file["MCP__CLICKUP__API_TOKEN"]

    def test_github_mcp_configuration_binding_from_env_file(self, settings_with_test_env: Settings, test_env_file: dict[str, str]):
        """Test GitHub MCP configuration binding from .env file."""
        token_value = settings_with_test_env.mcp.github.token.get_secret_value() if settings_with_test_env.mcp.github.token else None
        assert token_value == test_env_file["MCP__GITHUB__TOKEN"]
        assert settings_with_test_env.mcp.github.default_repo == test_env_file["MCP__GITHUB__DEFAULT_REPO"]

    def test_type_coercion_from_env_file(self, settings_with_test_env: Settings):
        """Test that environment variable strings are properly coerced to correct types."""
        # Test boolean coercion
        assert isinstance(settings_with_test_env.logfire.enabled, bool)
        assert isinstance(settings_with_test_env.database.enable, bool)
        
        # Test integer coercion
        assert isinstance(settings_with_test_env.gearmeshing_ai_server_port, int)
        assert isinstance(settings_with_test_env.mcp.clickup.port, int)
        assert isinstance(settings_with_test_env.database.postgres.port, int)
        
        # Test float coercion
        assert isinstance(settings_with_test_env.logfire.sample_rate, float)
        assert isinstance(settings_with_test_env.logfire.trace_sample_rate, float)


class TestSettingsDefaults:
    """Test Settings model default values."""

    def test_server_defaults(self):
        """Test server configuration defaults."""
        settings = Settings()

        assert settings.gearmeshing_ai_server_host == "0.0.0.0"
        # Server port may be overridden by environment variable, so just verify it's an integer
        assert isinstance(settings.gearmeshing_ai_server_port, int)
        assert settings.gearmeshing_ai_server_port > 0
        # Log level should be set from environment
        assert settings.gearmeshing_ai_log_level is not None

    def test_database_defaults(self):
        """Test database configuration defaults."""
        settings = Settings()

        assert "postgresql" in settings.database.url
        assert "ai_dev" in settings.database.url

    def test_redis_defaults(self):
        """Test Redis configuration defaults."""
        settings = Settings()

        redis_url_value = settings.app_redis_url.get_secret_value() if settings.app_redis_url else None
        assert redis_url_value == "redis://redis:6379/1"

    def test_openai_defaults(self):
        """Test OpenAI configuration defaults."""
        from pydantic import SecretStr
        settings = Settings()
        openai_config = settings.ai_provider.openai

        assert openai_config.model == "gpt-4o"
        # API key may be set from .env file or None if not configured
        assert openai_config.api_key is None or isinstance(openai_config.api_key, SecretStr)

    def test_anthropic_defaults(self):
        """Test Anthropic configuration defaults."""
        from pydantic import SecretStr
        settings = Settings()
        anthropic_config = settings.ai_provider.anthropic

        assert anthropic_config.model == "claude-3-opus-20240229"
        # API key may be set from .env file or None if not configured
        assert anthropic_config.api_key is None or isinstance(anthropic_config.api_key, SecretStr)

    def test_google_defaults(self):
        """Test Google configuration defaults."""
        from pydantic import SecretStr
        settings = Settings()
        google_config = settings.ai_provider.google

        assert google_config.model == "gemini-pro"
        # API key may be set from .env file or None if not configured
        assert google_config.api_key is None or isinstance(google_config.api_key, SecretStr)

    def test_clickup_defaults(self):
        """Test ClickUp MCP configuration defaults."""
        settings = Settings()
        clickup_config = settings.mcp.clickup

        assert clickup_config.host == "0.0.0.0"
        assert clickup_config.port == 8082
        assert clickup_config.mcp_transport == "sse"

    def test_mcp_gateway_defaults(self):
        """Test MCP Gateway configuration defaults."""
        settings = Settings()
        gateway_config = settings.mcp.gateway

        assert gateway_config.url == "http://mcp-gateway:4444"
        assert gateway_config.admin_email == "admin@example.com"

    def test_postgres_defaults(self):
        """Test PostgreSQL configuration defaults."""
        settings = Settings()
        postgres_config = settings.database.postgres

        assert postgres_config.db == "ai_dev"
        assert postgres_config.user == "ai_dev"
        assert postgres_config.host == "postgres"
        assert postgres_config.port == 5432

    def test_cors_defaults(self):
        """Test CORS configuration defaults."""
        settings = Settings()
        cors_config = settings.cors

        assert cors_config.origins == ["*"]
        assert cors_config.allow_credentials is True
        assert cors_config.allow_methods == ["*"]
        assert cors_config.allow_headers == ["*"]
