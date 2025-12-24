"""Unit tests for server configuration settings model.

Tests verify that the Settings model correctly binds environment variables
from the .env.example file and that all configuration models work as expected.
"""

import os
from pathlib import Path

import pytest

from gearmeshing_ai.server.core.config import (
    AnthropicConfig,
    ClickUpConfig,
    CORSConfig,
    GoogleConfig,
    MCPGatewayConfig,
    OpenAIConfig,
    PostgreSQLConfig,
    Settings,
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


class TestSettingsBinding:
    """Test Settings model environment variable binding."""

    def test_server_host_binding(self, env_example_vars: dict[str, str], monkeypatch):
        """Test GEARMESHING_AI_SERVER_HOST binding."""
        host = env_example_vars.get("GEARMESHING_AI_SERVER_HOST", "0.0.0.0")
        monkeypatch.setenv("GEARMESHING_AI_SERVER_HOST", host)
        
        settings = Settings()
        assert settings.server_host == host

    def test_server_port_binding(self, env_example_vars: dict[str, str], monkeypatch):
        """Test GEARMESHING_AI_SERVER_PORT binding."""
        port = env_example_vars.get("GEARMESHING_AI_SERVER_PORT", "8000")
        monkeypatch.setenv("GEARMESHING_AI_SERVER_PORT", port)
        
        settings = Settings()
        assert settings.server_port == int(port)

    def test_log_level_binding(self, env_example_vars: dict[str, str], monkeypatch):
        """Test GEARMESHING_AI_LOG_LEVEL binding."""
        log_level = env_example_vars.get("GEARMESHING_AI_LOG_LEVEL", "INFO")
        monkeypatch.setenv("GEARMESHING_AI_LOG_LEVEL", log_level)
        
        settings = Settings()
        assert settings.log_level.upper() == log_level.upper()

    def test_database_url_binding(self):
        """Test DATABASE_URL binding - verify it loads from environment."""
        settings = Settings()
        # Should load the async version from .env or default
        assert "postgresql" in settings.database_url
        assert "ai_dev" in settings.database_url

    def test_redis_url_binding(self):
        """Test REDIS_URL binding - verify it loads from environment."""
        settings = Settings()
        # Should load from .env or default
        assert "redis://" in settings.redis_url


class TestOpenAIConfigBinding:
    """Test OpenAI configuration binding."""

    def test_openai_api_key_binding(self):
        """Test OPENAI_API_KEY binding."""
        api_key = "sk-test-key"
        openai_config = OpenAIConfig.model_validate({"OPENAI_API_KEY": api_key})
        
        assert isinstance(openai_config, OpenAIConfig)
        assert openai_config.api_key == api_key

    def test_openai_model_binding(self):
        """Test OPENAI_MODEL binding."""
        model = "gpt-4o"
        openai_config = OpenAIConfig.model_validate({"OPENAI_MODEL": model})
        
        assert openai_config.model == model

    def test_openai_base_url_binding(self):
        """Test OPENAI_BASE_URL binding."""
        base_url = "https://custom.openai.com/v1"
        openai_config = OpenAIConfig.model_validate({"OPENAI_BASE_URL": base_url})
        
        assert openai_config.base_url == base_url


class TestAnthropicConfigBinding:
    """Test Anthropic configuration binding."""

    def test_anthropic_api_key_binding(self):
        """Test ANTHROPIC_API_KEY binding."""
        api_key = "sk-ant-test-key"
        anthropic_config = AnthropicConfig.model_validate({"ANTHROPIC_API_KEY": api_key})
        
        assert isinstance(anthropic_config, AnthropicConfig)
        assert anthropic_config.api_key == api_key

    def test_anthropic_model_binding(self):
        """Test ANTHROPIC_MODEL binding."""
        model = "claude-3-opus-20240229"
        anthropic_config = AnthropicConfig.model_validate({"ANTHROPIC_MODEL": model})
        
        assert anthropic_config.model == model


class TestGoogleConfigBinding:
    """Test Google configuration binding."""

    def test_google_api_key_binding(self):
        """Test GOOGLE_API_KEY binding."""
        api_key = "test-google-key"
        google_config = GoogleConfig.model_validate({"GOOGLE_API_KEY": api_key})
        
        assert isinstance(google_config, GoogleConfig)
        assert google_config.api_key == api_key

    def test_google_model_binding(self):
        """Test GOOGLE_MODEL binding."""
        model = "gemini-pro"
        google_config = GoogleConfig.model_validate({"GOOGLE_MODEL": model})
        
        assert google_config.model == model


class TestClickUpConfigBinding:
    """Test ClickUp configuration binding."""

    def test_clickup_api_token_binding(self):
        """Test CLICKUP_API_TOKEN binding."""
        api_token = "test-token"
        clickup_config = ClickUpConfig.model_validate({"CLICKUP_API_TOKEN": api_token})
        
        assert isinstance(clickup_config, ClickUpConfig)
        assert clickup_config.api_token == api_token

    def test_clickup_server_host_binding(self, env_example_vars: dict[str, str], monkeypatch):
        """Test CLICKUP_SERVER_HOST binding."""
        host = env_example_vars.get("CLICKUP_SERVER_HOST", "0.0.0.0")
        monkeypatch.setenv("CLICKUP_SERVER_HOST", host)
        
        settings = Settings()
        clickup_config = settings.clickup
        
        assert clickup_config.server_host == host

    def test_clickup_server_port_binding(self, env_example_vars: dict[str, str], monkeypatch):
        """Test CLICKUP_SERVER_PORT binding."""
        port = env_example_vars.get("CLICKUP_SERVER_PORT", "8082")
        monkeypatch.setenv("CLICKUP_SERVER_PORT", port)
        
        settings = Settings()
        clickup_config = settings.clickup
        
        assert clickup_config.server_port == int(port)

    def test_clickup_mcp_transport_binding(self, env_example_vars: dict[str, str], monkeypatch):
        """Test CLICKUP_MCP_TRANSPORT binding."""
        transport = env_example_vars.get("CLICKUP_MCP_TRANSPORT", "sse")
        monkeypatch.setenv("CLICKUP_MCP_TRANSPORT", transport)
        
        settings = Settings()
        clickup_config = settings.clickup
        
        assert clickup_config.mcp_transport == transport


class TestMCPGatewayConfigBinding:
    """Test MCP Gateway configuration binding."""

    def test_mcp_gateway_url_binding(self):
        """Test MCP_GATEWAY_URL binding."""
        url = "http://mcp-gateway:4444"
        gateway_config = MCPGatewayConfig.model_validate({"MCP_GATEWAY_URL": url})
        
        assert isinstance(gateway_config, MCPGatewayConfig)
        assert gateway_config.url == url

    def test_mcp_gateway_auth_token_binding(self):
        """Test MCP_GATEWAY_AUTH_TOKEN binding."""
        token = "test-gateway-token"
        gateway_config = MCPGatewayConfig.model_validate({"MCP_GATEWAY_AUTH_TOKEN": token})
        
        assert gateway_config.auth_token == token

    def test_mcpgateway_db_url_binding(self):
        """Test MCPGATEWAY_DB_URL binding."""
        db_url = "postgresql+psycopg://ai_dev:changeme@postgres:5432/ai_dev"
        gateway_config = MCPGatewayConfig.model_validate({"MCPGATEWAY_DB_URL": db_url})
        assert gateway_config.db_url == db_url

    def test_mcpgateway_redis_url_binding(self):
        """Test MCPGATEWAY_REDIS_URL binding."""
        redis_url = "redis://redis:6379/0"
        gateway_config = MCPGatewayConfig.model_validate({"MCPGATEWAY_REDIS_URL": redis_url})
        
        assert gateway_config.redis_url == redis_url

    def test_mcpgateway_admin_credentials_binding(self):
        """Test MCP Gateway admin credentials binding."""
        email = "admin@example.com"
        password = "changeme-admin"
        full_name = "Platform Administrator"
        
        gateway_config = MCPGatewayConfig.model_validate({
            "MCPGATEWAY_ADMIN_EMAIL": email,
            "MCPGATEWAY_ADMIN_PASSWORD": password,
            "MCPGATEWAY_ADMIN_FULL_NAME": full_name,
        })
        
        assert gateway_config.admin_email == email
        assert gateway_config.admin_password == password
        assert gateway_config.admin_full_name == full_name

    def test_mcpgateway_jwt_secret_binding(self):
        """Test MCPGATEWAY_JWT_SECRET binding."""
        secret = "super-secret-jwt-key"
        gateway_config = MCPGatewayConfig.model_validate({"MCPGATEWAY_JWT_SECRET": secret})
        
        assert gateway_config.jwt_secret == secret


class TestPostgreSQLConfigBinding:
    """Test PostgreSQL configuration binding."""

    def test_postgres_credentials_binding(self, env_example_vars: dict[str, str], monkeypatch):
        """Test PostgreSQL credentials binding."""
        db = env_example_vars.get("POSTGRES_DB", "ai_dev")
        user = env_example_vars.get("POSTGRES_USER", "ai_dev")
        password = env_example_vars.get("POSTGRES_PASSWORD", "changeme")
        
        monkeypatch.setenv("POSTGRES_DB", db)
        monkeypatch.setenv("POSTGRES_USER", user)
        monkeypatch.setenv("POSTGRES_PASSWORD", password)
        
        settings = Settings()
        postgres_config = settings.postgres
        
        assert isinstance(postgres_config, PostgreSQLConfig)
        assert postgres_config.db == db
        assert postgres_config.user == user
        assert postgres_config.password == password

    def test_postgres_host_port_binding(self, monkeypatch):
        """Test PostgreSQL host and port binding."""
        host = "postgres"
        port = "5432"
        
        monkeypatch.setenv("POSTGRES_HOST", host)
        monkeypatch.setenv("POSTGRES_PORT", port)
        
        settings = Settings()
        postgres_config = settings.postgres
        
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


class TestSettingsDefaults:
    """Test Settings model default values."""

    def test_server_defaults(self):
        """Test server configuration defaults."""
        settings = Settings()
        
        assert settings.server_host == "0.0.0.0"
        assert settings.server_port == 8000
        # Log level should be set from environment
        assert settings.log_level is not None

    def test_database_defaults(self):
        """Test database configuration defaults."""
        settings = Settings()
        
        assert "postgresql+asyncpg" in settings.database_url
        assert "ai_dev" in settings.database_url

    def test_redis_defaults(self):
        """Test Redis configuration defaults."""
        settings = Settings()
        
        assert settings.redis_url == "redis://redis:6379/0"

    def test_openai_defaults(self):
        """Test OpenAI configuration defaults."""
        settings = Settings()
        openai_config = settings.openai
        
        assert openai_config.model == "gpt-4o"
        assert openai_config.api_key is None

    def test_anthropic_defaults(self):
        """Test Anthropic configuration defaults."""
        settings = Settings()
        anthropic_config = settings.anthropic
        
        assert anthropic_config.model == "claude-3-opus-20240229"
        assert anthropic_config.api_key is None

    def test_google_defaults(self):
        """Test Google configuration defaults."""
        settings = Settings()
        google_config = settings.google
        
        assert google_config.model == "gemini-pro"
        assert google_config.api_key is None

    def test_clickup_defaults(self):
        """Test ClickUp configuration defaults."""
        settings = Settings()
        clickup_config = settings.clickup
        
        assert clickup_config.server_host == "0.0.0.0"
        assert clickup_config.server_port == 8082
        assert clickup_config.mcp_transport == "sse"

    def test_mcp_gateway_defaults(self):
        """Test MCP Gateway configuration defaults."""
        settings = Settings()
        gateway_config = settings.mcp_gateway
        
        assert gateway_config.url == "http://mcp-gateway:4444"
        assert gateway_config.admin_email == "admin@example.com"

    def test_postgres_defaults(self):
        """Test PostgreSQL configuration defaults."""
        settings = Settings()
        postgres_config = settings.postgres
        
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
