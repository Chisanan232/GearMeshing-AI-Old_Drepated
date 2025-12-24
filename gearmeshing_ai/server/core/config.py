"""
Configuration Settings.

This module defines the application configuration using Pydantic's BaseSettings.
It automatically loads all configuration from environment variables and .env file
without explicit dotenv loading.
"""

from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# =====================================================================
# LLM Provider Configuration Models
# =====================================================================


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""

    api_key: Optional[str] = Field(
        default=None, alias="OPENAI_API_KEY", description="OpenAI API key for authentication"
    )
    model: str = Field(default="gpt-4o", alias="OPENAI_MODEL", description="Default OpenAI model to use")
    base_url: Optional[str] = Field(
        default=None, alias="OPENAI_BASE_URL", description="Custom OpenAI API base URL (optional)"
    )

    model_config = {"populate_by_name": True}


class AnthropicConfig(BaseModel):
    """Anthropic API configuration."""

    api_key: Optional[str] = Field(
        default=None, alias="ANTHROPIC_API_KEY", description="Anthropic API key for authentication"
    )
    model: str = Field(
        default="claude-3-opus-20240229", alias="ANTHROPIC_MODEL", description="Default Anthropic model to use"
    )

    model_config = {"populate_by_name": True}


class GoogleConfig(BaseModel):
    """Google API configuration."""

    api_key: Optional[str] = Field(
        default=None, alias="GOOGLE_API_KEY", description="Google API key for authentication"
    )
    model: str = Field(default="gemini-pro", alias="GOOGLE_MODEL", description="Default Google model to use")

    model_config = {"populate_by_name": True}


class ClickUpConfig(BaseModel):
    """ClickUp MCP Server configuration."""

    api_token: Optional[str] = Field(
        default=None, alias="CLICKUP_API_TOKEN", description="ClickUp API token for authentication"
    )
    server_host: str = Field(
        default="0.0.0.0", alias="CLICKUP_SERVER_HOST", description="ClickUp MCP server host address"
    )
    server_port: int = Field(default=8082, alias="CLICKUP_SERVER_PORT", description="ClickUp MCP server port number")
    mcp_transport: str = Field(
        default="sse", alias="CLICKUP_MCP_TRANSPORT", description="MCP transport type (sse or stdio)"
    )

    model_config = {"populate_by_name": True}


class MCPGatewayConfig(BaseModel):
    """MCP Gateway configuration."""

    url: str = Field(default="http://mcp-gateway:4444", alias="MCP_GATEWAY_URL", description="MCP Gateway base URL")
    auth_token: Optional[str] = Field(
        default=None, alias="MCP_GATEWAY_AUTH_TOKEN", description="MCP Gateway authentication token"
    )
    db_url: str = Field(
        default="postgresql+psycopg://ai_dev:changeme@postgres:5432/ai_dev",
        alias="MCPGATEWAY_DB_URL",
        description="MCP Gateway PostgreSQL database URL",
    )
    redis_url: str = Field(
        default="redis://redis:6379/0", alias="MCPGATEWAY_REDIS_URL", description="MCP Gateway Redis connection URL"
    )
    admin_password: str = Field(
        default="adminpass", alias="MCPGATEWAY_ADMIN_PASSWORD", description="MCP Gateway admin password"
    )
    admin_email: str = Field(
        default="admin@example.com", alias="MCPGATEWAY_ADMIN_EMAIL", description="MCP Gateway admin email address"
    )
    admin_full_name: str = Field(
        default="Admin User", alias="MCPGATEWAY_ADMIN_FULL_NAME", description="MCP Gateway admin full name"
    )
    jwt_secret: str = Field(
        default="my-test-key", alias="MCPGATEWAY_JWT_SECRET", description="MCP Gateway JWT secret key for token signing"
    )

    model_config = {"populate_by_name": True}


class PostgreSQLConfig(BaseModel):
    """PostgreSQL database configuration."""

    db: str = Field(default="ai_dev", alias="POSTGRES_DB", description="PostgreSQL database name")
    user: str = Field(default="ai_dev", alias="POSTGRES_USER", description="PostgreSQL database user")
    password: str = Field(default="changeme", alias="POSTGRES_PASSWORD", description="PostgreSQL database password")
    host: str = Field(default="postgres", alias="POSTGRES_HOST", description="PostgreSQL database host address")
    port: int = Field(default=5432, alias="POSTGRES_PORT", description="PostgreSQL database port number")

    model_config = {"populate_by_name": True}


class CORSConfig(BaseModel):
    """CORS configuration."""

    origins: list[str] = Field(default=["*"], alias="CORS_ORIGINS", description="Allowed CORS origins (use * for all)")
    allow_credentials: bool = Field(
        default=True, alias="CORS_ALLOW_CREDENTIALS", description="Allow credentials in CORS requests"
    )
    allow_methods: list[str] = Field(
        default=["*"], alias="CORS_ALLOW_METHODS", description="Allowed HTTP methods (use * for all)"
    )
    allow_headers: list[str] = Field(
        default=["*"], alias="CORS_ALLOW_HEADERS", description="Allowed HTTP headers (use * for all)"
    )

    model_config = {"populate_by_name": True}


# =====================================================================
# Main Settings Class
# =====================================================================


class Settings(BaseSettings):
    """
    Application settings model.

    All properties are automatically bound from environment variables and .env file.
    Pydantic's BaseSettings handles dotenv loading automatically via model_config.
    """

    # =====================================================================
    # Pydantic Configuration
    # =====================================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # =====================================================================
    # GearMeshing-AI Server Configuration
    # =====================================================================
    server_host: str = Field(
        default="0.0.0.0",
        description="GearMeshing-AI server host address to bind to",
        alias="GEARMESHING_AI_SERVER_HOST",
    )
    server_port: int = Field(
        default=8000,
        description="GearMeshing-AI server port number",
        alias="GEARMESHING_AI_SERVER_PORT",
    )
    log_level: str = Field(
        default="INFO",
        description="GearMeshing-AI server logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        alias="GEARMESHING_AI_LOG_LEVEL",
    )

    # =====================================================================
    # Database Configuration
    # =====================================================================
    database_url: str = Field(
        default="postgresql+asyncpg://ai_dev:changeme@postgres:5432/ai_dev",
        description="Async PostgreSQL connection URL for application database",
    )

    # =====================================================================
    # Redis Configuration
    # =====================================================================
    redis_url: str = Field(
        default="redis://redis:6379/0",
        description="Redis connection URL for caching and message queue",
    )

    # =====================================================================
    # Computed Properties (Grouped Configurations)
    # =====================================================================

    @property
    def openai(self) -> OpenAIConfig:
        """Get OpenAI configuration from environment variables."""
        return OpenAIConfig.model_validate(self.model_dump(by_alias=True))

    @property
    def anthropic(self) -> AnthropicConfig:
        """Get Anthropic configuration from environment variables."""
        return AnthropicConfig.model_validate(self.model_dump(by_alias=True))

    @property
    def google(self) -> GoogleConfig:
        """Get Google configuration from environment variables."""
        return GoogleConfig.model_validate(self.model_dump(by_alias=True))

    @property
    def clickup(self) -> ClickUpConfig:
        """Get ClickUp MCP configuration from environment variables."""
        return ClickUpConfig.model_validate(self.model_dump(by_alias=True))

    @property
    def mcp_gateway(self) -> MCPGatewayConfig:
        """Get MCP Gateway configuration from environment variables."""
        return MCPGatewayConfig.model_validate(self.model_dump(by_alias=True))

    @property
    def postgres(self) -> PostgreSQLConfig:
        """Get PostgreSQL configuration from environment variables."""
        return PostgreSQLConfig.model_validate(self.model_dump(by_alias=True))

    @property
    def cors(self) -> CORSConfig:
        """Get CORS configuration from environment variables."""
        return CORSConfig.model_validate(self.model_dump(by_alias=True))


settings = Settings()
