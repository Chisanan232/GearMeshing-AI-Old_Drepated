"""
Configuration Settings.

This module defines the application configuration using Pydantic's BaseSettings.
Pydantic automatically loads configuration from the .env file via env_file configuration.

Environment variables use double underscore (__) as delimiters for nested properties.
For example: AI_PROVIDER__OPENAI__API_KEY maps to settings.ai_provider.openai.api_key
"""
from pathlib import Path
from typing import Optional

from pydantic import Field, BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_env_file_path() -> str:
    """
    Get the path to the .env file.
    
    This function is extracted to allow tests to mock it and point to a temporary
    .env file for testing environment variable binding without being affected by
    the local .env file.
    
    Returns:
        str: Path to the .env file as a string.
    """
    return str(Path(".env"))

# =====================================================================
# AI Provider Configuration Models
# =====================================================================


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""

    api_key: Optional[str] = Field(
        default=None, description="OpenAI API key for authentication"
    )
    org_id: Optional[str] = Field(
        default=None, description="OpenAI organization ID (optional)"
    )
    model: str = Field(default="gpt-4o", description="Default OpenAI model to use")
    base_url: Optional[str] = Field(
        default=None, description="Custom OpenAI API base URL (optional)"
    )

    model_config = ConfigDict(strict=False)


class AnthropicConfig(BaseModel):
    """Anthropic API configuration."""

    api_key: Optional[str] = Field(
        default=None, description="Anthropic API key for authentication"
    )
    model: str = Field(
        default="claude-3-opus-20240229", description="Default Anthropic model to use"
    )

    model_config = ConfigDict(strict=False)


class GoogleConfig(BaseModel):
    """Google API configuration."""

    api_key: Optional[str] = Field(
        default=None, description="Google API key for authentication"
    )
    project_id: Optional[str] = Field(
        default=None, description="Google Cloud project ID (optional)"
    )
    model: str = Field(default="gemini-pro", description="Default Google model to use")

    model_config = ConfigDict(strict=False)


class AIProviderConfig(BaseModel):
    """AI Provider configuration container."""

    openai: OpenAIConfig = Field(default_factory=OpenAIConfig, description="OpenAI configuration")
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig, description="Anthropic configuration")
    google: GoogleConfig = Field(default_factory=GoogleConfig, description="Google configuration")

    model_config = ConfigDict(strict=False)


# =====================================================================
# Logfire Monitoring Configuration
# =====================================================================


class LogfireConfig(BaseModel):
    """Logfire monitoring configuration."""

    enabled: bool = Field(default=False, description="Enable/disable Logfire monitoring")
    token: Optional[str] = Field(default=None, description="Logfire authentication token")
    project_name: str = Field(default="gearmeshing-ai", description="Logfire project name")
    environment: str = Field(default="development", description="Environment name (development, staging, production)")
    service_name: str = Field(default="gearmeshing-ai-server", description="Service name for identification")
    service_version: str = Field(default="0.0.0", description="Service version")
    sample_rate: float = Field(default=1.0, description="Sampling rate (0.0 to 1.0)")
    trace_sample_rate: float = Field(default=1.0, description="Trace sampling rate (0.0 to 1.0)")
    trace_pydantic_ai: bool = Field(default=True, description="Enable Pydantic AI tracing")
    trace_sqlalchemy: bool = Field(default=True, description="Enable SQLAlchemy tracing")
    trace_httpx: bool = Field(default=True, description="Enable HTTPX tracing")
    trace_fastapi: bool = Field(default=True, description="Enable FastAPI tracing")

    model_config = ConfigDict(strict=False)


# =====================================================================
# LangSmith Monitoring Configuration
# =====================================================================


class LangSmithConfig(BaseModel):
    """LangSmith monitoring configuration for LangGraph agent tracing."""

    tracing: bool = Field(default=False, description="Enable/disable LangSmith tracing")
    api_key: Optional[str] = Field(default=None, description="LangSmith API key")
    project: str = Field(default="gearmeshing-ai", description="LangSmith project name")
    endpoint: str = Field(default="https://api.smith.langchain.com", description="LangSmith API endpoint")
    workspace_id: Optional[str] = Field(default=None, description="LangSmith workspace ID (optional)")

    model_config = ConfigDict(strict=False)


# =====================================================================
# Database Configuration Models
# =====================================================================


class PostgreSQLConfig(BaseModel):
    """PostgreSQL database configuration."""

    db: str = Field(default="ai_dev", description="PostgreSQL database name")
    user: str = Field(default="ai_dev", description="PostgreSQL database user")
    password: str = Field(default="changeme", description="PostgreSQL database password")
    host: str = Field(default="postgres", description="PostgreSQL database host address")
    port: int = Field(default=5432, description="PostgreSQL database port number")

    model_config = ConfigDict(strict=False)


class DatabaseConfig(BaseModel):
    """Database configuration container."""

    postgres: PostgreSQLConfig = Field(default_factory=PostgreSQLConfig, description="PostgreSQL configuration")
    url: str = Field(
        default="postgresql://ai_dev:changeme@postgres:5432/ai_dev",
        description="Application database connection URL",
    )
    enable: bool = Field(
        default=True,
        description="Enable/disable database connectivity (true for production, false for standalone mode)",
    )

    model_config = ConfigDict(strict=False)


# =====================================================================
# MCP (Model Context Protocol) Configuration Models
# =====================================================================


class SlackMCPConfig(BaseModel):
    """Slack MCP Server configuration."""

    host: str = Field(default="0.0.0.0", description="Slack MCP server host address")
    port: int = Field(default=8081, description="Slack MCP server port number")
    mcp_transport: str = Field(default="sse", description="MCP transport type (sse or stdio)")
    bot_id: Optional[str] = Field(default=None, description="Slack bot ID")
    app_id: Optional[str] = Field(default=None, description="Slack app ID")
    bot_token: Optional[str] = Field(default=None, description="Slack bot token")
    user_token: Optional[str] = Field(default=None, description="Slack user token")
    signing_secret: Optional[str] = Field(default=None, description="Slack signing secret")

    model_config = ConfigDict(strict=False)


class ClickUpMCPConfig(BaseModel):
    """ClickUp MCP Server configuration."""

    host: str = Field(default="0.0.0.0", description="ClickUp MCP server host address")
    port: int = Field(default=8082, description="ClickUp MCP server port number")
    mcp_transport: str = Field(default="sse", description="MCP transport type (sse or stdio)")
    api_token: Optional[str] = Field(default=None, description="ClickUp API token for authentication")

    model_config = ConfigDict(strict=False)


class GitHubMCPConfig(BaseModel):
    """GitHub MCP configuration."""

    token: Optional[str] = Field(default=None, description="GitHub personal access token")
    default_repo: Optional[str] = Field(default=None, description="Default repository (org/repo format)")

    model_config = ConfigDict(strict=False)


class MCPGatewayConfig(BaseModel):
    """MCP Gateway configuration."""

    url: str = Field(default="http://mcp-gateway:4444", description="MCP Gateway base URL")
    token: Optional[str] = Field(default=None, description="MCP Gateway authentication token")
    db_url: str = Field(
        default="postgresql+psycopg://ai_dev:changeme@postgres:5432/ai_dev",
        description="MCP Gateway PostgreSQL database URL",
    )
    redis_url: str = Field(
        default="redis://redis:6379/0", description="MCP Gateway Redis connection URL"
    )
    admin_password: str = Field(
        default="adminpass", description="MCP Gateway admin password"
    )
    admin_email: str = Field(
        default="admin@example.com", description="MCP Gateway admin email address"
    )
    admin_full_name: str = Field(
        default="Admin User", description="MCP Gateway admin full name"
    )
    jwt_secret: str = Field(
        default="my-test-key", description="MCP Gateway JWT secret key for token signing"
    )

    model_config = ConfigDict(strict=False)


class MCPConfig(BaseModel):
    """MCP (Model Context Protocol) configuration container."""

    slack: SlackMCPConfig = Field(default_factory=SlackMCPConfig, description="Slack MCP configuration")
    clickup: ClickUpMCPConfig = Field(default_factory=ClickUpMCPConfig, description="ClickUp MCP configuration")
    github: GitHubMCPConfig = Field(default_factory=GitHubMCPConfig, description="GitHub MCP configuration")
    gateway: MCPGatewayConfig = Field(default_factory=MCPGatewayConfig, description="MCP Gateway configuration")

    model_config = ConfigDict(strict=False)


# =====================================================================
# Message Queue Configuration
# =====================================================================


class MessageQueueConfig(BaseModel):
    """Message Queue configuration."""

    backend: str = Field(default="kafka", description="Message queue backend (kafka, redis, etc.)")
    slack_kafka_topic: str = Field(default="slack.events", description="Slack events Kafka topic")
    clickup_kafka_topic: str = Field(default="clickup.events", description="ClickUp events Kafka topic")

    model_config = ConfigDict(strict=False)


# =====================================================================
# CORS Configuration
# =====================================================================


class CORSConfig(BaseModel):
    """CORS configuration."""

    origins: list[str] = Field(default=["*"], description="Allowed CORS origins (use * for all)")
    allow_credentials: bool = Field(
        default=True, description="Allow credentials in CORS requests"
    )
    allow_methods: list[str] = Field(
        default=["*"], description="Allowed HTTP methods (use * for all)"
    )
    allow_headers: list[str] = Field(
        default=["*"], description="Allowed HTTP headers (use * for all)"
    )

    model_config = ConfigDict(strict=False)


# =====================================================================
# Main Settings Class
# =====================================================================


class Settings(BaseSettings):
    """
    Application settings model.

    All properties are automatically bound from environment variables and .env file.
    Pydantic's BaseSettings handles dotenv loading automatically via model_config.

    Environment variables use double underscore (__) as delimiters for nested properties.
    Examples:
    - AI_PROVIDER__OPENAI__API_KEY → settings.ai_provider.openai.api_key
    - DATABASE__POSTGRES__DB → settings.database.postgres.db
    - LOGFIRE__ENABLED → settings.logfire.enabled
    - MCP__CLICKUP__API_TOKEN → settings.mcp.clickup.api_token
    """

    # =====================================================================
    # Pydantic Configuration
    # =====================================================================
    model_config = SettingsConfigDict(
        env_file=get_env_file_path(),
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter='__',
        case_sensitive=False,
    )

    # =====================================================================
    # GearMeshing-AI Server Configuration
    # =====================================================================
    gearmeshing_ai_server_host: str = Field(
        default="0.0.0.0",
        description="GearMeshing-AI server host address to bind to",
    )
    gearmeshing_ai_server_port: int = Field(
        default=8000,
        description="GearMeshing-AI server port number",
    )
    gearmeshing_ai_log_level: str = Field(
        default="INFO",
        description="GearMeshing-AI server logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # =====================================================================
    # AI Provider Configuration
    # =====================================================================
    ai_provider: AIProviderConfig = Field(
        default_factory=AIProviderConfig,
        description="AI provider configuration (OpenAI, Anthropic, Google)",
    )

    # =====================================================================
    # Monitoring Configuration
    # =====================================================================
    logfire: LogfireConfig = Field(
        default_factory=LogfireConfig,
        description="Logfire monitoring configuration",
    )
    langsmith: LangSmithConfig = Field(
        default_factory=LangSmithConfig,
        description="LangSmith monitoring configuration for LangGraph agent tracing",
    )

    # =====================================================================
    # Database Configuration
    # =====================================================================
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        description="Database configuration (PostgreSQL, connection URLs)",
    )

    # =====================================================================
    # Redis Configuration
    # =====================================================================
    app_redis_url: str = Field(
        default="redis://redis:6379/1",
        description="Application Redis connection URL for caching and message queue",
    )

    # =====================================================================
    # MCP (Model Context Protocol) Configuration
    # =====================================================================
    mcp: MCPConfig = Field(
        default_factory=MCPConfig,
        description="MCP configuration (Slack, ClickUp, GitHub, Gateway)",
    )

    # =====================================================================
    # Message Queue Configuration
    # =====================================================================
    mq: MessageQueueConfig = Field(
        default_factory=MessageQueueConfig,
        description="Message queue configuration (Kafka, topics)",
    )

    # =====================================================================
    # CORS Configuration
    # =====================================================================
    cors: CORSConfig = Field(
        default_factory=CORSConfig,
        description="CORS configuration",
    )


settings = Settings()
