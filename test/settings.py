"""
Test Configuration Settings.

This module defines the test environment configuration using Pydantic's BaseSettings.
Pydantic automatically loads configuration from the test/.env file via env_file configuration.

Environment variables use double underscore (__) as delimiters for nested properties.
For example: AI_PROVIDER__OPENAI__API_KEY maps to test_settings.ai_provider.openai.api_key
"""

from pathlib import Path
from typing import Optional

from pydantic import Field, BaseModel, ConfigDict, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# =====================================================================
# LLM Provider Configuration Models (for tests)
# =====================================================================


class TestOpenAIConfig(BaseModel):
    """OpenAI API configuration for tests."""

    api_key: Optional[SecretStr] = Field(
        default=None, description="OpenAI API key for authentication"
    )
    model: str = Field(default="gpt-4o", description="Default OpenAI model to use")
    base_url: Optional[str] = Field(
        default=None, description="Custom OpenAI API base URL (optional)"
    )

    model_config = ConfigDict(strict=False)


class TestAnthropicConfig(BaseModel):
    """Anthropic API configuration for tests."""

    api_key: Optional[SecretStr] = Field(
        default=None, description="Anthropic API key for authentication"
    )
    model: str = Field(
        default="claude-3-opus-20240229", description="Default Anthropic model to use"
    )

    model_config = ConfigDict(strict=False)


class TestGoogleConfig(BaseModel):
    """Google API configuration for tests."""

    api_key: Optional[SecretStr] = Field(
        default=None, description="Google API key for authentication"
    )
    model: str = Field(default="gemini-pro", description="Default Google model to use")

    model_config = ConfigDict(strict=False)


class TestXAIConfig(BaseModel):
    """xAI (Grok) API configuration for tests."""

    api_key: Optional[SecretStr] = Field(default=None, description="xAI API key for authentication")
    model: str = Field(default="grok-2", description="Default xAI model to use")

    model_config = ConfigDict(strict=False)


class TestAIProviderConfig(BaseModel):
    """AI Provider configuration container for tests."""

    openai: TestOpenAIConfig = Field(default_factory=TestOpenAIConfig, description="OpenAI configuration")
    anthropic: TestAnthropicConfig = Field(default_factory=TestAnthropicConfig, description="Anthropic configuration")
    google: TestGoogleConfig = Field(default_factory=TestGoogleConfig, description="Google configuration")
    xai: TestXAIConfig = Field(default_factory=TestXAIConfig, description="xAI (Grok) configuration")

    model_config = ConfigDict(strict=False)


class TestPostgreSQLConfig(BaseModel):
    """PostgreSQL database configuration for tests."""

    db: str = Field(default="ai_dev", description="PostgreSQL database name for tests")
    user: str = Field(default="ai_dev", description="PostgreSQL database user")
    password: SecretStr = Field(default=SecretStr("changeme"), description="PostgreSQL database password")
    host: str = Field(default="localhost", description="PostgreSQL database host address")
    port: int = Field(default=5432, description="PostgreSQL database port number")

    model_config = ConfigDict(strict=False)


class TestDatabaseConfig(BaseModel):
    """Database configuration container for tests."""

    url: str = Field(
        default="sqlite+aiosqlite:///:memory:",
        description="Test database connection URL (defaults to in-memory SQLite)",
    )
    enable_postgres_tests: bool = Field(
        default=False,
        description="Enable PostgreSQL-based tests (requires PostgreSQL running)",
    )
    postgres: TestPostgreSQLConfig = Field(default_factory=TestPostgreSQLConfig, description="PostgreSQL configuration")

    model_config = ConfigDict(strict=False)


class TestExecutionConfig(BaseModel):
    """Test execution configuration."""

    run_eval_tests: bool = Field(
        default=False,
        description="Enable evaluation tests",
    )

    model_config = ConfigDict(strict=False)


# =====================================================================
# Main Test Settings Class
# =====================================================================


class TestSettings(BaseSettings):
    """
    Test environment settings model.

    All properties are automatically bound from environment variables and .env file
    in the test directory. Pydantic's BaseSettings handles dotenv loading automatically
    via model_config.

    Environment variables use double underscore (__) as delimiters for nested properties.
    Examples:
    - AI_PROVIDER__OPENAI__API_KEY → test_settings.ai_provider.openai.api_key
    - DATABASE__POSTGRES__DB → test_settings.database.postgres.db
    - TEST__RUN_EVAL_TESTS → test_settings.test.run_eval_tests
    """

    # =====================================================================
    # Pydantic Configuration
    # =====================================================================
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter='__',
    )

    # =====================================================================
    # AI Provider Configuration
    # =====================================================================
    ai_provider: TestAIProviderConfig = Field(
        default_factory=TestAIProviderConfig,
        description="AI provider configuration (OpenAI, Anthropic, Google, xAI)",
    )

    # =====================================================================
    # Test Execution Configuration
    # =====================================================================
    test: TestExecutionConfig = Field(
        default_factory=TestExecutionConfig,
        description="Test execution configuration",
    )

    # =====================================================================
    # Database Configuration for Tests
    # =====================================================================
    database: TestDatabaseConfig = Field(
        default_factory=TestDatabaseConfig,
        description="Database configuration (SQLite, PostgreSQL)",
    )


# Create singleton instance
test_settings = TestSettings()
