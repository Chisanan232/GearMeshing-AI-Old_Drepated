"""
Test Configuration Settings.

This module defines the test environment configuration using Pydantic's BaseSettings.
It automatically loads all configuration from environment variables and .env file
in the test directory without explicit dotenv loading.
"""

from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# =====================================================================
# LLM Provider Configuration Models (for tests)
# =====================================================================


class TestOpenAIConfig(BaseModel):
    """OpenAI API configuration for tests."""

    api_key: Optional[str] = Field(
        default=None, alias="OPENAI_API_KEY", description="OpenAI API key for authentication"
    )
    model: str = Field(default="gpt-4o", alias="OPENAI_MODEL", description="Default OpenAI model to use")
    base_url: Optional[str] = Field(
        default=None, alias="OPENAI_BASE_URL", description="Custom OpenAI API base URL (optional)"
    )

    model_config = {"populate_by_name": True}


class TestAnthropicConfig(BaseModel):
    """Anthropic API configuration for tests."""

    api_key: Optional[str] = Field(
        default=None, alias="ANTHROPIC_API_KEY", description="Anthropic API key for authentication"
    )
    model: str = Field(
        default="claude-3-opus-20240229", alias="ANTHROPIC_MODEL", description="Default Anthropic model to use"
    )

    model_config = {"populate_by_name": True}


class TestGoogleConfig(BaseModel):
    """Google API configuration for tests."""

    api_key: Optional[str] = Field(
        default=None, alias="GOOGLE_API_KEY", description="Google API key for authentication"
    )
    model: str = Field(default="gemini-pro", alias="GOOGLE_MODEL", description="Default Google model to use")

    model_config = {"populate_by_name": True}


class TestXAIConfig(BaseModel):
    """xAI (Grok) API configuration for tests."""

    api_key: Optional[str] = Field(default=None, alias="XAI_API_KEY", description="xAI API key for authentication")
    model: str = Field(default="grok-2", alias="XAI_MODEL", description="Default xAI model to use")

    model_config = {"populate_by_name": True}


class TestPostgreSQLConfig(BaseModel):
    """PostgreSQL database configuration for tests."""

    db: str = Field(default="ai_dev", alias="POSTGRES_DB", description="PostgreSQL database name for tests")
    user: str = Field(default="ai_dev", alias="POSTGRES_USER", description="PostgreSQL database user")
    password: str = Field(default="changeme", alias="POSTGRES_PASSWORD", description="PostgreSQL database password")
    host: str = Field(default="localhost", alias="POSTGRES_HOST", description="PostgreSQL database host address")
    port: int = Field(default=5432, alias="POSTGRES_PORT", description="PostgreSQL database port number")

    model_config = {"populate_by_name": True}


# =====================================================================
# Main Test Settings Class
# =====================================================================


class TestSettings(BaseSettings):
    """
    Test environment settings model.

    All properties are automatically bound from environment variables and .env file
    in the test directory. Pydantic's BaseSettings handles dotenv loading automatically
    via model_config.
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
    # Test Execution Flags
    # =====================================================================
    run_eval_tests: bool = Field(
        default=False,
        description="Enable evaluation tests",
        alias="GM_RUN_EVAL_TESTS",
    )

    # =====================================================================
    # Database Configuration for Tests
    # =====================================================================
    test_database_url: str = Field(
        default="sqlite+aiosqlite:///:memory:",
        description="Test database connection URL (defaults to in-memory SQLite)",
        alias="TEST_DATABASE_URL",
    )
    enable_postgres_tests: bool = Field(
        default=False,
        description="Enable PostgreSQL-based tests (requires PostgreSQL running)",
        alias="ENABLE_POSTGRES_TESTS",
    )

    # =====================================================================
    # Computed Properties (Grouped Configurations)
    # =====================================================================

    @property
    def openai(self) -> TestOpenAIConfig:
        """Get OpenAI configuration from environment variables."""
        return TestOpenAIConfig.model_validate(self.model_dump(by_alias=True))

    @property
    def anthropic(self) -> TestAnthropicConfig:
        """Get Anthropic configuration from environment variables."""
        return TestAnthropicConfig.model_validate(self.model_dump(by_alias=True))

    @property
    def google(self) -> TestGoogleConfig:
        """Get Google configuration from environment variables."""
        return TestGoogleConfig.model_validate(self.model_dump(by_alias=True))

    @property
    def xai(self) -> TestXAIConfig:
        """Get xAI configuration from environment variables."""
        return TestXAIConfig.model_validate(self.model_dump(by_alias=True))

    @property
    def postgres(self) -> TestPostgreSQLConfig:
        """Get PostgreSQL configuration from environment variables."""
        return TestPostgreSQLConfig.model_validate(self.model_dump(by_alias=True))


test_settings = TestSettings()
