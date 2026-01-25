"""
Test Configuration Settings.

This module defines the test environment configuration using Pydantic's BaseSettings.
Pydantic automatically loads configuration from the test/.env file via env_file configuration.

Environment variables use double underscore (__) as delimiters for nested properties.
For example: AI_PROVIDER__OPENAI__API_KEY maps to test_settings.ai_provider.openai.api_key
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import SettingsConfigDict

from gearmeshing_ai.agent_core.abstraction.provider_env_standards import (
    export_all_provider_env_vars_from_settings,
)
from gearmeshing_ai.server.core.config import (
    BaseAISetting,
    MCPConfig,
    PostgreSQLConfig,
)


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
    postgres: PostgreSQLConfig = Field(default_factory=PostgreSQLConfig, description="PostgreSQL configuration")

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


class TestSettings(BaseAISetting):
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
        env_nested_delimiter="__",
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

    # =====================================================================
    # MCP Configuration for Tests
    # =====================================================================
    mcp: MCPConfig = Field(
        default_factory=MCPConfig,
        description="MCP configuration (Slack, ClickUp, GitHub, Gateway)",
    )


_test_settings_instance: Optional[TestSettings] = None
_test_export_completed: bool = False


def get_test_settings() -> TestSettings:
    """
    Get the test settings instance.

    This function initializes the TestSettings model and automatically exports all AI provider
    API keys from the test settings to official environment variables (e.g., OPENAI_API_KEY).
    This makes the API keys accessible to external libraries via os.getenv() for smoke tests
    that call real AI models.

    The export happens after settings initialization to ensure all configuration values
    are loaded from the test/.env file before being exported to environment variables.

    Returns:
        TestSettings: The initialized test settings instance with exported environment variables.

    Example:
        ```python
        from test.settings import get_test_settings

        test_settings = get_test_settings()
        # Now OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY are available via os.getenv()
        # for smoke tests calling real AI models
        ```
    """
    global _test_settings_instance, _test_export_completed

    if _test_settings_instance is None:
        _test_settings_instance = TestSettings()

    # Export AI provider API keys only once after settings initialization
    # Pass the settings instance to avoid circular imports
    if not _test_export_completed:
        export_all_provider_env_vars_from_settings(_test_settings_instance)
        _test_export_completed = True

    return _test_settings_instance


# Create singleton instance with exported environment variables
test_settings = get_test_settings()
