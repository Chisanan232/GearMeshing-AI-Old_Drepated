# Test Settings Guide

This guide explains how to use the Pydantic-based test settings model (`test.settings.TestSettings`) across different test types in the GearMeshing-AI project.

## Overview

The test settings model provides a centralized, type-safe way to manage test configuration from environment variables and `.env` files. It replaces scattered `os.getenv()` calls with a structured Pydantic model that validates and provides IDE autocomplete support.

### Key Features

- **Type-Safe**: Pydantic validation ensures configuration is correct
- **Centralized**: Single source of truth for all test configuration
- **IDE Support**: Full autocomplete and type hints in IDEs
- **Environment-Driven**: All configuration via environment variables or `.env` files
- **Flexible**: Supports multiple test types (unit, integration, smoke, e2e, contract)
- **Documented**: Built-in field descriptions and defaults

## Setup

### 1. Copy the Example `.env` File

```bash
cp test/.env.example test/.env
```

### 2. Fill in Your Secrets

Edit `test/.env` and add your API keys and configuration:

```bash
# LLM Provider API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
XAI_API_KEY=...

# Database configuration
TEST_DATABASE_URL=sqlite+aiosqlite:///:memory:
ENABLE_POSTGRES_TESTS=0
```

## Usage Patterns

### Pattern 1: Root Test Conftest (All Test Types)

In `test/conftest.py`, the `test_config` fixture is available to all tests:

```python
import pytest
from test.settings import test_settings

@pytest.fixture(scope="session")
def test_config():
    """Fixture providing test configuration from Pydantic settings model."""
    return test_settings
```

Use it in any test:

```python
from test.settings import TestSettings

def test_example(test_config: TestSettings) -> None:
    """Test using test configuration."""
    # Access provider configurations
    api_key = test_config.openai.api_key
    model = test_config.openai.model
```

### Pattern 2: Unit Tests

For unit tests in `test/unit_test/`, use the `test_config` fixture:

```python
# test/unit_test/example/test_something.py
import pytest
from test.settings import TestSettings

def test_with_config(test_config: TestSettings) -> None:
    """Unit test using test configuration."""
    # Access LLM provider configs
    openai_key = test_config.openai.api_key
    anthropic_key = test_config.anthropic.api_key
```

### Pattern 3: Integration Tests

For integration tests in `test/integration_test/`, use database configuration:

```python
# test/integration_test/example/test_database.py
import pytest
from test.settings import TestSettings

def test_database_integration(test_config: TestSettings) -> None:
    """Integration test using database configuration."""
    # Use test database URL
    db_url: str = test_config.test_database_url
    
    # For PostgreSQL tests, check if enabled
    if test_config.enable_postgres_tests:
        postgres_config = test_config.postgres
        # Connect to PostgreSQL
        connection_string: str = f"postgresql://{postgres_config.user}:{postgres_config.password}@{postgres_config.host}:{postgres_config.port}/{postgres_config.db}"
```

### Pattern 4: Smoke Tests

For smoke tests in `test/smoke_test/`, the conftest already provides provider-specific fixtures:

```python
# test/smoke_test/test_example.py
import pytest
from test.settings import TestSettings, TestOpenAIConfig

def test_with_openai(openai_config: TestOpenAIConfig) -> None:
    """Smoke test using OpenAI configuration."""
    if openai_config.api_key:
        # Use OpenAI
        api_key: str = openai_config.api_key
        model: str = openai_config.model

def test_with_all_providers(test_config: TestSettings) -> None:
    """Smoke test using all provider configurations."""
    for provider_name in ["openai", "anthropic", "google", "xai"]:
        provider_config = getattr(test_config, provider_name)
        if provider_config.api_key:
            # Test with this provider
            pass
```

### Pattern 5: E2E Tests

For end-to-end tests in `test/e2e_test/`, use the test configuration:

```python
# test/e2e_test/example/test_workflow.py
import pytest
from test.settings import TestSettings

def test_end_to_end_workflow(test_config: TestSettings) -> None:
    """E2E test using test configuration."""
    # Run full end-to-end workflow
    openai_config = test_config.openai
    postgres_config = test_config.postgres
```

### Pattern 6: Contract Tests

For contract tests in `test/contract_test/`, use strict mode configuration:

```python
# test/contract_test/example/test_contract.py
import pytest
from test.settings import TestSettings

def test_contract_with_framework(test_config: TestSettings) -> None:
    """Contract test using test configuration."""
    # Run contract tests
    # Use pytest markers or environment variables for test selection
```

## Configuration Reference


### LLM Provider Configuration

#### OpenAI

```python
config = test_config.openai
# config.api_key: Optional[str]
# config.model: str (default: "gpt-4o")
# config.base_url: Optional[str]
```

#### Anthropic

```python
config = test_config.anthropic
# config.api_key: Optional[str]
# config.model: str (default: "claude-3-opus-20240229")
```

#### Google

```python
config = test_config.google
# config.api_key: Optional[str]
# config.model: str (default: "gemini-pro")
```

#### xAI (Grok)

```python
config = test_config.xai
# config.api_key: Optional[str]
# config.model: str (default: "grok-2")
```

### Database Configuration

#### Test Database

```python
# Use in-memory SQLite by default, or specify custom URL
db_url = test_config.test_database_url
# Default: "sqlite+aiosqlite:///:memory:"
```

#### PostgreSQL Configuration

```python
postgres_config = test_config.postgres
# postgres_config.db: str (default: "ai_dev_test")
# postgres_config.user: str (default: "ai_dev")
# postgres_config.password: str (default: "changeme")
# postgres_config.host: str (default: "localhost")
# postgres_config.port: int (default: 5432)
```

## Common Usage Examples

### Example 1: Skip Test if API Key Missing

```python
from test.settings import TestSettings

def test_requires_openai(test_config: TestSettings) -> None:
    """Test that requires OpenAI API key."""
    if not test_config.openai.api_key:
        pytest.skip("OpenAI API key not configured")
    
    # Use OpenAI
    api_key: str = test_config.openai.api_key
```

### Example 2: Parametrize Tests Across Providers

```python
from test.settings import TestSettings

@pytest.mark.parametrize("provider_name", ["openai", "anthropic", "google", "xai"])
def test_with_provider(test_config: TestSettings, provider_name: str) -> None:
    """Test with multiple providers."""
    provider_config = getattr(test_config, provider_name)
    
    if not provider_config.api_key:
        pytest.skip(f"{provider_name} API key not configured")
    
    # Test with provider
    assert provider_config.model is not None
```

### Example 3: Conditional Database Tests

```python
from test.settings import TestSettings

def test_with_postgres(test_config: TestSettings) -> None:
    """Test using PostgreSQL."""
    if not test_config.enable_postgres_tests:
        pytest.skip("PostgreSQL tests disabled")
    
    postgres_config = test_config.postgres
    db_url: str = f"postgresql://{postgres_config.user}:{postgres_config.password}@{postgres_config.host}:{postgres_config.port}/{postgres_config.db}"
    # Connect and test
```

## Environment Variable Reference

### Setting Environment Variables

#### Option 1: `.env` File (Recommended)

Create or edit `test/.env`:

```bash
# LLM API Keys
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-opus-20240229
GOOGLE_API_KEY=...
GOOGLE_MODEL=gemini-pro
XAI_API_KEY=...
XAI_MODEL=grok-2

# Database
TEST_DATABASE_URL=sqlite+aiosqlite:///:memory:
ENABLE_POSTGRES_TESTS=0
POSTGRES_DB=ai_dev_test
POSTGRES_USER=ai_dev
POSTGRES_PASSWORD=changeme
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

#### Option 2: Shell Environment

```bash
export OPENAI_API_KEY=sk-...
pytest test/unit_test/
```

#### Option 3: pytest Command Line

```bash
pytest test/unit_test/ -o env=OPENAI_API_KEY=sk-...
```

## Best Practices

1. **Never Commit Secrets**: Add `test/.env` to `.gitignore` (already done)
2. **Use `.env.example`**: Keep `test/.env.example` updated with all available settings
3. **Type Hints**: Use the settings model for IDE autocomplete and type checking
4. **Conditional Tests**: Use `pytest.skip()` for tests that require optional configuration
5. **Parametrize**: Use `@pytest.mark.parametrize` to test across multiple providers
6. **Document Defaults**: Check `test/settings.py` for default values
7. **Validate Early**: Let Pydantic validate configuration at test startup

## Troubleshooting

### Issue: Settings Not Loading

**Problem**: Environment variables not being read

**Solution**: Ensure `test/.env` exists and is in the correct location:

```bash
ls -la test/.env
```

### Issue: API Key is None

**Problem**: API key is None even though it's set in `.env`

**Solution**: Check the environment variable name matches exactly:

```bash
# In test/.env
OPENAI_API_KEY=sk-...

# In code
test_config.openai.api_key  # Should not be None
```

### Issue: Type Errors in IDE

**Problem**: IDE shows type errors for test_config

**Solution**: Ensure the import is correct:

```python
from test.settings import test_settings, TestSettings
```

## Migration Guide

### From `os.getenv()` to `test_settings`

**Before:**

```python
import os
from typing import Optional

def test_example() -> None:
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    if not api_key:
        pytest.skip("OpenAI API key not configured")
```

**After:**

```python
from test.settings import TestSettings

def test_example(test_config: TestSettings) -> None:
    api_key = test_config.openai.api_key
    model = test_config.openai.model
    if not api_key:
        pytest.skip("OpenAI API key not configured")
```

Test execution control (eval tests, e2e tests, contract tests) should be managed via pytest markers or environment-specific test selection rather than settings flags.

## See Also

- `test/settings.py` - Settings model implementation
- `test/.env.example` - Example configuration file
- `test/conftest.py` - Root test configuration
- `test/smoke_test/conftest.py` - Smoke test configuration with provider fixtures
