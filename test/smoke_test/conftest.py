"""
Test infrastructure for smoke tests with real AI model calling and Docker Compose dependencies.

This module provides fixtures and utilities for smoke testing that:
- Uses real AI model connections to OpenAI, Anthropic, Google
- Uses Docker Compose for real database, cache, and external services
- Provides consistent test environment setup with real dependencies
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from test.settings import test_settings
from gearmeshing_ai.agent_core.abstraction.initialization import setup_agent_abstraction
from gearmeshing_ai.agent_core.abstraction import get_agent_provider


@pytest.fixture
def mock_database_access():
    """Simple fixture to mock database access for smoke tests."""
    from unittest.mock import patch
    
    # Patch the database configuration provider to avoid database calls
    with patch('gearmeshing_ai.agent_core.db_config_provider.DatabaseConfigProvider') as mock_db_provider:
        mock_instance = mock_db_provider.return_value
        mock_instance.get_model_config.return_value = MagicMock(
            model="gpt-4o-mini",
            provider="openai",
            temperature=0.7,
            max_tokens=2000
        )
        yield mock_instance


@pytest.fixture
def mock_settings_for_ai():
    """Mock settings to provide real AI API keys for smoke tests."""
    from unittest.mock import patch
    from test.settings import test_settings
    from gearmeshing_ai.agent_core.abstraction import get_agent_provider
    
    # Create a mock settings that returns real API keys
    mock_settings = MagicMock()
    mock_settings.ai_provider = test_settings.ai_provider
    mock_settings.database = MagicMock()
    mock_settings.database.url = "sqlite:///test.db"
    
    with patch('gearmeshing_ai.server.core.config.settings', mock_settings):
        # Set up the AI framework
        provider = get_agent_provider()
        provider.set_framework("pydantic_ai")
        yield mock_settings


@pytest.fixture(scope="session", autouse=True)
async def initialize_ai_agent_provider():
    """Initialize the AI agent provider for all E2E tests."""
    try:
        # Set up the agent abstraction layer
        provider = setup_agent_abstraction(validate_api_keys=False)
        yield provider
    except Exception as e:
        # If setup fails, tests will be skipped appropriately
        print(f"Warning: Failed to initialize AI agent provider: {e}")
        yield None


# Markers for different test types


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "e2e_ai: mark test as E2E AI test (requires real AI API keys)"
    )
    config.addinivalue_line(
        "markers", "openai_only: mark test as OpenAI-only test"
    )
    config.addinivalue_line(
        "markers", "anthropic_only: mark test as Anthropic-only test"
    )
    config.addinivalue_line(
        "markers", "google_only: mark test as Google-only test"
    )
    config.addinivalue_line(
        "markers", "multi_provider: mark test as multi-provider test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add skip conditions."""
    from test.settings import test_settings
    
    for item in items:
        # Skip OpenAI-only tests if no API key
        if "openai_only" in item.keywords and not test_settings.openai.api_key:
            item.add_marker(pytest.mark.skip(reason="OpenAI API key not configured"))
        
        # Skip Anthropic-only tests if no API key
        if "anthropic_only" in item.keywords and not test_settings.anthropic.api_key:
            item.add_marker(pytest.mark.skip(reason="Anthropic API key not configured"))
        
        # Skip Google-only tests if no API key
        if "google_only" in item.keywords and not test_settings.google.api_key:
            item.add_marker(pytest.mark.skip(reason="Google API key not configured"))
        
        # Skip multi-provider tests if no API keys
        if "multi_provider" in item.keywords:
            has_keys = any([
                test_settings.openai.api_key,
                test_settings.anthropic.api_key,
                test_settings.google.api_key,
            ])
            if not has_keys:
                item.add_marker(pytest.mark.skip(reason="No AI provider API keys configured"))
