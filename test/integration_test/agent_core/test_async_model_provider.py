"""Tests for async model provider functions."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from gearmeshing_ai.agent_core.model_provider import (
    async_create_model_for_role,
    async_get_model_provider,
)


class TestAsyncModelProvider:
    """Tests for async model provider functions."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    async def test_async_create_model_for_role(self):
        """Test async_create_model_for_role creates model successfully."""
        with patch("gearmeshing_ai.core.database.create_engine") as mock_engine:
            with patch("gearmeshing_ai.server.core.database.create_sessionmaker") as mock_sessionmaker:
                # Mock engine and session
                mock_engine_instance = MagicMock()
                mock_engine.return_value = mock_engine_instance

                # Mock session maker
                mock_session = MagicMock()
                mock_sessionmaker_instance = MagicMock(return_value=mock_session)
                mock_sessionmaker.return_value = mock_sessionmaker_instance

                # Mock context manager for session
                mock_session.__enter__ = MagicMock(return_value=mock_session)
                mock_session.__exit__ = MagicMock(return_value=None)

                # Mock the model provider and model creation
                with patch("gearmeshing_ai.agent_core.model_provider.get_model_provider") as mock_get_provider:
                    mock_provider = MagicMock()
                    mock_model = MagicMock()
                    mock_provider.create_model_for_role.return_value = mock_model
                    mock_get_provider.return_value = mock_provider

                    # Call async function
                    result = await async_create_model_for_role("dev", tenant_id="acme-corp")

                    # Verify model was created
                    assert result is mock_model
                    mock_provider.create_model_for_role.assert_called_once_with("dev", "acme-corp")

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    async def test_async_create_model_for_role_without_tenant(self):
        """Test async_create_model_for_role without tenant_id."""
        with patch("gearmeshing_ai.core.database.create_engine") as mock_engine:
            with patch("gearmeshing_ai.server.core.database.create_sessionmaker") as mock_sessionmaker:
                mock_engine_instance = MagicMock()
                mock_engine.return_value = mock_engine_instance

                mock_session = MagicMock()
                mock_sessionmaker_instance = MagicMock(return_value=mock_session)
                mock_sessionmaker.return_value = mock_sessionmaker_instance

                mock_session.__enter__ = MagicMock(return_value=mock_session)
                mock_session.__exit__ = MagicMock(return_value=None)

                with patch("gearmeshing_ai.agent_core.model_provider.get_model_provider") as mock_get_provider:
                    mock_provider = MagicMock()
                    mock_model = MagicMock()
                    mock_provider.create_model_for_role.return_value = mock_model
                    mock_get_provider.return_value = mock_provider

                    result = await async_create_model_for_role("planner")

                    assert result is mock_model
                    mock_provider.create_model_for_role.assert_called_once_with("planner", None)

    @pytest.mark.asyncio
    async def test_async_create_model_for_role_missing_config(self):
        """Test async_create_model_for_role raises when config not found."""
        with patch("gearmeshing_ai.core.database.create_engine") as mock_engine:
            with patch("gearmeshing_ai.server.core.database.create_sessionmaker") as mock_sessionmaker:
                mock_engine_instance = MagicMock()
                mock_engine.return_value = mock_engine_instance

                mock_session = MagicMock()
                mock_sessionmaker_instance = MagicMock(return_value=mock_session)
                mock_sessionmaker.return_value = mock_sessionmaker_instance

                mock_session.__enter__ = MagicMock(return_value=mock_session)
                mock_session.__exit__ = MagicMock(return_value=None)

                with patch("gearmeshing_ai.agent_core.model_provider.get_model_provider") as mock_get_provider:
                    mock_provider = MagicMock()
                    mock_provider.create_model_for_role.side_effect = ValueError("Role not found")
                    mock_get_provider.return_value = mock_provider

                    with pytest.raises(ValueError, match="Role not found"):
                        await async_create_model_for_role("nonexistent-role")

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    async def test_async_get_model_provider(self):
        """Test async_get_model_provider alias function."""
        with patch("gearmeshing_ai.core.database.create_engine") as mock_engine:
            with patch("gearmeshing_ai.server.core.database.create_sessionmaker") as mock_sessionmaker:
                mock_engine_instance = MagicMock()
                mock_engine.return_value = mock_engine_instance

                mock_session = MagicMock()
                mock_sessionmaker_instance = MagicMock(return_value=mock_session)
                mock_sessionmaker.return_value = mock_sessionmaker_instance

                mock_session.__enter__ = MagicMock(return_value=mock_session)
                mock_session.__exit__ = MagicMock(return_value=None)

                with patch("gearmeshing_ai.agent_core.model_provider.get_model_provider") as mock_get_provider:
                    mock_provider = MagicMock()
                    mock_model = MagicMock()
                    mock_provider.create_model_for_role.return_value = mock_model
                    mock_get_provider.return_value = mock_provider

                    # Call async_get_model_provider (alias)
                    result = await async_get_model_provider("dev", tenant_id="acme-corp")

                    # Should work same as async_create_model_for_role
                    assert result is mock_model

    @pytest.mark.asyncio
    async def test_async_create_model_handles_session_cleanup(self):
        """Test async_create_model_for_role properly cleans up session."""
        with patch("sqlalchemy.create_engine") as mock_sync_engine:
            with patch("sqlmodel.Session") as mock_session_class:
                mock_engine_instance = MagicMock()
                mock_sync_engine.return_value = mock_engine_instance

                mock_session = MagicMock()
                mock_session_class.return_value = mock_session

                with patch("gearmeshing_ai.agent_core.model_provider.get_model_provider") as mock_get_provider:
                    mock_provider = MagicMock()
                    mock_model = MagicMock()
                    mock_provider.create_model_for_role.return_value = mock_model
                    mock_get_provider.return_value = mock_provider

                    await async_create_model_for_role("dev")

                    # Verify session was created and closed
                    mock_session_class.assert_called_once_with(mock_engine_instance)
                    mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_create_model_for_role_error_propagation(self):
        """Test async_create_model_for_role propagates errors correctly."""
        with patch("gearmeshing_ai.core.database.create_engine") as mock_engine:
            with patch("gearmeshing_ai.server.core.database.create_sessionmaker") as mock_sessionmaker:
                mock_engine_instance = MagicMock()
                mock_engine.return_value = mock_engine_instance

                mock_session = MagicMock()
                mock_sessionmaker_instance = MagicMock(return_value=mock_session)
                mock_sessionmaker.return_value = mock_sessionmaker_instance

                mock_session.__enter__ = MagicMock(return_value=mock_session)
                mock_session.__exit__ = MagicMock(return_value=None)

                with patch("gearmeshing_ai.agent_core.model_provider.get_model_provider") as mock_get_provider:
                    mock_provider = MagicMock()
                    mock_provider.create_model_for_role.side_effect = RuntimeError("API error")
                    mock_get_provider.return_value = mock_provider

                    with pytest.raises(RuntimeError, match="API error"):
                        await async_create_model_for_role("dev")
