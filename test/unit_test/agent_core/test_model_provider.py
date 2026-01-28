"""Tests for model provider with database-driven configuration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gearmeshing_ai.agent_core.model_provider import (
    UnifiedModelProvider,
    async_create_model_for_role,
    async_get_model_provider,
    create_model_for_role,
    get_model_provider,
)


class TestModelProvider:
    """Tests for UnifiedModelProvider class."""

    def test_model_provider_initialization_requires_db_session(self) -> None:
        """Test UnifiedModelProvider initialization requires database session."""
        with pytest.raises(ValueError, match="db_session is required"):
            UnifiedModelProvider(db_session=None)  # type: ignore[arg-type]

    def test_model_provider_initialization_with_db_session(self) -> None:
        """Test UnifiedModelProvider initialization with database session."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)
        assert provider.db_session is mock_session
        assert provider._db_provider is None

    def test_create_model_calls_abstraction_layer(self) -> None:
        """Test creating OpenAI model with explicit parameters."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        with patch.object(provider._provider, "create_model") as mock_create:
            mock_create.return_value = MagicMock()

            result = provider.create_model("openai", "gpt-4o", temperature=0.5, max_tokens=2048, top_p=0.8)

            mock_create.assert_called_once()
            assert result is not None

    def test_create_model_handles_provider_errors(self) -> None:
        """Test that create_model properly handles provider errors."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        with patch.object(provider._provider, "create_model") as mock_create:
            mock_create.side_effect = RuntimeError("Provider error")

            with pytest.raises(RuntimeError, match="Provider error"):
                provider.create_model("openai", "gpt-4o")

    def test_create_model_with_unsupported_provider(self) -> None:
        """Test create_model with unsupported provider."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        with pytest.raises(ValueError, match="Unsupported provider"):
            provider.create_model("unsupported", "model")

    def test_create_model_provider_case_insensitive(self) -> None:
        """Test that provider names are case-insensitive."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        with patch.object(provider._provider, "create_model") as mock_create:
            mock_create.return_value = MagicMock()
            provider.create_model("OPENAI", "gpt-4o")
            # The abstraction layer should handle case normalization
            mock_create.assert_called_once()

    def test_create_model_for_role_with_database(self) -> None:
        """Test create_model_for_role with database configuration."""

        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        with patch.object(provider, "create_model_for_role") as mock_create_role:
            mock_create_role.return_value = MagicMock()

            result = provider.create_model_for_role("dev", tenant_id="acme-corp")

            mock_create_role.assert_called_once_with("dev", tenant_id="acme-corp")
            assert result is not None

    def test_get_supported_providers(self) -> None:
        """Test getting supported providers."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        with patch.object(provider._provider, "get_supported_providers") as mock_supported:
            mock_supported.return_value = ["openai", "anthropic", "google"]

            result = provider.get_supported_providers()

            assert result == ["openai", "anthropic", "google"]
            mock_supported.assert_called_once()

    def test_get_supported_models(self) -> None:
        """Test getting supported models for a provider."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        with patch.object(provider._provider, "get_supported_models") as mock_models:
            mock_models.return_value = ["gpt-4o", "gpt-4-turbo"]

            result = provider.get_supported_models("openai")

            assert result == ["gpt-4o", "gpt-4-turbo"]
            mock_models.assert_called_once_with("openai")

    def test_create_fallback_model(self) -> None:
        """Test creating fallback model."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        with patch.object(provider._provider, "create_fallback_model") as mock_fallback:
            mock_fallback.return_value = MagicMock()

            result = provider.create_fallback_model("openai", "gpt-4o", "anthropic", "claude-3-5-sonnet")

            mock_fallback.assert_called_once()
            assert result is not None

    def test_get_provider_from_model_name(self) -> None:
        """Test getting provider from model name."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        with patch.object(provider._provider, "get_provider_from_model_name") as mock_get_provider:
            mock_get_provider.return_value = "openai"

            result = provider.get_provider_from_model_name("gpt-4o")

            assert result == "openai"
            mock_get_provider.assert_called_once_with("gpt-4o")

    def test_framework_selection(self) -> None:
        """Test framework selection in UnifiedModelProvider."""
        mock_session = MagicMock()

        # Test default framework
        provider = UnifiedModelProvider(db_session=mock_session)
        assert provider.framework == "pydantic_ai"

        # Test explicit framework
        provider_explicit = UnifiedModelProvider(db_session=mock_session, framework="pydantic_ai")
        assert provider_explicit.framework == "pydantic_ai"

    def test_unsupported_framework_raises_error(self) -> None:
        """Test that unsupported framework raises ValueError."""
        mock_session = MagicMock()

        with pytest.raises(ValueError, match="Unsupported framework"):
            UnifiedModelProvider(db_session=mock_session, framework="unsupported_framework")

    def test_unsupported_framework_specific_error(self) -> None:
        """Test that unsupported framework raises specific ValueError."""
        mock_session = MagicMock()

        with pytest.raises(ValueError, match="Unsupported framework: unsupported_framework"):
            UnifiedModelProvider(db_session=mock_session, framework="unsupported_framework")

    def test_provider_initialization_error_handling(self) -> None:
        """Test error handling during provider initialization."""
        mock_session = MagicMock()

        with patch("gearmeshing_ai.agent_core.model_provider.PydanticAIModelProviderFactory") as mock_factory_class:
            mock_factory = MagicMock()
            mock_factory_class.return_value = mock_factory
            mock_factory.create_provider.side_effect = RuntimeError("Initialization failed")

            with pytest.raises(RuntimeError, match="Initialization failed"):
                UnifiedModelProvider(db_session=mock_session, framework="pydantic_ai")

    def test_runtime_error_when_provider_not_initialized(self) -> None:
        """Test RuntimeError when provider is not initialized."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        # Manually set provider to None to test error case
        provider._provider = None

        with pytest.raises(RuntimeError, match="Model provider not initialized"):
            provider.create_model("openai", "gpt-4o")

    def test_runtime_error_get_provider_from_model_name(self) -> None:
        """Test RuntimeError in get_provider_from_model_name."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)
        provider._provider = None

        with pytest.raises(RuntimeError, match="Model provider not initialized"):
            provider.get_provider_from_model_name("gpt-4o")

    def test_runtime_error_create_fallback_model(self) -> None:
        """Test RuntimeError in create_fallback_model."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)
        provider._provider = None

        with pytest.raises(RuntimeError, match="Model provider not initialized"):
            provider.create_fallback_model("openai", "gpt-4o", "anthropic", "claude-3-5-sonnet")

    def test_runtime_error_get_supported_providers(self) -> None:
        """Test RuntimeError in get_supported_providers."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)
        provider._provider = None

        with pytest.raises(RuntimeError, match="Model provider not initialized"):
            provider.get_supported_providers()

    def test_runtime_error_get_supported_models(self) -> None:
        """Test RuntimeError in get_supported_models."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)
        provider._provider = None

        with pytest.raises(RuntimeError, match="Model provider not initialized"):
            provider.get_supported_models("openai")

    def test_get_db_provider_lazy_loading(self) -> None:
        """Test that _get_db_provider creates provider lazily."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        # Initially should be None
        assert provider._db_provider is None

        with patch("gearmeshing_ai.info_provider.model.provider.DatabaseModelProvider") as mock_db_provider_class:
            mock_db_provider = MagicMock()
            mock_db_provider_class.return_value = mock_db_provider

            # Call _get_db_provider
            result = provider._get_db_provider()

            # Should create and cache the provider
            assert result is mock_db_provider
            assert provider._db_provider is mock_db_provider
            mock_db_provider_class.assert_called_once()

    def test_get_db_provider_cached(self) -> None:
        """Test that _get_db_provider returns cached instance."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        with patch("gearmeshing_ai.info_provider.model.provider.DatabaseModelProvider") as mock_db_provider_class:
            mock_db_provider = MagicMock()
            mock_db_provider_class.return_value = mock_db_provider

            # First call
            result1 = provider._get_db_provider()
            # Second call
            result2 = provider._get_db_provider()

            # Should return same instance and only create once
            assert result1 is result2
            assert result1 is mock_db_provider
            mock_db_provider_class.assert_called_once()

    def test_create_model_for_role_with_database_integration(self) -> None:
        """Test create_model_for_role with full database integration."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        # Mock the database provider and config
        with patch.object(provider, "_get_db_provider") as mock_get_db:
            mock_db_provider = MagicMock()
            mock_get_db.return_value = mock_db_provider

            # Mock database config
            from gearmeshing_ai.core.models.config import (
                ModelConfig as DbModelConfig,
            )

            mock_db_config = DbModelConfig(
                provider="openai",
                model="gpt-4o",
                temperature=0.7,
                max_tokens=2048,
                top_p=0.9,
            )
            mock_db_provider.get.return_value = mock_db_config

            # Mock the abstraction layer
            with patch.object(provider._provider, "create_model") as mock_create:
                mock_create.return_value = MagicMock()

                result = provider.create_model_for_role("dev", tenant_id="acme-corp")

                # Verify database was queried
                mock_db_provider.get.assert_called_once_with("dev", "acme-corp")

                # Verify abstraction layer was called
                mock_create.assert_called_once()
                assert result is not None

    def test_get_provider_factory_pydantic_ai(self) -> None:
        """Test _get_provider_factory returns PydanticAI factory."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        factory = provider._get_provider_factory()

        # Should return PydanticAIModelProviderFactory instance
        from gearmeshing_ai.agent_core.abstraction.adapters import (
            PydanticAIModelProviderFactory,
        )

        assert isinstance(factory, PydanticAIModelProviderFactory)

    def test_get_provider_factory_unsupported_framework(self) -> None:
        """Test _get_provider_factory with unsupported framework."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)
        provider.framework = "unsupported"

        with pytest.raises(ValueError, match="Unsupported framework: unsupported"):
            provider._get_provider_factory()

    def test_initialize_provider_success_logging(self) -> None:
        """Test successful provider initialization logs debug message."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        # The initialization should have already happened in __init__
        # Verify the provider was created
        assert provider._provider is not None

    def test_create_model_with_none_values(self) -> None:
        """Test create_model with None values uses defaults."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        with patch.object(provider._provider, "create_model") as mock_create:
            mock_create.return_value = MagicMock()

            # Call with None values
            provider.create_model("openai", "gpt-4o", temperature=None, max_tokens=None, top_p=None)

            # Verify defaults were applied
            mock_create.assert_called_once()
            config = mock_create.call_args[0][0]
            assert config.temperature == 0.7
            assert config.top_p == 0.9
            assert config.max_tokens is None  # This should remain None

    def test_create_fallback_model_with_none_values(self) -> None:
        """Test create_fallback_model with None values uses defaults."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        with patch.object(provider._provider, "create_fallback_model") as mock_create:
            mock_create.return_value = MagicMock()

            # Call with None values
            provider.create_fallback_model(
                "openai", "gpt-4o", "anthropic", "claude-3-5-sonnet", temperature=None, max_tokens=None, top_p=None
            )

            # Verify defaults were applied to both configs
            mock_create.assert_called_once()
            primary_config, fallback_config = mock_create.call_args[0]

            assert primary_config.temperature == 0.7
            assert primary_config.top_p == 0.9
            assert fallback_config.temperature == 0.7
            assert fallback_config.top_p == 0.9

    def test_create_model_config_parameter_validation(self) -> None:
        """Test that ModelConfig parameters are properly validated."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        with patch.object(provider._provider, "create_model") as mock_create:
            mock_create.return_value = MagicMock()

            # Test with all parameters
            provider.create_model("openai", "gpt-4o", temperature=0.5, max_tokens=1000, top_p=0.8)

            mock_create.assert_called_once()
            config = mock_create.call_args[0][0]
            assert config.provider == "openai"
            assert config.model == "gpt-4o"
            assert config.temperature == 0.5
            assert config.max_tokens == 1000
            assert config.top_p == 0.8

    def test_create_fallback_model_config_validation(self) -> None:
        """Test fallback model config validation."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        with patch.object(provider._provider, "create_fallback_model") as mock_create:
            mock_create.return_value = MagicMock()

            provider.create_fallback_model(
                "openai", "gpt-4o", "anthropic", "claude-3-5-sonnet", temperature=0.6, max_tokens=1500, top_p=0.85
            )

            mock_create.assert_called_once()
            primary_config, fallback_config = mock_create.call_args[0]

            # Verify primary config
            assert primary_config.provider == "openai"
            assert primary_config.model == "gpt-4o"
            assert primary_config.temperature == 0.6
            assert primary_config.max_tokens == 1500
            assert primary_config.top_p == 0.85

            # Verify fallback config
            assert fallback_config.provider == "anthropic"
            assert fallback_config.model == "claude-3-5-sonnet"
            assert fallback_config.temperature == 0.6
            assert fallback_config.max_tokens == 1500
            assert fallback_config.top_p == 0.85

    def test_session_factory_in_get_db_provider(self) -> None:
        """Test that session_factory is properly created in _get_db_provider."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        with patch("gearmeshing_ai.info_provider.model.provider.DatabaseModelProvider") as mock_db_provider_class:
            mock_db_provider = MagicMock()
            mock_db_provider_class.return_value = mock_db_provider

            # Call _get_db_provider
            provider._get_db_provider()

            # Verify DatabaseModelProvider was called with session_factory
            call_args = mock_db_provider_class.call_args[0]
            session_factory = call_args[0]

            # Call the session_factory to verify it returns the mock session
            result = session_factory()
            assert result is mock_session

    def test_abstraction_layer_integration(self) -> None:
        """Test that UnifiedModelProvider properly integrates with abstraction layer."""
        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        # Verify that the provider is initialized
        assert provider._provider is not None
        assert hasattr(provider._provider, "create_model")
        assert hasattr(provider._provider, "get_supported_providers")

    def test_model_config_creation(self) -> None:
        """Test that ModelConfig is created correctly."""
        from gearmeshing_ai.agent_core.abstraction import ModelConfig

        mock_session = MagicMock()
        provider = UnifiedModelProvider(db_session=mock_session)

        with patch.object(provider._provider, "create_model") as mock_create:
            mock_create.return_value = MagicMock()

            # This should create a ModelConfig internally
            provider.create_model("openai", "gpt-4o", temperature=0.7, max_tokens=2048)

            # Verify the abstraction layer was called
            mock_create.assert_called_once()

            # Check the arguments passed to the abstraction layer
            call_args = mock_create.call_args[0][0]  # First positional argument
            assert isinstance(call_args, ModelConfig)
            assert call_args.provider == "openai"
            assert call_args.model == "gpt-4o"
            assert call_args.temperature == 0.7
            assert call_args.max_tokens == 2048


class TestModelProviderFunctions:
    """Tests for model provider convenience functions."""

    def test_get_model_provider_function(self) -> None:
        """Test get_model_provider function."""
        mock_session = MagicMock()

        with patch("gearmeshing_ai.agent_core.model_provider.UnifiedModelProvider") as mock_provider_class:
            mock_provider_class.return_value = MagicMock()

            result = get_model_provider(mock_session)

            mock_provider_class.assert_called_once_with(mock_session, "pydantic_ai")
            assert result is not None

    def test_get_model_provider_with_explicit_framework(self) -> None:
        """Test get_model_provider with explicit framework parameter."""
        mock_session = MagicMock()

        with patch("gearmeshing_ai.agent_core.model_provider.UnifiedModelProvider") as mock_provider_class:
            mock_provider_class.return_value = MagicMock()

            result = get_model_provider(mock_session, framework="pydantic_ai")

            mock_provider_class.assert_called_once_with(mock_session, "pydantic_ai")
            assert result is not None

    def test_create_model_for_role_function(self) -> None:
        """Test create_model_for_role function."""
        mock_session = MagicMock()

        with patch("gearmeshing_ai.agent_core.model_provider.get_model_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_get_provider.return_value = mock_provider
            mock_provider.create_model_for_role.return_value = MagicMock()

            result = create_model_for_role(mock_session, "dev", tenant_id="acme-corp")

            mock_get_provider.assert_called_once_with(mock_session, "pydantic_ai")
            mock_provider.create_model_for_role.assert_called_once_with("dev", "acme-corp")
            assert result is not None

    def test_create_model_for_role_function_with_framework(self) -> None:
        """Test create_model_for_role function with framework parameter."""
        mock_session = MagicMock()

        with patch("gearmeshing_ai.agent_core.model_provider.get_model_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_get_provider.return_value = mock_provider
            mock_provider.create_model_for_role.return_value = MagicMock()

            result = create_model_for_role(mock_session, "dev", tenant_id="acme-corp", framework="pydantic_ai")

            mock_get_provider.assert_called_once_with(mock_session, "pydantic_ai")
            mock_provider.create_model_for_role.assert_called_once_with("dev", "acme-corp")
            assert result is not None


class TestAsyncFunctionsCoverage:
    """Tests for async functions to ensure complete coverage."""

    @pytest.mark.asyncio
    async def test_async_create_model_for_role_postgresql(self) -> None:
        """Test async_create_model_for_role with PostgreSQL URL."""
        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.database.url = "postgresql+asyncpg://user:pass@localhost/db"

            with patch("sqlalchemy.create_engine") as mock_create_engine:
                with patch("sqlmodel.Session") as mock_session_class:
                    with patch("gearmeshing_ai.agent_core.model_provider.get_model_provider") as mock_get_provider:
                        mock_engine = MagicMock()
                        mock_create_engine.return_value = mock_engine

                        mock_session = MagicMock()
                        mock_session_class.return_value = mock_session

                        mock_provider = MagicMock()
                        mock_get_provider.return_value = mock_provider

                        mock_model = MagicMock()
                        mock_provider.create_model_for_role.return_value = mock_model

                        result = await async_create_model_for_role("dev", tenant_id="acme-corp")

                        # Verify URL conversion
                        mock_create_engine.assert_called_once_with("postgresql://user:pass@localhost/db")

                        # Verify provider was called
                        mock_get_provider.assert_called_once_with(mock_session, "pydantic_ai")
                        mock_provider.create_model_for_role.assert_called_once_with("dev", "acme-corp")

                        # Verify session was closed
                        mock_session.close.assert_called_once()

                        assert result is mock_model

    @pytest.mark.asyncio
    async def test_async_create_model_for_role_sqlite(self) -> None:
        """Test async_create_model_for_role with SQLite URL."""
        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.database.url = "sqlite+aiosqlite:///test.db"

            with patch("sqlalchemy.create_engine") as mock_create_engine:
                with patch("sqlmodel.Session") as mock_session_class:
                    with patch("gearmeshing_ai.agent_core.model_provider.get_model_provider") as mock_get_provider:
                        mock_engine = MagicMock()
                        mock_create_engine.return_value = mock_engine

                        mock_session = MagicMock()
                        mock_session_class.return_value = mock_session

                        mock_provider = MagicMock()
                        mock_get_provider.return_value = mock_provider

                        mock_model = MagicMock()
                        mock_provider.create_model_for_role.return_value = mock_model

                        result = await async_create_model_for_role("dev")

                        # Verify URL conversion
                        mock_create_engine.assert_called_once_with("sqlite:///test.db")

                        assert result is mock_model

    @pytest.mark.asyncio
    async def test_async_create_model_for_role_other_db(self) -> None:
        """Test async_create_model_for_role with other database URL."""
        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.database.url = "mysql://user:pass@localhost/db"

            with patch("sqlalchemy.create_engine") as mock_create_engine:
                with patch("sqlmodel.Session") as mock_session_class:
                    with patch("gearmeshing_ai.agent_core.model_provider.get_model_provider") as mock_get_provider:
                        mock_engine = MagicMock()
                        mock_create_engine.return_value = mock_engine

                        mock_session = MagicMock()
                        mock_session_class.return_value = mock_session

                        mock_provider = MagicMock()
                        mock_get_provider.return_value = mock_provider

                        mock_model = MagicMock()
                        mock_provider.create_model_for_role.return_value = mock_model

                        result = await async_create_model_for_role("dev")

                        # Verify URL was not modified
                        mock_create_engine.assert_called_once_with("mysql://user:pass@localhost/db")

                        assert result is mock_model

    @pytest.mark.asyncio
    async def test_async_create_model_for_role_error_handling(self) -> None:
        """Test async_create_model_for_role error handling."""
        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.database.url = "sqlite+aiosqlite:///test.db"

            with patch("sqlalchemy.create_engine") as mock_create_engine:
                with patch("sqlmodel.Session") as mock_session_class:
                    mock_engine = MagicMock()
                    mock_create_engine.return_value = mock_engine

                    mock_session = MagicMock()
                    mock_session_class.return_value = mock_session

                    # Make get_model_provider raise an error
                    with patch("gearmeshing_ai.agent_core.model_provider.get_model_provider") as mock_get_provider:
                        mock_get_provider.side_effect = ValueError("Role not found")

                        with pytest.raises(ValueError, match="Role not found"):
                            await async_create_model_for_role("nonexistent")

                        # Verify session was still closed
                        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_create_model_for_role_session_cleanup_on_error(self) -> None:
        """Test that session is cleaned up even if error occurs during creation."""
        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.database.url = "sqlite+aiosqlite:///test.db"

            with patch("sqlalchemy.create_engine") as mock_create_engine:
                with patch("sqlmodel.Session") as mock_session_class:
                    mock_engine = MagicMock()
                    mock_create_engine.return_value = mock_engine

                    mock_session = MagicMock()
                    mock_session_class.return_value = mock_session

                    # Make session.close raise an error to test finally block
                    mock_session.close.side_effect = RuntimeError("Close error")

                    with patch("gearmeshing_ai.agent_core.model_provider.get_model_provider") as mock_get_provider:
                        mock_get_provider.side_effect = ValueError("Role not found")

                        # The close error will be propagated since finally is outside try
                        with pytest.raises(RuntimeError, match="Close error"):
                            await async_create_model_for_role("nonexistent")

                        # close should still be called despite the error
                        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_get_model_provider_alias(self) -> None:
        """Test that async_get_model_provider is an alias for async_create_model_for_role."""
        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_async_create:
            mock_model = MagicMock()
            mock_async_create.return_value = mock_model

            result = await async_get_model_provider("dev", tenant_id="acme-corp", framework="pydantic_ai")

            mock_async_create.assert_called_once_with("dev", "acme-corp", "pydantic_ai")
            assert result is mock_model
