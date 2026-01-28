"""Integration tests for model provider with engine and planner."""

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
from gearmeshing_ai.core.models.config import ModelConfig


class TestModelProviderIntegration:
    """Integration tests for UnifiedModelProvider with database configuration."""

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        return MagicMock()

    @pytest.fixture
    def mock_model_config(self):
        """Create a mock model configuration."""
        return ModelConfig(
            role="dev",
            provider="openai",
            model="gpt-4o",
            temperature=0.7,
            max_tokens=4096,
            top_p=0.9,
        )

    def test_get_model_provider_returns_unified_provider(self, mock_db_session):
        """Test get_model_provider factory function returns UnifiedModelProvider."""
        provider = get_model_provider(mock_db_session)
        assert isinstance(provider, UnifiedModelProvider)
        assert provider.db_session is mock_db_session

    def test_unified_model_provider_initialization(self, mock_db_session):
        """Test UnifiedModelProvider initialization with database session."""
        provider = UnifiedModelProvider(db_session=mock_db_session)
        assert provider.db_session is mock_db_session
        assert provider._db_provider is None
        assert provider.framework == "pydantic_ai"

    def test_create_model_integration_with_abstraction(self, mock_db_session):
        """Test that create_model properly integrates with abstraction layer."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

        with patch.object(provider._provider, "create_model") as mock_create:
            mock_create.return_value = MagicMock()

            result = provider.create_model("openai", "gpt-4o", temperature=0.5)

            mock_create.assert_called_once()
            assert result is not None

    def test_create_model_with_explicit_framework(self, mock_db_session):
        """Test UnifiedModelProvider with explicit framework selection."""
        provider = UnifiedModelProvider(db_session=mock_db_session, framework="pydantic_ai")

        with patch.object(provider._provider, "create_model") as mock_create:
            mock_create.return_value = MagicMock()

            result = provider.create_model("openai", "gpt-4o")

            mock_create.assert_called_once()
            assert result is not None

    def test_create_model_handles_provider_errors(self, mock_db_session):
        """Test that create_model properly handles abstraction layer errors."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

        with patch.object(provider._provider, "create_model") as mock_create:
            mock_create.side_effect = RuntimeError("Provider error")

            with pytest.raises(RuntimeError, match="Provider error"):
                provider.create_model("openai", "gpt-4o")

    def test_create_model_for_role_integration(self, mock_db_session, mock_model_config):
        """Test create_model_for_role with database configuration."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

        with patch.object(provider, "create_model_for_role") as mock_create_role:
            mock_create_role.return_value = MagicMock()

            result = provider.create_model_for_role("dev", tenant_id="acme-corp")

            mock_create_role.assert_called_once_with("dev", tenant_id="acme-corp")
            assert result is not None

    def test_get_supported_providers_integration(self, mock_db_session):
        """Test getting supported providers from abstraction layer."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

        with patch.object(provider._provider, "get_supported_providers") as mock_supported:
            mock_supported.return_value = ["openai", "anthropic", "google"]

            result = provider.get_supported_providers()

            assert result == ["openai", "anthropic", "google"]
            mock_supported.assert_called_once()

    def test_get_supported_models_integration(self, mock_db_session):
        """Test getting supported models from abstraction layer."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

        with patch.object(provider._provider, "get_supported_models") as mock_models:
            mock_models.return_value = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]

            result = provider.get_supported_models("openai")

            assert result == ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
            mock_models.assert_called_once_with("openai")

    def test_create_fallback_model_integration(self, mock_db_session):
        """Test creating fallback model through abstraction layer."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

        with patch.object(provider._provider, "create_fallback_model") as mock_fallback:
            mock_fallback.return_value = MagicMock()

            result = provider.create_fallback_model("openai", "gpt-4o", "anthropic", "claude-3-5-sonnet")

            mock_fallback.assert_called_once()
            assert result is not None

    def test_get_provider_from_model_name_integration(self, mock_db_session):
        """Test getting provider from model name through abstraction layer."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

        with patch.object(provider._provider, "get_provider_from_model_name") as mock_get_provider:
            mock_get_provider.return_value = "openai"

            result = provider.get_provider_from_model_name("gpt-4o")

            assert result == "openai"
            mock_get_provider.assert_called_once_with("gpt-4o")

    def test_model_provider_factory_function_integration(self, mock_db_session):
        """Test get_model_provider factory function integration."""
        with patch("gearmeshing_ai.agent_core.model_provider.UnifiedModelProvider") as mock_provider_class:
            mock_provider_class.return_value = MagicMock()

            result = get_model_provider(mock_db_session)

            mock_provider_class.assert_called_once_with(mock_db_session, "pydantic_ai")
            assert result is not None

    def test_create_model_for_role_function_integration(self, mock_db_session):
        """Test create_model_for_role function integration."""
        with patch("gearmeshing_ai.agent_core.model_provider.get_model_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_get_provider.return_value = mock_provider
            mock_provider.create_model_for_role.return_value = MagicMock()

            result = create_model_for_role(mock_db_session, "dev", tenant_id="acme-corp")

            mock_get_provider.assert_called_once_with(mock_db_session, "pydantic_ai")
            mock_provider.create_model_for_role.assert_called_once_with("dev", "acme-corp")
            assert result is not None

    def test_abstraction_layer_error_propagation(self, mock_db_session):
        """Test that errors from abstraction layer are properly propagated."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

        # Test ValueError from abstraction layer
        with patch.object(provider._provider, "create_model") as mock_create:
            mock_create.side_effect = ValueError("Invalid provider")

            with pytest.raises(ValueError, match="Invalid provider"):
                provider.create_model("invalid", "model")

    def test_framework_selection_integration(self, mock_db_session):
        """Test framework selection in integration context."""
        # Test default framework
        provider_default = UnifiedModelProvider(db_session=mock_db_session)
        assert provider_default.framework == "pydantic_ai"

        # Test explicit framework
        provider_explicit = UnifiedModelProvider(db_session=mock_db_session, framework="pydantic_ai")
        assert provider_explicit.framework == "pydantic_ai"

    def test_unsupported_framework_integration(self, mock_db_session):
        """Test unsupported framework handling in integration context."""
        with pytest.raises(ValueError, match="Unsupported framework"):
            UnifiedModelProvider(db_session=mock_db_session, framework="unsupported_framework")

    def test_model_config_parameter_passing(self, mock_db_session):
        """Test that ModelConfig parameters are correctly passed to abstraction layer."""
        from gearmeshing_ai.agent_core.abstraction import ModelConfig

        provider = UnifiedModelProvider(db_session=mock_db_session)

        with patch.object(provider._provider, "create_model") as mock_create:
            mock_create.return_value = MagicMock()

            # Create model with all parameters
            provider.create_model("openai", "gpt-4o", temperature=0.7, max_tokens=2048, top_p=0.9)

            # Verify ModelConfig was created correctly
            mock_create.assert_called_once()
            call_args = mock_create.call_args[0][0]
            assert isinstance(call_args, ModelConfig)
            assert call_args.provider == "openai"
            assert call_args.model == "gpt-4o"
            assert call_args.temperature == 0.7
            assert call_args.max_tokens == 2048
            assert call_args.top_p == 0.9

    def test_provider_lazy_initialization(self, mock_db_session):
        """Test that abstraction layer provider is properly initialized."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

        # Provider should be initialized on construction
        assert provider._provider is not None
        assert hasattr(provider._provider, "create_model")
        assert hasattr(provider._provider, "get_supported_providers")
        assert hasattr(provider._provider, "get_supported_models")

    def test_assert_statement_in_create_model_for_role(self, mock_db_session) -> None:
        """Test the assertion statement in create_model_for_role (line 241)."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

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

                # This should trigger the assertion on line 241
                result = provider.create_model_for_role("dev", tenant_id="acme-corp")

                # Verify the assertion passed (no exception raised)
                assert result is not None
                mock_create.assert_called_once()

    def test_provider_initialization_error_logging(self, mock_db_session) -> None:
        """Test error logging during provider initialization (lines 76-79)."""
        with patch("gearmeshing_ai.agent_core.model_provider.PydanticAIModelProviderFactory") as mock_factory_class:
            mock_factory = MagicMock()
            mock_factory_class.return_value = mock_factory
            mock_factory.create_provider.side_effect = RuntimeError("Initialization failed")

            with patch("gearmeshing_ai.agent_core.model_provider.logger") as mock_logger:
                with pytest.raises(RuntimeError, match="Initialization failed"):
                    UnifiedModelProvider(db_session=mock_db_session, framework="pydantic_ai")

                # Verify error was logged
                mock_logger.error.assert_called_once()
                assert "Failed to initialize provider" in mock_logger.error.call_args[0][0]

    def test_debug_logging_on_initialization(self, mock_db_session) -> None:
        """Test debug logging on successful initialization (line 76)."""
        with patch("gearmeshing_ai.agent_core.model_provider.logger") as mock_logger:
            provider = UnifiedModelProvider(db_session=mock_db_session)

            # Verify debug message was logged
            mock_logger.debug.assert_called_once()
            assert "Initialized model provider for framework: pydantic_ai" in mock_logger.debug.call_args[0][0]

    def test_debug_logging_in_create_model_for_role(self, mock_db_session) -> None:
        """Test debug logging in create_model_for_role (line 231)."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

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

            with patch.object(provider._provider, "create_model") as mock_create:
                mock_create.return_value = MagicMock()

                with patch("gearmeshing_ai.agent_core.model_provider.logger") as mock_logger:
                    result = provider.create_model_for_role("dev", tenant_id="acme-corp")

                    # Verify debug message was logged
                    mock_logger.debug.assert_called_once()
                    assert "Creating model for role 'dev'" in mock_logger.debug.call_args[0][0]

    def test_session_factory_function_creation(self, mock_db_session) -> None:
        """Test session factory function creation in _get_db_provider (lines 99-100)."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

        with patch("gearmeshing_ai.info_provider.model.provider.DatabaseModelProvider") as mock_db_provider_class:
            mock_db_provider = MagicMock()
            mock_db_provider_class.return_value = mock_db_provider

            # Call _get_db_provider
            provider._get_db_provider()

            # Verify DatabaseModelProvider was called with a callable
            call_args = mock_db_provider_class.call_args[0]
            session_factory = call_args[0]

            # Verify it's a callable that returns the mock session
            assert callable(session_factory)
            result = session_factory()
            assert result is mock_db_session

    def test_factory_registry_creation(self, mock_db_session) -> None:
        """Test factory registry creation in _get_provider_factory (lines 83-85)."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

        # This should work without error
        factory = provider._get_provider_factory()

        # Verify it's the PydanticAI factory
        from gearmeshing_ai.agent_core.abstraction.adapters import (
            PydanticAIModelProviderFactory,
        )

        assert isinstance(factory, PydanticAIModelProviderFactory)

    def test_factory_registry_unsupported_framework(self, mock_db_session) -> None:
        """Test unsupported framework in factory registry (line 88)."""
        provider = UnifiedModelProvider(db_session=mock_db_session)
        provider.framework = "unsupported"

        with pytest.raises(ValueError, match="Unsupported framework: unsupported"):
            provider._get_provider_factory()

    def test_model_config_defaults_in_create_model(self, mock_db_session) -> None:
        """Test ModelConfig defaults in create_model (lines 136-138)."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

        with patch.object(provider._provider, "create_model") as mock_create:
            mock_create.return_value = MagicMock()

            # Call with None values to test defaults
            provider.create_model("openai", "gpt-4o", temperature=None, max_tokens=None, top_p=None)

            # Verify the ModelConfig was created with defaults
            mock_create.assert_called_once()
            config = mock_create.call_args[0][0]
            assert config.temperature == 0.7  # Default value
            assert config.top_p == 0.9  # Default value
            assert config.max_tokens is None  # Should remain None

    def test_model_config_defaults_in_fallback_model(self, mock_db_session) -> None:
        """Test ModelConfig defaults in create_fallback_model (lines 191-193, 199-201)."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

        with patch.object(provider._provider, "create_fallback_model") as mock_create:
            mock_create.return_value = MagicMock()

            # Call with None values to test defaults
            provider.create_fallback_model(
                "openai", "gpt-4o", "anthropic", "claude-3-5-sonnet", temperature=None, max_tokens=None, top_p=None
            )

            # Verify both configs were created with defaults
            mock_create.assert_called_once()
            primary_config, fallback_config = mock_create.call_args[0]

            # Check primary config defaults
            assert primary_config.temperature == 0.7
            assert primary_config.top_p == 0.9
            assert primary_config.max_tokens is None

            # Check fallback config defaults
            assert fallback_config.temperature == 0.7
            assert fallback_config.top_p == 0.9
            assert fallback_config.max_tokens is None

    def test_runtime_error_when_provider_not_initialized(self, mock_db_session) -> None:
        """Test RuntimeError when provider is not initialized (multiple methods)."""
        provider = UnifiedModelProvider(db_session=mock_db_session)

        # Manually set provider to None to test error cases
        provider._provider = None

        # Test all methods that check for provider initialization
        with pytest.raises(RuntimeError, match="Model provider not initialized"):
            provider.create_model("openai", "gpt-4o")

        with pytest.raises(RuntimeError, match="Model provider not initialized"):
            provider.get_provider_from_model_name("gpt-4o")

        with pytest.raises(RuntimeError, match="Model provider not initialized"):
            provider.create_fallback_model("openai", "gpt-4o", "anthropic", "claude-3-5-sonnet")

        with pytest.raises(RuntimeError, match="Model provider not initialized"):
            provider.get_supported_providers()

        with pytest.raises(RuntimeError, match="Model provider not initialized"):
            provider.get_supported_models("openai")

    def test_backward_compatibility_functions(self, mock_db_session) -> None:
        """Test all backward compatibility functions work correctly."""
        # Test get_model_provider function
        with patch("gearmeshing_ai.agent_core.model_provider.UnifiedModelProvider") as mock_provider_class:
            mock_provider_class.return_value = MagicMock()

            result = get_model_provider(mock_db_session)
            assert result is not None
            mock_provider_class.assert_called_once_with(mock_db_session, "pydantic_ai")

        # Test create_model_for_role function
        with patch("gearmeshing_ai.agent_core.model_provider.get_model_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_get_provider.return_value = mock_provider
            mock_provider.create_model_for_role.return_value = MagicMock()

            result = create_model_for_role(mock_db_session, "dev", tenant_id="acme-corp")  # type: ignore[assignment]
            assert result is not None
            mock_get_provider.assert_called_once_with(mock_db_session, "pydantic_ai")
            mock_provider.create_model_for_role.assert_called_once_with("dev", "acme-corp")


class TestAsyncIntegrationCoverage:
    """Integration tests for async functions with complete coverage."""

    @pytest.mark.asyncio
    async def test_async_function_debug_logging(self) -> None:
        """Test debug logging in async_create_model_for_role (line 339)."""
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

                        with patch("gearmeshing_ai.agent_core.model_provider.logger") as mock_logger:
                            result = await async_create_model_for_role("dev")

                            # Verify debug message was logged
                            mock_logger.debug.assert_called_once()
                            assert "Created model for role 'dev' in async context" in mock_logger.debug.call_args[0][0]

    @pytest.mark.asyncio
    async def test_async_function_error_logging(self) -> None:
        """Test error logging in async_create_model_for_role (line 344)."""
        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.database.url = "sqlite+aiosqlite:///test.db"

            with patch("sqlalchemy.create_engine") as mock_create_engine:
                with patch("sqlmodel.Session") as mock_session_class:
                    mock_engine = MagicMock()
                    mock_create_engine.return_value = mock_engine

                    mock_session = MagicMock()
                    mock_session_class.return_value = mock_session

                    with patch("gearmeshing_ai.agent_core.model_provider.get_model_provider") as mock_get_provider:
                        mock_get_provider.side_effect = ValueError("Role not found")

                        with patch("gearmeshing_ai.agent_core.model_provider.logger") as mock_logger:
                            with pytest.raises(ValueError, match="Role not found"):
                                await async_create_model_for_role("nonexistent")

                            # Verify error was logged
                            mock_logger.error.assert_called_once()
                            assert (
                                "Failed to create model for role 'nonexistent' in async context"
                                in mock_logger.error.call_args[0][0]
                            )

    @pytest.mark.asyncio
    async def test_database_url_conversion_postgresql(self) -> None:
        """Test PostgreSQL URL conversion in async function (lines 326-327)."""
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

                        mock_provider.create_model_for_role.return_value = MagicMock()

                        await async_create_model_for_role("dev")

                        # Verify URL conversion
                        mock_create_engine.assert_called_once_with("postgresql://user:pass@localhost/db")

    @pytest.mark.asyncio
    async def test_database_url_conversion_sqlite(self) -> None:
        """Test SQLite URL conversion in async function (lines 328-329)."""
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

                        mock_provider.create_model_for_role.return_value = MagicMock()

                        await async_create_model_for_role("dev")

                        # Verify URL conversion
                        mock_create_engine.assert_called_once_with("sqlite:///test.db")

    @pytest.mark.asyncio
    async def test_database_url_no_conversion(self) -> None:
        """Test database URL without conversion (line 331)."""
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

                        mock_provider.create_model_for_role.return_value = MagicMock()

                        await async_create_model_for_role("dev")

                        # Verify URL was not modified
                        mock_create_engine.assert_called_once_with("mysql://user:pass@localhost/db")

    @pytest.mark.asyncio
    async def test_async_functions_backward_compatibility(self) -> None:
        """Test async functions backward compatibility."""
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

                        # Test async_create_model_for_role
                        result1 = await async_create_model_for_role("dev")
                        assert result1 is mock_model

                        # Test async_get_model_provider alias
                        result2 = await async_get_model_provider("dev")
                        assert result2 is mock_model
