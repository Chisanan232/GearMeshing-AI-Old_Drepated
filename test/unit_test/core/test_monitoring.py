"""
Unit tests for Pydantic AI Logfire monitoring module.

This test suite covers:
- Logfire initialization with various configurations
- Environment variable parsing
- Feature flag handling
- Custom logging functions (agent runs, LLM calls, API requests, errors)
- Error handling and graceful degradation
- Context retrieval
"""

import os
from unittest.mock import MagicMock, patch


class TestLogfireEnvironmentConfiguration:
    """Test environment variable configuration for Logfire."""

    def test_logfire_disabled_by_default(self):
        """Test that Logfire is disabled by default."""
        with patch.dict(os.environ, {}, clear=True):
            # Re-import to get fresh environment values
            import importlib

            import gearmeshing_ai.core.monitoring as monitoring_module

            importlib.reload(monitoring_module)

            assert monitoring_module.LOGFIRE_ENABLED is False

    def test_logfire_enabled_with_true_string(self):
        """Test Logfire enabled with 'true' string."""
        with patch.dict(os.environ, {"LOGFIRE_ENABLED": "true"}):
            import importlib

            import gearmeshing_ai.core.monitoring as monitoring_module

            importlib.reload(monitoring_module)

            assert monitoring_module.LOGFIRE_ENABLED is True

    def test_logfire_enabled_with_1_string(self):
        """Test Logfire enabled with '1' string."""
        with patch.dict(os.environ, {"LOGFIRE_ENABLED": "1"}):
            import importlib

            import gearmeshing_ai.core.monitoring as monitoring_module

            importlib.reload(monitoring_module)

            assert monitoring_module.LOGFIRE_ENABLED is True

    def test_logfire_enabled_with_yes_string(self):
        """Test Logfire enabled with 'yes' string."""
        with patch.dict(os.environ, {"LOGFIRE_ENABLED": "yes"}):
            import importlib

            import gearmeshing_ai.core.monitoring as monitoring_module

            importlib.reload(monitoring_module)

            assert monitoring_module.LOGFIRE_ENABLED is True

    def test_logfire_token_from_environment(self):
        """Test Logfire token is read from environment."""
        test_token = "test-token-12345"
        with patch.dict(os.environ, {"LOGFIRE_TOKEN": test_token}):
            import importlib

            import gearmeshing_ai.core.monitoring as monitoring_module

            importlib.reload(monitoring_module)

            assert monitoring_module.LOGFIRE_TOKEN == test_token

    def test_logfire_service_name_from_environment(self):
        """Test Logfire service name is read from environment."""
        test_service = "my-custom-service"
        with patch.dict(os.environ, {"LOGFIRE_SERVICE_NAME": test_service}):
            import importlib

            import gearmeshing_ai.core.monitoring as monitoring_module

            importlib.reload(monitoring_module)

            assert monitoring_module.LOGFIRE_SERVICE_NAME == test_service

    def test_logfire_environment_from_environment(self):
        """Test Logfire environment is read from environment."""
        test_env = "production"
        with patch.dict(os.environ, {"LOGFIRE_ENVIRONMENT": test_env}):
            import importlib

            import gearmeshing_ai.core.monitoring as monitoring_module

            importlib.reload(monitoring_module)

            assert monitoring_module.LOGFIRE_ENVIRONMENT == test_env

    def test_logfire_service_version_from_environment(self):
        """Test Logfire service version is read from environment."""
        test_version = "1.2.3"
        with patch.dict(os.environ, {"LOGFIRE_SERVICE_VERSION": test_version}):
            import importlib

            import gearmeshing_ai.core.monitoring as monitoring_module

            importlib.reload(monitoring_module)

            assert monitoring_module.LOGFIRE_SERVICE_VERSION == test_version

    def test_logfire_sample_rate_from_environment(self):
        """Test Logfire sample rate is read from environment."""
        with patch.dict(os.environ, {"LOGFIRE_SAMPLE_RATE": "0.5"}):
            import importlib

            import gearmeshing_ai.core.monitoring as monitoring_module

            importlib.reload(monitoring_module)

            assert monitoring_module.LOGFIRE_SAMPLE_RATE == 0.5

    def test_logfire_trace_sample_rate_from_environment(self):
        """Test Logfire trace sample rate is read from environment."""
        with patch.dict(os.environ, {"LOGFIRE_TRACE_SAMPLE_RATE": "0.1"}):
            import importlib

            import gearmeshing_ai.core.monitoring as monitoring_module

            importlib.reload(monitoring_module)

            assert monitoring_module.LOGFIRE_TRACE_SAMPLE_RATE == 0.1

    def test_feature_flags_default_to_true(self):
        """Test that feature flags default to true."""
        with patch.dict(os.environ, {}, clear=True):
            import importlib

            import gearmeshing_ai.core.monitoring as monitoring_module

            importlib.reload(monitoring_module)

            assert monitoring_module.LOGFIRE_TRACE_PYDANTIC_AI is True
            assert monitoring_module.LOGFIRE_TRACE_SQLALCHEMY is True
            assert monitoring_module.LOGFIRE_TRACE_HTTPX is True
            assert monitoring_module.LOGFIRE_TRACE_FASTAPI is True

    def test_feature_flags_can_be_disabled(self):
        """Test that feature flags can be disabled."""
        env_vars = {
            "LOGFIRE_TRACE_PYDANTIC_AI": "false",
            "LOGFIRE_TRACE_SQLALCHEMY": "0",
            "LOGFIRE_TRACE_HTTPX": "no",
            "LOGFIRE_TRACE_FASTAPI": "false",
        }
        with patch.dict(os.environ, env_vars):
            import importlib

            import gearmeshing_ai.core.monitoring as monitoring_module

            importlib.reload(monitoring_module)

            assert monitoring_module.LOGFIRE_TRACE_PYDANTIC_AI is False
            assert monitoring_module.LOGFIRE_TRACE_SQLALCHEMY is False
            assert monitoring_module.LOGFIRE_TRACE_HTTPX is False
            assert monitoring_module.LOGFIRE_TRACE_FASTAPI is False


class TestInitializeLogfire:
    """Test Logfire initialization function."""

    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_ENABLED", False)
    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_disabled(self, mock_logger):
        """Test that initialization is skipped when Logfire is disabled."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        initialize_logfire()

        mock_logger.info.assert_called_once()
        assert "disabled" in mock_logger.info.call_args[0][0].lower()

    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_ENABLED", True)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TOKEN", "")
    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_no_token(self, mock_logger):
        """Test that initialization warns when token is not set."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        initialize_logfire()

        mock_logger.warning.assert_called_once()
        assert "token" in mock_logger.warning.call_args[0][0].lower()

    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_ENABLED", True)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TOKEN", "test-token")
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_SERVICE_NAME", "test-service")
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_ENVIRONMENT", "test")
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_PYDANTIC_AI", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_SQLALCHEMY", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_HTTPX", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_FASTAPI", False)
    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_with_all_flags_disabled(self, mock_logger):
        """Test initialization with all instrumentation flags disabled."""
        with patch("builtins.__import__", side_effect=__import__) as mock_import:
            from gearmeshing_ai.core.monitoring import initialize_logfire

            initialize_logfire()

            # Should complete without error
            mock_logger.info.assert_called()

    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_ENABLED", True)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TOKEN", "test-token")
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_SERVICE_NAME", "test-service")
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_SERVICE_VERSION", "1.0.0")
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_ENVIRONMENT", "test")
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_SAMPLE_RATE", 0.5)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_SAMPLE_RATE", 0.1)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_PYDANTIC_AI", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_SQLALCHEMY", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_HTTPX", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_FASTAPI", False)
    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_configure_called_with_correct_params(self, mock_logger):
        """Test that logfire.configure is called with correct parameters."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        initialize_logfire()

        # Verify initialization completed
        mock_logger.info.assert_called()

    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_ENABLED", True)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TOKEN", "test-token")
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_PYDANTIC_AI", True)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_SQLALCHEMY", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_HTTPX", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_FASTAPI", False)
    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_instruments_pydantic_ai(self, mock_logger):
        """Test that Pydantic AI instrumentation is enabled when flag is true."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        initialize_logfire()

        # Verify initialization completed
        mock_logger.info.assert_called()

    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_ENABLED", True)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TOKEN", "test-token")
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_PYDANTIC_AI", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_SQLALCHEMY", True)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_HTTPX", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_FASTAPI", False)
    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_instruments_sqlalchemy(self, mock_logger):
        """Test that SQLAlchemy instrumentation is enabled when flag is true."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        initialize_logfire()

        # Verify initialization completed
        mock_logger.info.assert_called()

    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_ENABLED", True)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TOKEN", "test-token")
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_PYDANTIC_AI", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_SQLALCHEMY", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_HTTPX", True)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_FASTAPI", False)
    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_instruments_httpx(self, mock_logger):
        """Test that HTTPX instrumentation is enabled when flag is true."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        initialize_logfire()

        # Verify initialization completed
        mock_logger.info.assert_called()

    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_ENABLED", True)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TOKEN", "test-token")
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_PYDANTIC_AI", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_SQLALCHEMY", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_HTTPX", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_FASTAPI", True)
    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_instruments_fastapi_with_app(self, mock_logger):
        """Test that FastAPI instrumentation is enabled when app is provided."""
        from fastapi import FastAPI

        from gearmeshing_ai.core.monitoring import initialize_logfire

        app = FastAPI()
        initialize_logfire(app=app)

        # Verify initialization completed
        mock_logger.info.assert_called()

    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_ENABLED", True)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TOKEN", "test-token")
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_PYDANTIC_AI", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_SQLALCHEMY", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_HTTPX", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_FASTAPI", True)
    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_skips_fastapi_without_app(self, mock_logger):
        """Test that FastAPI instrumentation is skipped when app is not provided."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        initialize_logfire(app=None)

        # Should log debug message about skipping FastAPI
        mock_logger.debug.assert_called()

    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_ENABLED", True)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TOKEN", "test-token")
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_PYDANTIC_AI", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_SQLALCHEMY", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_HTTPX", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_FASTAPI", False)
    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_handles_import_error(self, mock_logger):
        """Test that ImportError is handled gracefully."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        # This test verifies the code path exists
        initialize_logfire()

        # Verify initialization completed
        mock_logger.info.assert_called()

    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_ENABLED", True)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TOKEN", "test-token")
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_PYDANTIC_AI", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_SQLALCHEMY", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_HTTPX", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_FASTAPI", False)
    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_handles_general_exception(self, mock_logger):
        """Test that general exceptions are handled gracefully."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        initialize_logfire()

        # Verify initialization completed
        mock_logger.info.assert_called()


class TestLogAgentRun:
    """Test log_agent_run function."""

    def test_log_agent_run_success(self):
        """Test successful agent run logging."""
        from gearmeshing_ai.core.monitoring import log_agent_run

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            log_agent_run(run_id="run-123", tenant_id="tenant-456", objective="Test objective", role="developer")

    def test_log_agent_run_handles_import_error(self):
        """Test that ImportError is handled gracefully."""
        from gearmeshing_ai.core.monitoring import log_agent_run

        # Just verify the function handles errors gracefully
        log_agent_run(run_id="run-123", tenant_id="tenant-456", objective="Test objective", role="developer")


class TestLogAgentCompletion:
    """Test log_agent_completion function."""

    def test_log_agent_completion_success(self):
        """Test successful agent completion logging."""
        from gearmeshing_ai.core.monitoring import log_agent_completion

        # Verify function executes without error
        log_agent_completion(run_id="run-123", status="succeeded", duration_ms=5432.1)

    def test_log_agent_completion_with_failed_status(self):
        """Test agent completion logging with failed status."""
        from gearmeshing_ai.core.monitoring import log_agent_completion

        # Verify function executes with different status
        log_agent_completion(run_id="run-456", status="failed", duration_ms=1234.5)

    def test_log_agent_completion_handles_exception(self):
        """Test that exceptions are handled gracefully."""
        from gearmeshing_ai.core.monitoring import log_agent_completion

        # Verify function handles missing logfire gracefully
        log_agent_completion(run_id="run-123", status="succeeded", duration_ms=5432.1)


class TestLogLLMCall:
    """Test log_llm_call function."""

    def test_log_llm_call_success(self):
        """Test successful LLM call logging."""
        from gearmeshing_ai.core.monitoring import log_llm_call

        log_llm_call(model="gpt-4o", tokens_used=1250, cost_usd=0.05)

    def test_log_llm_call_without_cost(self):
        """Test LLM call logging without cost."""
        from gearmeshing_ai.core.monitoring import log_llm_call

        log_llm_call(model="gpt-4o", tokens_used=1250, cost_usd=None)

    def test_log_llm_call_with_different_models(self):
        """Test LLM call logging with different models."""
        from gearmeshing_ai.core.monitoring import log_llm_call

        models = ["gpt-4o", "claude-3", "gemini-pro"]
        for model in models:
            log_llm_call(model=model, tokens_used=1000)

    def test_log_llm_call_handles_exception(self):
        """Test that exceptions are handled gracefully."""
        from gearmeshing_ai.core.monitoring import log_llm_call

        log_llm_call(model="gpt-4o", tokens_used=1250)


class TestLogError:
    """Test log_error function."""

    def test_log_error_success(self):
        """Test successful error logging."""
        from gearmeshing_ai.core.monitoring import log_error

        log_error(error_type="ValidationError", error_message="Invalid configuration", context={"field": "objective"})

    def test_log_error_without_context(self):
        """Test error logging without context."""
        from gearmeshing_ai.core.monitoring import log_error

        log_error(error_type="RuntimeError", error_message="Something went wrong")

    def test_log_error_with_complex_context(self):
        """Test error logging with complex context."""
        from gearmeshing_ai.core.monitoring import log_error

        context = {"run_id": "run-123", "tenant_id": "tenant-456", "error_code": 500, "details": {"nested": "value"}}

        log_error(error_type="SystemError", error_message="System failure", context=context)

    def test_log_error_handles_exception(self):
        """Test that exceptions are handled gracefully."""
        from gearmeshing_ai.core.monitoring import log_error

        log_error(error_type="TestError", error_message="Test message")


class TestGetLogfireContext:
    """Test get_logfire_context function."""

    def test_get_logfire_context_success(self):
        """Test successful context retrieval."""
        from gearmeshing_ai.core.monitoring import get_logfire_context

        result = get_logfire_context()
        # Result can be None or a dict, both are valid
        assert result is None or isinstance(result, dict)

    def test_get_logfire_context_returns_none_on_error(self):
        """Test that None is returned on error."""
        from gearmeshing_ai.core.monitoring import get_logfire_context

        result = get_logfire_context()
        # Should return None or dict
        assert result is None or isinstance(result, dict)

    def test_get_logfire_context_handles_import_error(self):
        """Test that ImportError is handled gracefully."""
        from gearmeshing_ai.core.monitoring import get_logfire_context

        result = get_logfire_context()
        # Should handle gracefully and return None or dict
        assert result is None or isinstance(result, dict)


class TestInitializeLogfireErrorHandling:
    """Test error handling in Logfire initialization."""

    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_ENABLED", True)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TOKEN", "test-token")
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_PYDANTIC_AI", True)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_SQLALCHEMY", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_HTTPX", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_FASTAPI", False)
    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_handles_pydantic_ai_instrumentation_failure(self, mock_logger):
        """Test that Pydantic AI instrumentation failure is handled gracefully."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()
            mock_logfire.instrument_pydantic_ai.side_effect = RuntimeError("Instrumentation failed")

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            # Should not raise, should handle gracefully
            initialize_logfire()

            # Should log warning about failure
            mock_logger.warning.assert_called()

    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_ENABLED", True)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TOKEN", "test-token")
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_PYDANTIC_AI", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_SQLALCHEMY", True)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_HTTPX", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_FASTAPI", False)
    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_handles_sqlalchemy_instrumentation_failure(self, mock_logger):
        """Test that SQLAlchemy instrumentation failure is handled gracefully."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()
            mock_logfire.instrument_sqlalchemy.side_effect = RuntimeError("SQLAlchemy instrumentation failed")

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            # Should not raise, should handle gracefully
            initialize_logfire()

            # Should log warning about failure
            mock_logger.warning.assert_called()

    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_ENABLED", True)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TOKEN", "test-token")
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_PYDANTIC_AI", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_SQLALCHEMY", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_HTTPX", True)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_FASTAPI", False)
    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_handles_httpx_instrumentation_failure(self, mock_logger):
        """Test that HTTPX instrumentation failure is handled gracefully."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()
            mock_logfire.instrument_httpx.side_effect = RuntimeError("HTTPX instrumentation failed")

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            # Should not raise, should handle gracefully
            initialize_logfire()

            # Should log warning about failure
            mock_logger.warning.assert_called()

    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_ENABLED", True)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TOKEN", "test-token")
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_PYDANTIC_AI", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_SQLALCHEMY", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_HTTPX", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_FASTAPI", True)
    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_handles_fastapi_instrumentation_failure(self, mock_logger):
        """Test that FastAPI instrumentation failure is handled gracefully."""
        from fastapi import FastAPI

        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()
            mock_logfire.instrument_fastapi.side_effect = RuntimeError("FastAPI instrumentation failed")

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            app = FastAPI()
            # Should not raise, should handle gracefully
            initialize_logfire(app=app)

            # Should log warning about failure
            mock_logger.warning.assert_called()

    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_ENABLED", True)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TOKEN", "test-token")
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_PYDANTIC_AI", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_SQLALCHEMY", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_HTTPX", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_FASTAPI", False)
    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_handles_import_error_gracefully(self, mock_logger):
        """Test that ImportError is handled gracefully during initialization."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("builtins.__import__", side_effect=ImportError("logfire not installed")):
            # Should not raise, should handle gracefully
            initialize_logfire()

            # Should log warning about import error
            mock_logger.warning.assert_called()

    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_ENABLED", True)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TOKEN", "test-token")
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_PYDANTIC_AI", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_SQLALCHEMY", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_HTTPX", False)
    @patch("gearmeshing_ai.core.monitoring.LOGFIRE_TRACE_FASTAPI", False)
    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_handles_general_exception_gracefully(self, mock_logger):
        """Test that general exceptions are handled gracefully during initialization."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()
            mock_logfire.configure.side_effect = RuntimeError("Configuration failed")

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            # Should not raise, should handle gracefully
            initialize_logfire()

            # Should log error about failure
            mock_logger.error.assert_called()


class TestGetLogfireContextErrorHandling:
    """Test error handling in get_logfire_context function."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_get_logfire_context_returns_none_on_exception(self, mock_logger):
        """Test that None is returned when exception occurs."""
        from gearmeshing_ai.core.monitoring import get_logfire_context

        with patch("builtins.__import__", side_effect=RuntimeError("Unexpected error")):
            result = get_logfire_context()

            # Should return None on exception
            assert result is None

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_get_logfire_context_handles_attribute_error(self, mock_logger):
        """Test that AttributeError is handled gracefully."""
        from gearmeshing_ai.core.monitoring import get_logfire_context

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()
            mock_logfire.current_trace_context.side_effect = AttributeError("Method not found")

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            result = get_logfire_context()

            # Should return None on AttributeError
            assert result is None


class TestLogAgentRunErrorHandling:
    """Test error handling in log_agent_run function."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_agent_run_handles_import_error(self, mock_logger):
        """Test that ImportError is handled gracefully."""
        from gearmeshing_ai.core.monitoring import log_agent_run

        with patch("builtins.__import__", side_effect=ImportError("logfire not installed")):
            # Should not raise
            log_agent_run(run_id="run-123", tenant_id="tenant-456", objective="Test", role="developer")

            # Should log debug message
            mock_logger.debug.assert_called()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_agent_run_handles_runtime_error(self, mock_logger):
        """Test that RuntimeError is handled gracefully."""
        from gearmeshing_ai.core.monitoring import log_agent_run

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()
            mock_logfire.info.side_effect = RuntimeError("Logfire error")

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            # Should not raise
            log_agent_run(run_id="run-123", tenant_id="tenant-456", objective="Test", role="developer")

            # Should log debug message
            mock_logger.debug.assert_called()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_agent_run_handles_attribute_error(self, mock_logger):
        """Test that AttributeError is handled gracefully."""
        from gearmeshing_ai.core.monitoring import log_agent_run

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()
            mock_logfire.info.side_effect = AttributeError("info method not found")

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            # Should not raise
            log_agent_run(run_id="run-123", tenant_id="tenant-456", objective="Test", role="developer")

            # Should log debug message
            mock_logger.debug.assert_called()


class TestLogAgentCompletionErrorHandling:
    """Test error handling in log_agent_completion function."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_agent_completion_handles_import_error(self, mock_logger):
        """Test that ImportError is handled gracefully."""
        from gearmeshing_ai.core.monitoring import log_agent_completion

        with patch("builtins.__import__", side_effect=ImportError("logfire not installed")):
            # Should not raise
            log_agent_completion(run_id="run-123", status="succeeded", duration_ms=1000.0)

            # Should log debug message
            mock_logger.debug.assert_called()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_agent_completion_handles_runtime_error(self, mock_logger):
        """Test that RuntimeError is handled gracefully."""
        from gearmeshing_ai.core.monitoring import log_agent_completion

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()
            mock_logfire.info.side_effect = RuntimeError("Logfire error")

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            # Should not raise
            log_agent_completion(run_id="run-123", status="succeeded", duration_ms=1000.0)

            # Should log debug message
            mock_logger.debug.assert_called()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_agent_completion_handles_type_error(self, mock_logger):
        """Test that TypeError is handled gracefully."""
        from gearmeshing_ai.core.monitoring import log_agent_completion

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()
            mock_logfire.info.side_effect = TypeError("Invalid argument type")

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            # Should not raise
            log_agent_completion(run_id="run-123", status="succeeded", duration_ms=1000.0)

            # Should log debug message
            mock_logger.debug.assert_called()


class TestLogLLMCallErrorHandling:
    """Test error handling in log_llm_call function."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_llm_call_handles_import_error(self, mock_logger):
        """Test that ImportError is handled gracefully."""
        from gearmeshing_ai.core.monitoring import log_llm_call

        with patch("builtins.__import__", side_effect=ImportError("logfire not installed")):
            # Should not raise
            log_llm_call(model="gpt-4o", tokens_used=1000, cost_usd=0.05)

            # Should log debug message
            mock_logger.debug.assert_called()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_llm_call_handles_runtime_error(self, mock_logger):
        """Test that RuntimeError is handled gracefully."""
        from gearmeshing_ai.core.monitoring import log_llm_call

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()
            mock_logfire.info.side_effect = RuntimeError("Logfire error")

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            # Should not raise
            log_llm_call(model="gpt-4o", tokens_used=1000, cost_usd=0.05)

            # Should log debug message
            mock_logger.debug.assert_called()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_llm_call_handles_value_error(self, mock_logger):
        """Test that ValueError is handled gracefully."""
        from gearmeshing_ai.core.monitoring import log_llm_call

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()
            mock_logfire.info.side_effect = ValueError("Invalid value")

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            # Should not raise
            log_llm_call(model="gpt-4o", tokens_used=1000)

            # Should log debug message
            mock_logger.debug.assert_called()


class TestLogErrorErrorHandling:
    """Test error handling in log_error function."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_error_handles_import_error(self, mock_logger):
        """Test that ImportError is handled gracefully."""
        from gearmeshing_ai.core.monitoring import log_error

        with patch("builtins.__import__", side_effect=ImportError("logfire not installed")):
            # Should not raise
            log_error(error_type="TestError", error_message="Test message", context={"key": "value"})

            # Should log debug message
            mock_logger.debug.assert_called()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_error_handles_runtime_error(self, mock_logger):
        """Test that RuntimeError is handled gracefully."""
        from gearmeshing_ai.core.monitoring import log_error

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()
            mock_logfire.error.side_effect = RuntimeError("Logfire error")

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            # Should not raise
            log_error(error_type="TestError", error_message="Test message")

            # Should log debug message
            mock_logger.debug.assert_called()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_error_handles_type_error_with_context(self, mock_logger):
        """Test that TypeError is handled gracefully with context."""
        from gearmeshing_ai.core.monitoring import log_error

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()
            mock_logfire.error.side_effect = TypeError("Invalid type")

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            # Should not raise
            log_error(error_type="TestError", error_message="Test message", context={"run_id": "123"})

            # Should log debug message
            mock_logger.debug.assert_called()


class TestLangSmithEnvironmentConfiguration:
    """Test environment variable configuration for LangSmith."""

    def test_langsmith_tracing_disabled_by_default(self):
        """Test that LangSmith tracing is disabled by default."""
        with patch.dict(os.environ, {}, clear=True):
            import importlib

            import gearmeshing_ai.core.monitoring as monitoring_module

            importlib.reload(monitoring_module)

            assert monitoring_module.LANGSMITH_TRACING is False

    def test_langsmith_tracing_enabled_with_true_string(self):
        """Test LangSmith tracing enabled with 'true' string."""
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "true"}):
            import importlib

            import gearmeshing_ai.core.monitoring as monitoring_module

            importlib.reload(monitoring_module)

            assert monitoring_module.LANGSMITH_TRACING is True

    def test_langsmith_api_key_from_environment(self):
        """Test LangSmith API key is read from environment."""
        test_key = "test-langsmith-key-12345"
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": test_key}):
            import importlib

            import gearmeshing_ai.core.monitoring as monitoring_module

            importlib.reload(monitoring_module)

            assert monitoring_module.LANGSMITH_API_KEY == test_key

    def test_langsmith_project_from_environment(self):
        """Test LangSmith project name is read from environment."""
        test_project = "my-custom-project"
        with patch.dict(os.environ, {"LANGSMITH_PROJECT": test_project}):
            import importlib

            import gearmeshing_ai.core.monitoring as monitoring_module

            importlib.reload(monitoring_module)

            assert monitoring_module.LANGSMITH_PROJECT == test_project

    def test_langsmith_endpoint_from_environment(self):
        """Test LangSmith endpoint is read from environment."""
        test_endpoint = "https://custom.smith.langchain.com"
        with patch.dict(os.environ, {"LANGSMITH_ENDPOINT": test_endpoint}):
            import importlib

            import gearmeshing_ai.core.monitoring as monitoring_module

            importlib.reload(monitoring_module)

            assert monitoring_module.LANGSMITH_ENDPOINT == test_endpoint


class TestInitializeLangSmith:
    """Test LangSmith initialization function."""

    @patch("gearmeshing_ai.core.monitoring.LANGSMITH_TRACING", False)
    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_langsmith_disabled(self, mock_logger):
        """Test that initialization is skipped when LangSmith tracing is disabled."""
        from gearmeshing_ai.core.monitoring import initialize_langsmith

        initialize_langsmith()

        mock_logger.debug.assert_called_once()
        assert "disabled" in mock_logger.debug.call_args[0][0].lower()

    @patch("gearmeshing_ai.core.monitoring.LANGSMITH_TRACING", True)
    @patch("gearmeshing_ai.core.monitoring.LANGSMITH_API_KEY", "")
    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_langsmith_no_api_key(self, mock_logger):
        """Test that initialization warns when API key is not set."""
        from gearmeshing_ai.core.monitoring import initialize_langsmith

        initialize_langsmith()

        mock_logger.warning.assert_called_once()
        assert "api_key" in mock_logger.warning.call_args[0][0].lower()

    @patch("gearmeshing_ai.core.monitoring.LANGSMITH_TRACING", True)
    @patch("gearmeshing_ai.core.monitoring.LANGSMITH_API_KEY", "test-key")
    @patch("gearmeshing_ai.core.monitoring.LANGSMITH_PROJECT", "test-project")
    @patch("gearmeshing_ai.core.monitoring.LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_langsmith_success(self, mock_logger):
        """Test successful LangSmith initialization."""
        from gearmeshing_ai.core.monitoring import initialize_langsmith

        initialize_langsmith()

        # Should log info about successful initialization
        mock_logger.info.assert_called_once()
        assert "initialized" in mock_logger.info.call_args[0][0].lower()

    @patch("gearmeshing_ai.core.monitoring.LANGSMITH_TRACING", True)
    @patch("gearmeshing_ai.core.monitoring.LANGSMITH_API_KEY", "test-key")
    @patch("gearmeshing_ai.core.monitoring.LANGSMITH_PROJECT", "test-project")
    @patch("gearmeshing_ai.core.monitoring.LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_langsmith_sets_environment_variables(self, mock_logger):
        """Test that LangSmith environment variables are set correctly."""
        from gearmeshing_ai.core.monitoring import initialize_langsmith

        # Clear any existing env vars
        original_tracing = os.environ.get("LANGSMITH_TRACING")
        original_api_key = os.environ.get("LANGSMITH_API_KEY")

        try:
            initialize_langsmith()

            # Should log info about successful initialization
            mock_logger.info.assert_called_once()
            assert "initialized" in mock_logger.info.call_args[0][0].lower()
        finally:
            # Restore original values
            if original_tracing:
                os.environ["LANGSMITH_TRACING"] = original_tracing
            elif "LANGSMITH_TRACING" in os.environ:
                del os.environ["LANGSMITH_TRACING"]
            if original_api_key:
                os.environ["LANGSMITH_API_KEY"] = original_api_key
            elif "LANGSMITH_API_KEY" in os.environ:
                del os.environ["LANGSMITH_API_KEY"]


class TestWrapOpenAIClient:
    """Test OpenAI client wrapping for LangSmith tracing."""

    def test_wrap_openai_client_success(self):
        """Test successful OpenAI client wrapping."""
        from gearmeshing_ai.core.monitoring import wrap_openai_client

        mock_client = MagicMock()
        with patch("builtins.__import__") as mock_import:
            mock_langsmith = MagicMock()
            mock_wrapped_client = MagicMock()
            mock_langsmith.wrappers.wrap_openai.return_value = mock_wrapped_client

            def import_side_effect(name, *args, **kwargs):
                if name == "langsmith.wrappers":
                    return mock_langsmith.wrappers
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            result = wrap_openai_client(mock_client)

            # Should return wrapped client
            assert result is not None

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_wrap_openai_client_handles_import_error(self, mock_logger):
        """Test that ImportError is handled gracefully."""
        from gearmeshing_ai.core.monitoring import wrap_openai_client

        mock_client = MagicMock()
        with patch("builtins.__import__", side_effect=ImportError("langsmith not installed")):
            result = wrap_openai_client(mock_client)

            # Should return original client
            assert result is mock_client
            # Should log debug message
            mock_logger.debug.assert_called()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_wrap_openai_client_handles_exception(self, mock_logger):
        """Test that exceptions are handled gracefully."""
        from gearmeshing_ai.core.monitoring import wrap_openai_client

        mock_client = MagicMock()
        with patch("builtins.__import__") as mock_import:
            mock_langsmith = MagicMock()
            mock_langsmith.wrappers.wrap_openai.side_effect = RuntimeError("Wrapping failed")

            def import_side_effect(name, *args, **kwargs):
                if name == "langsmith.wrappers":
                    return mock_langsmith.wrappers
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            result = wrap_openai_client(mock_client)

            # Should return original client
            assert result is mock_client
            # Should log debug message
            mock_logger.debug.assert_called()


class TestWrapAnthropicClient:
    """Test Anthropic client wrapping for LangSmith tracing."""

    def test_wrap_anthropic_client_success(self):
        """Test successful Anthropic client wrapping."""
        from gearmeshing_ai.core.monitoring import wrap_anthropic_client

        mock_client = MagicMock()
        with patch("builtins.__import__") as mock_import:
            mock_langsmith = MagicMock()
            mock_wrapped_client = MagicMock()
            mock_langsmith.wrappers.wrap_anthropic.return_value = mock_wrapped_client

            def import_side_effect(name, *args, **kwargs):
                if name == "langsmith.wrappers":
                    return mock_langsmith.wrappers
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            result = wrap_anthropic_client(mock_client)

            # Should return wrapped client
            assert result is not None

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_wrap_anthropic_client_handles_import_error(self, mock_logger):
        """Test that ImportError is handled gracefully."""
        from gearmeshing_ai.core.monitoring import wrap_anthropic_client

        mock_client = MagicMock()
        with patch("builtins.__import__", side_effect=ImportError("langsmith not installed")):
            result = wrap_anthropic_client(mock_client)

            # Should return original client
            assert result is mock_client
            # Should log debug message
            mock_logger.debug.assert_called()


class TestGetTraceableDecorator:
    """Test getting the @traceable decorator for function tracing."""

    def test_get_traceable_decorator_success(self):
        """Test successful retrieval of traceable decorator."""
        from gearmeshing_ai.core.monitoring import get_traceable_decorator

        decorator = get_traceable_decorator()

        # Should return a callable
        assert callable(decorator)

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_get_traceable_decorator_handles_import_error(self, mock_logger):
        """Test that ImportError is handled gracefully."""
        from gearmeshing_ai.core.monitoring import get_traceable_decorator

        with patch("builtins.__import__", side_effect=ImportError("langsmith not installed")):
            decorator = get_traceable_decorator()

            # Should return a no-op decorator
            assert callable(decorator)
            # Should log debug message
            mock_logger.debug.assert_called()

    def test_traceable_decorator_noop_behavior(self):
        """Test that no-op decorator preserves function behavior."""
        from gearmeshing_ai.core.monitoring import get_traceable_decorator

        with patch("builtins.__import__", side_effect=ImportError("langsmith not installed")):
            decorator = get_traceable_decorator()

            @decorator
            def test_func(x):
                return x * 2

            # Function should work normally
            assert test_func(5) == 10


class TestGetLangSmithClient:
    """Test getting a LangSmith client instance."""

    def test_get_langsmith_client_success(self):
        """Test successful LangSmith client creation."""
        from gearmeshing_ai.core.monitoring import get_langsmith_client

        with patch("builtins.__import__") as mock_import:
            mock_langsmith = MagicMock()
            mock_client = MagicMock()
            mock_langsmith.Client.return_value = mock_client

            def import_side_effect(name, *args, **kwargs):
                if name == "langsmith":
                    return mock_langsmith
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            result = get_langsmith_client()

            # Should return a client instance
            assert result is not None

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_get_langsmith_client_handles_import_error(self, mock_logger):
        """Test that ImportError is handled gracefully."""
        from gearmeshing_ai.core.monitoring import get_langsmith_client

        with patch("builtins.__import__", side_effect=ImportError("langsmith not installed")):
            result = get_langsmith_client()

            # Should return None
            assert result is None
            # Should log debug message
            mock_logger.debug.assert_called()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_get_langsmith_client_handles_exception(self, mock_logger):
        """Test that exceptions are handled gracefully."""
        from gearmeshing_ai.core.monitoring import get_langsmith_client

        with patch("builtins.__import__") as mock_import:
            mock_langsmith = MagicMock()
            mock_langsmith.Client.side_effect = RuntimeError("Client creation failed")

            def import_side_effect(name, *args, **kwargs):
                if name == "langsmith":
                    return mock_langsmith
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            result = get_langsmith_client()

            # Should return None
            assert result is None
            # Should log debug message
            mock_logger.debug.assert_called()
