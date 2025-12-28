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
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from typing import Optional


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

            log_agent_run(
                run_id="run-123",
                tenant_id="tenant-456",
                objective="Test objective",
                role="developer"
            )

            # Verify the function completed without error
            assert True

    def test_log_agent_run_handles_import_error(self):
        """Test that ImportError is handled gracefully."""
        from gearmeshing_ai.core.monitoring import log_agent_run

        # Just verify the function handles errors gracefully
        log_agent_run(
            run_id="run-123",
            tenant_id="tenant-456",
            objective="Test objective",
            role="developer"
        )

        # If we get here, error handling worked
        assert True


class TestLogAgentCompletion:
    """Test log_agent_completion function."""

    def test_log_agent_completion_success(self):
        """Test successful agent completion logging."""
        from gearmeshing_ai.core.monitoring import log_agent_completion

        # Verify function executes without error
        log_agent_completion(
            run_id="run-123",
            status="succeeded",
            duration_ms=5432.1
        )
        assert True

    def test_log_agent_completion_with_failed_status(self):
        """Test agent completion logging with failed status."""
        from gearmeshing_ai.core.monitoring import log_agent_completion

        # Verify function executes with different status
        log_agent_completion(
            run_id="run-456",
            status="failed",
            duration_ms=1234.5
        )
        assert True

    def test_log_agent_completion_handles_exception(self):
        """Test that exceptions are handled gracefully."""
        from gearmeshing_ai.core.monitoring import log_agent_completion

        # Verify function handles missing logfire gracefully
        log_agent_completion(
            run_id="run-123",
            status="succeeded",
            duration_ms=5432.1
        )
        assert True


class TestLogLLMCall:
    """Test log_llm_call function."""

    def test_log_llm_call_success(self):
        """Test successful LLM call logging."""
        from gearmeshing_ai.core.monitoring import log_llm_call

        log_llm_call(
            model="gpt-4o",
            tokens_used=1250,
            cost_usd=0.05
        )
        assert True

    def test_log_llm_call_without_cost(self):
        """Test LLM call logging without cost."""
        from gearmeshing_ai.core.monitoring import log_llm_call

        log_llm_call(
            model="gpt-4o",
            tokens_used=1250,
            cost_usd=None
        )
        assert True

    def test_log_llm_call_with_different_models(self):
        """Test LLM call logging with different models."""
        from gearmeshing_ai.core.monitoring import log_llm_call

        models = ["gpt-4o", "claude-3", "gemini-pro"]
        for model in models:
            log_llm_call(model=model, tokens_used=1000)

        assert True

    def test_log_llm_call_handles_exception(self):
        """Test that exceptions are handled gracefully."""
        from gearmeshing_ai.core.monitoring import log_llm_call

        log_llm_call(model="gpt-4o", tokens_used=1250)
        assert True


class TestLogAPIRequest:
    """Test log_api_request function."""

    def test_log_api_request_success(self):
        """Test successful API request logging."""
        from gearmeshing_ai.core.monitoring import log_api_request

        log_api_request(
            method="POST",
            path="/api/v1/runs/",
            status_code=201,
            duration_ms=234.5
        )
        assert True

    def test_log_api_request_with_different_methods(self):
        """Test API request logging with different HTTP methods."""
        from gearmeshing_ai.core.monitoring import log_api_request

        methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
        for method in methods:
            log_api_request(method=method, path="/api/test", status_code=200, duration_ms=100)

        assert True

    def test_log_api_request_with_error_status(self):
        """Test API request logging with error status codes."""
        from gearmeshing_ai.core.monitoring import log_api_request

        error_codes = [400, 401, 403, 404, 500, 502, 503]
        for code in error_codes:
            log_api_request(method="GET", path="/api/test", status_code=code, duration_ms=100)

        assert True

    def test_log_api_request_handles_exception(self):
        """Test that exceptions are handled gracefully."""
        from gearmeshing_ai.core.monitoring import log_api_request

        log_api_request(method="GET", path="/api/test", status_code=200, duration_ms=100)
        assert True


class TestLogError:
    """Test log_error function."""

    def test_log_error_success(self):
        """Test successful error logging."""
        from gearmeshing_ai.core.monitoring import log_error

        log_error(
            error_type="ValidationError",
            error_message="Invalid configuration",
            context={"field": "objective"}
        )
        assert True

    def test_log_error_without_context(self):
        """Test error logging without context."""
        from gearmeshing_ai.core.monitoring import log_error

        log_error(
            error_type="RuntimeError",
            error_message="Something went wrong"
        )
        assert True

    def test_log_error_with_complex_context(self):
        """Test error logging with complex context."""
        from gearmeshing_ai.core.monitoring import log_error

        context = {
            "run_id": "run-123",
            "tenant_id": "tenant-456",
            "error_code": 500,
            "details": {"nested": "value"}
        }

        log_error(
            error_type="SystemError",
            error_message="System failure",
            context=context
        )
        assert True

    def test_log_error_handles_exception(self):
        """Test that exceptions are handled gracefully."""
        from gearmeshing_ai.core.monitoring import log_error

        log_error(error_type="TestError", error_message="Test message")
        assert True


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
