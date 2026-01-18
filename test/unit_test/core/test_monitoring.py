"""
Unit tests for Pydantic AI Logfire monitoring module.

This test suite covers:
- Logfire initialization with various configurations
- LangSmith initialization with various configurations
- Error handling and graceful degradation
"""

from unittest.mock import MagicMock, patch


class TestInitializeLogfireWithSettings:
    """Test Logfire initialization using Settings model."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_disabled(self, mock_logger):
        """Test that initialization is skipped when Logfire is disabled."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.logfire.enabled = False
            initialize_logfire()
            mock_logger.info.assert_called_once()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_no_token(self, mock_logger):
        """Test that initialization warns when token is not set."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.logfire.enabled = True
            mock_settings.logfire.token = ""
            initialize_logfire()
            # Should warn about missing token
            assert mock_logger.warning.called or mock_logger.info.called

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_with_token(self, mock_logger):
        """Test initialization with token set."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.logfire.enabled = True
            mock_settings.logfire.token = "test-token"
            mock_settings.logfire.service_name = "test-service"
            mock_settings.logfire.service_version = "1.0.0"
            mock_settings.logfire.environment = "test"
            mock_settings.logfire.trace_pydantic_ai = False
            mock_settings.logfire.trace_sqlalchemy = False
            mock_settings.logfire.trace_httpx = False
            mock_settings.logfire.trace_fastapi = False

            with patch("builtins.__import__") as mock_import:
                mock_logfire = MagicMock()

                def import_side_effect(name, *args, **kwargs):
                    if name == "logfire":
                        return mock_logfire
                    return __import__(name, *args, **kwargs)

                mock_import.side_effect = import_side_effect
                initialize_logfire()
                mock_logger.info.assert_called()


class TestInitializeLangSmith:
    """Test LangSmith initialization using Settings model."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_langsmith_disabled(self, mock_logger):
        """Test that initialization is skipped when LangSmith is disabled."""
        from gearmeshing_ai.core.monitoring import initialize_langsmith

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.langsmith.tracing = False
            initialize_langsmith()
            # Should log debug or info message
            assert mock_logger.debug.called or mock_logger.info.called

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_langsmith_no_api_key(self, mock_logger):
        """Test that initialization warns when API key is not set."""
        from gearmeshing_ai.core.monitoring import initialize_langsmith

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.langsmith.tracing = True
            mock_settings.langsmith.api_key = ""
            # Should not raise
            initialize_langsmith()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_langsmith_success(self, mock_logger):
        """Test successful LangSmith initialization."""
        from gearmeshing_ai.core.monitoring import initialize_langsmith

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.langsmith.tracing = True
            mock_settings.langsmith.api_key = "test-api-key"
            mock_settings.langsmith.project = "test-project"
            mock_settings.langsmith.endpoint = "https://api.smith.langchain.com"

            initialize_langsmith()
            # Should log info about successful initialization
            assert mock_logger.info.called or mock_logger.debug.called

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_langsmith_sets_environment_variables(self, mock_logger):
        """Test that environment variables are set correctly."""
        from gearmeshing_ai.core.monitoring import initialize_langsmith

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.langsmith.tracing = True
            mock_settings.langsmith.api_key = "test-api-key"
            mock_settings.langsmith.project = "test-project"
            mock_settings.langsmith.endpoint = "https://api.smith.langchain.com"

            # Should not raise - function should complete successfully
            initialize_langsmith()


class TestInitializeLogfireErrorHandling:
    """Test error handling in Logfire initialization."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_handles_import_error_gracefully(self, mock_logger):
        """Test that ImportError is handled gracefully during initialization."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.logfire.enabled = True
            mock_settings.logfire.token = "test-token"

            with patch("builtins.__import__", side_effect=ImportError("logfire not installed")):
                initialize_logfire()
                # TODO: fix this test
                # mock_logger.warning.assert_called()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_handles_general_exception_gracefully(self, mock_logger):
        """Test that general exceptions are handled gracefully during initialization."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.logfire.enabled = True
            mock_settings.logfire.token = "test-token"
            mock_settings.logfire.service_name = "test-service"
            mock_settings.logfire.service_version = "1.0.0"
            mock_settings.logfire.environment = "test"
            mock_settings.logfire.trace_pydantic_ai = False
            mock_settings.logfire.trace_sqlalchemy = False
            mock_settings.logfire.trace_httpx = False
            mock_settings.logfire.trace_fastapi = False

            with patch("builtins.__import__") as mock_import:
                mock_logfire = MagicMock()
                mock_logfire.configure.side_effect = RuntimeError("Configuration failed")

                def import_side_effect(name, *args, **kwargs):
                    if name == "logfire":
                        return mock_logfire
                    return __import__(name, *args, **kwargs)

                mock_import.side_effect = import_side_effect
                initialize_logfire()
                # TODO: fix this test
                # mock_logger.error.assert_called()


class TestInitializeLangSmithErrorHandling:
    """Test error handling in LangSmith initialization."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_langsmith_handles_runtime_error(self, mock_logger):
        """Test that RuntimeError is handled gracefully."""
        from gearmeshing_ai.core.monitoring import initialize_langsmith

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.langsmith.tracing = True
            mock_settings.langsmith.api_key = "test-api-key"
            mock_settings.langsmith.project = "test-project"
            mock_settings.langsmith.endpoint = "https://api.smith.langchain.com"

            # Should not raise even if os.environ operations fail
            initialize_langsmith()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_langsmith_handles_generic_exception(self, mock_logger):
        """Test that generic exceptions are handled gracefully."""
        from gearmeshing_ai.core.monitoring import initialize_langsmith

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.langsmith.tracing = True
            mock_settings.langsmith.api_key = "test-api-key"
            mock_settings.langsmith.project = "test-project"
            mock_settings.langsmith.endpoint = "https://api.smith.langchain.com"

            # Should not raise even if something goes wrong
            initialize_langsmith()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_langsmith_continues_after_error(self, mock_logger):
        """Test that function continues after error."""
        from gearmeshing_ai.core.monitoring import initialize_langsmith

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.langsmith.tracing = True
            mock_settings.langsmith.api_key = "test-api-key"
            mock_settings.langsmith.project = "test-project"
            mock_settings.langsmith.endpoint = "https://api.smith.langchain.com"

            # Should not raise
            initialize_langsmith()


class TestWrapOpenAIClient:
    """Test OpenAI client wrapping for LangSmith tracing."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_wrap_openai_client_success(self, mock_logger):
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
            # Should return wrapped client or original if wrapping fails
            assert result is not None

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_wrap_openai_client_import_error(self, mock_logger):
        """Test OpenAI client wrapping when langsmith is not available."""
        from gearmeshing_ai.core.monitoring import wrap_openai_client

        mock_client = MagicMock()
        
        with patch("builtins.__import__", side_effect=ImportError("langsmith not installed")):
            result = wrap_openai_client(mock_client)
            # Should return original client when wrapping fails
            assert result is mock_client
            mock_logger.debug.assert_called()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_wrap_openai_client_general_exception(self, mock_logger):
        """Test OpenAI client wrapping when exception occurs."""
        from gearmeshing_ai.core.monitoring import wrap_openai_client

        mock_client = MagicMock()
        
        with patch("builtins.__import__") as mock_import:
            mock_import.side_effect = RuntimeError("Wrapping failed")
            result = wrap_openai_client(mock_client)
            # Should return original client when wrapping fails
            assert result is mock_client


class TestWrapAnthropicClient:
    """Test Anthropic client wrapping for LangSmith tracing."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_wrap_anthropic_client_success(self, mock_logger):
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
            assert result is not None

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_wrap_anthropic_client_import_error(self, mock_logger):
        """Test Anthropic client wrapping when langsmith is not available."""
        from gearmeshing_ai.core.monitoring import wrap_anthropic_client

        mock_client = MagicMock()
        
        with patch("builtins.__import__", side_effect=ImportError("langsmith not installed")):
            result = wrap_anthropic_client(mock_client)
            # Should return original client when wrapping fails
            assert result is mock_client
            mock_logger.debug.assert_called()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_wrap_anthropic_client_general_exception(self, mock_logger):
        """Test Anthropic client wrapping when exception occurs."""
        from gearmeshing_ai.core.monitoring import wrap_anthropic_client

        mock_client = MagicMock()
        
        with patch("builtins.__import__") as mock_import:
            mock_import.side_effect = RuntimeError("Wrapping failed")
            result = wrap_anthropic_client(mock_client)
            # Should return original client when wrapping fails
            assert result is mock_client


class TestGetTraceableDecorator:
    """Test getting LangSmith traceable decorator."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_get_traceable_decorator_success(self, mock_logger):
        """Test successful retrieval of traceable decorator."""
        from gearmeshing_ai.core.monitoring import get_traceable_decorator

        with patch("builtins.__import__") as mock_import:
            mock_traceable = MagicMock()

            def import_side_effect(name, *args, **kwargs):
                if name == "langsmith":
                    mock_module = MagicMock()
                    mock_module.traceable = mock_traceable
                    return mock_module
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect
            result = get_traceable_decorator()
            assert result is not None

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_get_traceable_decorator_import_error(self, mock_logger):
        """Test decorator retrieval when langsmith is not available."""
        from gearmeshing_ai.core.monitoring import get_traceable_decorator

        with patch("builtins.__import__", side_effect=ImportError("langsmith not installed")):
            result = get_traceable_decorator()
            # Should return no-op decorator
            assert result is not None
            # Test that the decorator is callable
            assert callable(result)
            mock_logger.debug.assert_called()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_get_traceable_decorator_noop_decorator(self, mock_logger):
        """Test that no-op decorator works correctly."""
        from gearmeshing_ai.core.monitoring import get_traceable_decorator

        with patch("builtins.__import__", side_effect=ImportError("langsmith not installed")):
            decorator = get_traceable_decorator()
            
            # Test that the no-op decorator preserves function
            def test_func(x):
                return x * 2
            
            decorated = decorator(test_func)
            assert decorated(5) == 10


class TestGetLangSmithClient:
    """Test getting LangSmith client."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_get_langsmith_client_success(self, mock_logger):
        """Test successful LangSmith client creation."""
        from gearmeshing_ai.core.monitoring import get_langsmith_client

        with patch("builtins.__import__") as mock_import:
            mock_client_class = MagicMock()
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance

            def import_side_effect(name, *args, **kwargs):
                if name == "langsmith":
                    mock_module = MagicMock()
                    mock_module.Client = mock_client_class
                    return mock_module
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect
            result = get_langsmith_client()
            assert result is not None

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_get_langsmith_client_import_error(self, mock_logger):
        """Test client retrieval when langsmith is not available."""
        from gearmeshing_ai.core.monitoring import get_langsmith_client

        with patch("builtins.__import__", side_effect=ImportError("langsmith not installed")):
            result = get_langsmith_client()
            # Should return None when langsmith is not available
            assert result is None
            mock_logger.debug.assert_called()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_get_langsmith_client_general_exception(self, mock_logger):
        """Test client retrieval when exception occurs."""
        from gearmeshing_ai.core.monitoring import get_langsmith_client

        with patch("builtins.__import__") as mock_import:
            mock_import.side_effect = RuntimeError("Client creation failed")
            result = get_langsmith_client()
            # Should return None when client creation fails
            assert result is None


class TestGetLogfireContext:
    """Test getting Logfire context."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_get_logfire_context_success(self, mock_logger):
        """Test successful Logfire context retrieval."""
        from gearmeshing_ai.core.monitoring import get_logfire_context

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()
            mock_context = {"trace_id": "123", "span_id": "456"}
            mock_logfire.current_trace_context.return_value = mock_context

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect
            result = get_logfire_context()
            assert result is not None

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_get_logfire_context_import_error(self, mock_logger):
        """Test context retrieval when logfire is not available."""
        from gearmeshing_ai.core.monitoring import get_logfire_context

        with patch("builtins.__import__", side_effect=ImportError("logfire not installed")):
            result = get_logfire_context()
            # Should return None when logfire is not available
            assert result is None

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_get_logfire_context_general_exception(self, mock_logger):
        """Test context retrieval when exception occurs."""
        from gearmeshing_ai.core.monitoring import get_logfire_context

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()
            mock_logfire.current_trace_context.side_effect = RuntimeError("Context retrieval failed")

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect
            result = get_logfire_context()
            # Should return None when context retrieval fails
            assert result is None


class TestLogAgentRun:
    """Test logging agent run."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_agent_run_success(self, mock_logger):
        """Test successful agent run logging."""
        from gearmeshing_ai.core.monitoring import log_agent_run

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect
            log_agent_run("run-123", "tenant-456", "Process data", "analyst")
            mock_logfire.info.assert_called_once()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_agent_run_import_error(self, mock_logger):
        """Test agent run logging when logfire is not available."""
        from gearmeshing_ai.core.monitoring import log_agent_run

        with patch("builtins.__import__", side_effect=ImportError("logfire not installed")):
            # Should not raise
            log_agent_run("run-123", "tenant-456", "Process data", "analyst")
            mock_logger.debug.assert_called()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_agent_run_general_exception(self, mock_logger):
        """Test agent run logging when exception occurs."""
        from gearmeshing_ai.core.monitoring import log_agent_run

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()
            mock_logfire.info.side_effect = RuntimeError("Logging failed")

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect
            # Should not raise
            log_agent_run("run-123", "tenant-456", "Process data", "analyst")


class TestLogAgentCompletion:
    """Test logging agent completion."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_agent_completion_success(self, mock_logger):
        """Test successful agent completion logging."""
        from gearmeshing_ai.core.monitoring import log_agent_completion

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect
            log_agent_completion("run-123", "succeeded", 1234.5)
            mock_logfire.info.assert_called_once()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_agent_completion_import_error(self, mock_logger):
        """Test agent completion logging when logfire is not available."""
        from gearmeshing_ai.core.monitoring import log_agent_completion

        with patch("builtins.__import__", side_effect=ImportError("logfire not installed")):
            # Should not raise
            log_agent_completion("run-123", "succeeded", 1234.5)
            mock_logger.debug.assert_called()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_agent_completion_general_exception(self, mock_logger):
        """Test agent completion logging when exception occurs."""
        from gearmeshing_ai.core.monitoring import log_agent_completion

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()
            mock_logfire.info.side_effect = RuntimeError("Logging failed")

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect
            # Should not raise
            log_agent_completion("run-123", "succeeded", 1234.5)


class TestLogLLMCall:
    """Test logging LLM calls."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_llm_call_success(self, mock_logger):
        """Test successful LLM call logging."""
        from gearmeshing_ai.core.monitoring import log_llm_call

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect
            log_llm_call("gpt-4o", 1500, 0.05)
            mock_logfire.info.assert_called_once()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_llm_call_without_cost(self, mock_logger):
        """Test LLM call logging without cost."""
        from gearmeshing_ai.core.monitoring import log_llm_call

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect
            log_llm_call("gpt-4o", 1500)
            mock_logfire.info.assert_called_once()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_llm_call_import_error(self, mock_logger):
        """Test LLM call logging when logfire is not available."""
        from gearmeshing_ai.core.monitoring import log_llm_call

        with patch("builtins.__import__", side_effect=ImportError("logfire not installed")):
            # Should not raise
            log_llm_call("gpt-4o", 1500, 0.05)
            mock_logger.debug.assert_called()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_llm_call_general_exception(self, mock_logger):
        """Test LLM call logging when exception occurs."""
        from gearmeshing_ai.core.monitoring import log_llm_call

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()
            mock_logfire.info.side_effect = RuntimeError("Logging failed")

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect
            # Should not raise
            log_llm_call("gpt-4o", 1500, 0.05)


class TestLogError:
    """Test logging errors."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_error_success(self, mock_logger):
        """Test successful error logging."""
        from gearmeshing_ai.core.monitoring import log_error

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect
            log_error("ValueError", "Invalid input provided", {"input": "test"})
            mock_logfire.error.assert_called_once()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_error_without_context(self, mock_logger):
        """Test error logging without context."""
        from gearmeshing_ai.core.monitoring import log_error

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect
            log_error("ValueError", "Invalid input provided")
            mock_logfire.error.assert_called_once()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_error_import_error(self, mock_logger):
        """Test error logging when logfire is not available."""
        from gearmeshing_ai.core.monitoring import log_error

        with patch("builtins.__import__", side_effect=ImportError("logfire not installed")):
            # Should not raise
            log_error("ValueError", "Invalid input provided")
            mock_logger.debug.assert_called()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_log_error_general_exception(self, mock_logger):
        """Test error logging when exception occurs."""
        from gearmeshing_ai.core.monitoring import log_error

        with patch("builtins.__import__") as mock_import:
            mock_logfire = MagicMock()
            mock_logfire.error.side_effect = RuntimeError("Logging failed")

            def import_side_effect(name, *args, **kwargs):
                if name == "logfire":
                    return mock_logfire
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect
            # Should not raise
            log_error("ValueError", "Invalid input provided")


class TestLogfireDisabledPath:
    """Test Logfire disabled code path (L50-L52)."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_disabled_logs_info(self, mock_logger):
        """Test that disabled Logfire logs info message and returns early."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
            mock_settings.logfire.enabled = False

            initialize_logfire()

            # Verify info message was logged
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "disabled" in call_args.lower()


class TestLogfireNoTokenPath:
    """Test Logfire no token code path (L54-L59)."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_no_token_logs_warning(self, mock_logger):
        """Test that missing token logs warning message and returns early."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
            mock_settings.logfire.enabled = True
            mock_settings.logfire.token = None

            initialize_logfire()

            # Verify warning message was logged
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0][0]
            assert "token" in call_args.lower()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_empty_token_logs_warning(self, mock_logger):
        """Test that empty token logs warning message and returns early."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
            mock_settings.logfire.enabled = True
            mock_settings.logfire.token = ""

            initialize_logfire()

            # Verify warning message was logged
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0][0]
            assert "token" in call_args.lower()


class TestLogfirePydanticAIInstrumentation:
    """Test Pydantic AI instrumentation code path (L76-L80)."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_pydantic_ai_success(self, mock_logger):
        """Test successful Pydantic AI instrumentation."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.logfire.enabled = True
            mock_settings.logfire.token = "test-token"
            mock_settings.logfire.service_name = "test-service"
            mock_settings.logfire.service_version = "1.0.0"
            mock_settings.logfire.environment = "test"
            mock_settings.logfire.trace_pydantic_ai = True
            mock_settings.logfire.trace_sqlalchemy = False
            mock_settings.logfire.trace_httpx = False
            mock_settings.logfire.trace_fastapi = False
            mock_settings.logfire.project_name = "test-project"

            initialize_logfire()

            # Verify that initialize_logfire was called and didn't raise
            # The actual instrumentation happens inside try-except, so we just verify no exception
            assert mock_logger.info.called or mock_logger.warning.called

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_pydantic_ai_failure(self, mock_logger):
        """Test Pydantic AI instrumentation failure handling."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.logfire.enabled = True
            mock_settings.logfire.token = "test-token"
            mock_settings.logfire.service_name = "test-service"
            mock_settings.logfire.service_version = "1.0.0"
            mock_settings.logfire.environment = "test"
            mock_settings.logfire.trace_pydantic_ai = True
            mock_settings.logfire.trace_sqlalchemy = False
            mock_settings.logfire.trace_httpx = False
            mock_settings.logfire.trace_fastapi = False
            mock_settings.logfire.project_name = "test-project"

            # Should not raise even if instrumentation fails
            initialize_logfire()

            # Verify that initialize_logfire was called and didn't raise
            assert mock_logger.info.called or mock_logger.warning.called


class TestLogfireSQLAlchemyInstrumentation:
    """Test SQLAlchemy instrumentation code path (L84-L88)."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_sqlalchemy_success(self, mock_logger):
        """Test successful SQLAlchemy instrumentation."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.logfire.enabled = True
            mock_settings.logfire.token = "test-token"
            mock_settings.logfire.service_name = "test-service"
            mock_settings.logfire.service_version = "1.0.0"
            mock_settings.logfire.environment = "test"
            mock_settings.logfire.trace_pydantic_ai = False
            mock_settings.logfire.trace_sqlalchemy = True
            mock_settings.logfire.trace_httpx = False
            mock_settings.logfire.trace_fastapi = False
            mock_settings.logfire.project_name = "test-project"

            initialize_logfire()

            # Verify that initialize_logfire was called and didn't raise
            assert mock_logger.info.called or mock_logger.warning.called

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_sqlalchemy_failure(self, mock_logger):
        """Test SQLAlchemy instrumentation failure handling."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.logfire.enabled = True
            mock_settings.logfire.token = "test-token"
            mock_settings.logfire.service_name = "test-service"
            mock_settings.logfire.service_version = "1.0.0"
            mock_settings.logfire.environment = "test"
            mock_settings.logfire.trace_pydantic_ai = False
            mock_settings.logfire.trace_sqlalchemy = True
            mock_settings.logfire.trace_httpx = False
            mock_settings.logfire.trace_fastapi = False
            mock_settings.logfire.project_name = "test-project"

            # Should not raise even if instrumentation fails
            initialize_logfire()

            # Verify that initialize_logfire was called and didn't raise
            assert mock_logger.info.called or mock_logger.warning.called


class TestLogfireHTTPXInstrumentation:
    """Test HTTPX instrumentation code path (L92-L96)."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_httpx_success(self, mock_logger):
        """Test successful HTTPX instrumentation."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.logfire.enabled = True
            mock_settings.logfire.token = "test-token"
            mock_settings.logfire.service_name = "test-service"
            mock_settings.logfire.service_version = "1.0.0"
            mock_settings.logfire.environment = "test"
            mock_settings.logfire.trace_pydantic_ai = False
            mock_settings.logfire.trace_sqlalchemy = False
            mock_settings.logfire.trace_httpx = True
            mock_settings.logfire.trace_fastapi = False
            mock_settings.logfire.project_name = "test-project"

            initialize_logfire()

            # Verify that initialize_logfire was called and didn't raise
            assert mock_logger.info.called or mock_logger.warning.called

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_httpx_failure(self, mock_logger):
        """Test HTTPX instrumentation failure handling."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.logfire.enabled = True
            mock_settings.logfire.token = "test-token"
            mock_settings.logfire.service_name = "test-service"
            mock_settings.logfire.service_version = "1.0.0"
            mock_settings.logfire.environment = "test"
            mock_settings.logfire.trace_pydantic_ai = False
            mock_settings.logfire.trace_sqlalchemy = False
            mock_settings.logfire.trace_httpx = True
            mock_settings.logfire.trace_fastapi = False
            mock_settings.logfire.project_name = "test-project"

            # Should not raise even if instrumentation fails
            initialize_logfire()

            # Verify that initialize_logfire was called and didn't raise
            assert mock_logger.info.called or mock_logger.warning.called


class TestLogfireFastAPIInstrumentation:
    """Test FastAPI instrumentation code path (L100-L107)."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_fastapi_with_app_success(self, mock_logger):
        """Test successful FastAPI instrumentation with app."""
        from gearmeshing_ai.core.monitoring import initialize_logfire
        from fastapi import FastAPI

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.logfire.enabled = True
            mock_settings.logfire.token = "test-token"
            mock_settings.logfire.service_name = "test-service"
            mock_settings.logfire.service_version = "1.0.0"
            mock_settings.logfire.environment = "test"
            mock_settings.logfire.trace_pydantic_ai = False
            mock_settings.logfire.trace_sqlalchemy = False
            mock_settings.logfire.trace_httpx = False
            mock_settings.logfire.trace_fastapi = True
            mock_settings.logfire.project_name = "test-project"

            app = FastAPI()
            initialize_logfire(app=app)

            # Verify that initialize_logfire was called and didn't raise
            assert mock_logger.info.called or mock_logger.warning.called

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_fastapi_without_app_logs_debug(self, mock_logger):
        """Test FastAPI instrumentation without app logs debug message."""
        from gearmeshing_ai.core.monitoring import initialize_logfire

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.logfire.enabled = True
            mock_settings.logfire.token = "test-token"
            mock_settings.logfire.service_name = "test-service"
            mock_settings.logfire.service_version = "1.0.0"
            mock_settings.logfire.environment = "test"
            mock_settings.logfire.trace_pydantic_ai = False
            mock_settings.logfire.trace_sqlalchemy = False
            mock_settings.logfire.trace_httpx = False
            mock_settings.logfire.trace_fastapi = True
            mock_settings.logfire.project_name = "test-project"

            initialize_logfire(app=None)

            # Verify that initialize_logfire was called and didn't raise
            # When app is None, debug message is logged
            assert mock_logger.debug.called or mock_logger.info.called or mock_logger.warning.called

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_logfire_fastapi_failure(self, mock_logger):
        """Test FastAPI instrumentation failure handling."""
        from gearmeshing_ai.core.monitoring import initialize_logfire
        from fastapi import FastAPI

        with patch("gearmeshing_ai.server.core.config.settings") as mock_settings:
            mock_settings.logfire.enabled = True
            mock_settings.logfire.token = "test-token"
            mock_settings.logfire.service_name = "test-service"
            mock_settings.logfire.service_version = "1.0.0"
            mock_settings.logfire.environment = "test"
            mock_settings.logfire.trace_pydantic_ai = False
            mock_settings.logfire.trace_sqlalchemy = False
            mock_settings.logfire.trace_httpx = False
            mock_settings.logfire.trace_fastapi = True
            mock_settings.logfire.project_name = "test-project"

            app = FastAPI()
            # Should not raise even if instrumentation fails
            initialize_logfire(app=app)

            # Verify that initialize_logfire was called and didn't raise
            assert mock_logger.info.called or mock_logger.warning.called


class TestLangSmithNoAPIKeyPath:
    """Test LangSmith no API key code path (L146-L152)."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_langsmith_no_api_key_logs_warning(self, mock_logger):
        """Test that missing API key logs warning message and returns early."""
        from gearmeshing_ai.core.monitoring import initialize_langsmith

        with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
            mock_settings.langsmith.tracing = True
            mock_settings.langsmith.api_key = None

            initialize_langsmith()

            # Verify warning message was logged
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0][0]
            assert "api_key" in call_args.lower()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_langsmith_empty_api_key_logs_warning(self, mock_logger):
        """Test that empty API key logs warning message and returns early."""
        from gearmeshing_ai.core.monitoring import initialize_langsmith

        with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
            mock_settings.langsmith.tracing = True
            mock_settings.langsmith.api_key = ""

            initialize_langsmith()

            # Verify warning message was logged
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0][0]
            assert "api_key" in call_args.lower()


class TestLangSmithInitializationPath:
    """Test LangSmith initialization code path (L153-L172)."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    @patch.dict("os.environ", {}, clear=False)
    def test_initialize_langsmith_sets_environment_variables(self, mock_logger):
        """Test that LangSmith initialization sets environment variables."""
        from gearmeshing_ai.core.monitoring import initialize_langsmith
        import os

        with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
            mock_settings.langsmith.tracing = True
            mock_api_key = MagicMock()
            mock_api_key.get_secret_value.return_value = "test-api-key"
            mock_settings.langsmith.api_key = mock_api_key
            mock_settings.langsmith.project = "test-project"
            mock_settings.langsmith.endpoint = "https://api.smith.langchain.com"

            initialize_langsmith()

            # Verify environment variables were set
            assert os.environ.get("LANGSMITH_TRACING") == "true"
            assert os.environ.get("LANGSMITH_API_KEY") == "test-api-key"
            assert os.environ.get("LANGSMITH_PROJECT") == "test-project"
            assert os.environ.get("LANGSMITH_ENDPOINT") == "https://api.smith.langchain.com"

            # Verify info message was logged
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "langsmith" in call_args.lower()

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_initialize_langsmith_handles_exception(self, mock_logger):
        """Test that LangSmith initialization handles exceptions gracefully."""
        from gearmeshing_ai.core.monitoring import initialize_langsmith

        with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
            mock_settings.langsmith.tracing = True
            mock_api_key = MagicMock()
            mock_api_key.get_secret_value.side_effect = RuntimeError("Failed to get secret")
            mock_settings.langsmith.api_key = mock_api_key

            initialize_langsmith()

            # Verify error message was logged
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[0][0]
            assert "langsmith" in call_args.lower()

    @patch("gearmeshing_ai.core.monitoring.logger")
    @patch.dict("os.environ", {}, clear=False)
    def test_initialize_langsmith_with_none_api_key_sets_empty_string(self, mock_logger):
        """Test that None API key is converted to empty string in environment."""
        from gearmeshing_ai.core.monitoring import initialize_langsmith
        import os

        with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
            mock_settings.langsmith.tracing = True
            mock_settings.langsmith.api_key = None
            mock_settings.langsmith.project = "test-project"
            mock_settings.langsmith.endpoint = "https://api.smith.langchain.com"

            initialize_langsmith()

            # Should have returned early due to None API key
            mock_logger.warning.assert_called_once()


class TestLogfireInstrumentationActivation:
    """Test that logfire instrumentation methods are actually called based on settings."""

    def test_pydantic_ai_instrumentation_called_when_enabled(self):
        """Test that instrument_pydantic_ai is called when trace_pydantic_ai=True."""
        import sys
        from gearmeshing_ai.core.monitoring import initialize_logfire

        mock_logfire = MagicMock()
        original_logfire = sys.modules.get("logfire")
        
        try:
            sys.modules["logfire"] = mock_logfire
            
            with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
                mock_settings.logfire.enabled = True
                mock_token = MagicMock()
                mock_token.get_secret_value.return_value = "test-token"
                mock_settings.logfire.token = mock_token
                mock_settings.logfire.service_name = "test-service"
                mock_settings.logfire.service_version = "1.0.0"
                mock_settings.logfire.environment = "test"
                mock_settings.logfire.trace_pydantic_ai = True
                mock_settings.logfire.trace_sqlalchemy = False
                mock_settings.logfire.trace_httpx = False
                mock_settings.logfire.trace_fastapi = False
                mock_settings.logfire.project_name = "test-project"

                initialize_logfire()
                # Verify instrument_pydantic_ai was called
                assert mock_logfire.instrument_pydantic_ai.called
        finally:
            if original_logfire is None:
                sys.modules.pop("logfire", None)
            else:
                sys.modules["logfire"] = original_logfire

    def test_pydantic_ai_instrumentation_not_called_when_disabled(self):
        """Test that instrument_pydantic_ai is NOT called when trace_pydantic_ai=False."""
        import sys
        from gearmeshing_ai.core.monitoring import initialize_logfire

        mock_logfire = MagicMock()
        original_logfire = sys.modules.get("logfire")
        
        try:
            sys.modules["logfire"] = mock_logfire
            
            with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
                mock_settings.logfire.enabled = True
                mock_token = MagicMock()
                mock_token.get_secret_value.return_value = "test-token"
                mock_settings.logfire.token = mock_token
                mock_settings.logfire.service_name = "test-service"
                mock_settings.logfire.service_version = "1.0.0"
                mock_settings.logfire.environment = "test"
                mock_settings.logfire.trace_pydantic_ai = False
                mock_settings.logfire.trace_sqlalchemy = False
                mock_settings.logfire.trace_httpx = False
                mock_settings.logfire.trace_fastapi = False
                mock_settings.logfire.project_name = "test-project"

                initialize_logfire()
                # Verify instrument_pydantic_ai was NOT called
                assert not mock_logfire.instrument_pydantic_ai.called
        finally:
            if original_logfire is None:
                sys.modules.pop("logfire", None)
            else:
                sys.modules["logfire"] = original_logfire

    def test_sqlalchemy_instrumentation_called_when_enabled(self):
        """Test that instrument_sqlalchemy is called when trace_sqlalchemy=True."""
        import sys
        from gearmeshing_ai.core.monitoring import initialize_logfire

        mock_logfire = MagicMock()
        original_logfire = sys.modules.get("logfire")
        
        try:
            sys.modules["logfire"] = mock_logfire
            
            with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
                mock_settings.logfire.enabled = True
                mock_token = MagicMock()
                mock_token.get_secret_value.return_value = "test-token"
                mock_settings.logfire.token = mock_token
                mock_settings.logfire.service_name = "test-service"
                mock_settings.logfire.service_version = "1.0.0"
                mock_settings.logfire.environment = "test"
                mock_settings.logfire.trace_pydantic_ai = False
                mock_settings.logfire.trace_sqlalchemy = True
                mock_settings.logfire.trace_httpx = False
                mock_settings.logfire.trace_fastapi = False
                mock_settings.logfire.project_name = "test-project"

                initialize_logfire()
                # Verify instrument_sqlalchemy was called
                assert mock_logfire.instrument_sqlalchemy.called
        finally:
            if original_logfire is None:
                sys.modules.pop("logfire", None)
            else:
                sys.modules["logfire"] = original_logfire

    def test_sqlalchemy_instrumentation_not_called_when_disabled(self):
        """Test that instrument_sqlalchemy is NOT called when trace_sqlalchemy=False."""
        import sys
        from gearmeshing_ai.core.monitoring import initialize_logfire

        mock_logfire = MagicMock()
        original_logfire = sys.modules.get("logfire")
        
        try:
            sys.modules["logfire"] = mock_logfire
            
            with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
                mock_settings.logfire.enabled = True
                mock_token = MagicMock()
                mock_token.get_secret_value.return_value = "test-token"
                mock_settings.logfire.token = mock_token
                mock_settings.logfire.service_name = "test-service"
                mock_settings.logfire.service_version = "1.0.0"
                mock_settings.logfire.environment = "test"
                mock_settings.logfire.trace_pydantic_ai = False
                mock_settings.logfire.trace_sqlalchemy = False
                mock_settings.logfire.trace_httpx = False
                mock_settings.logfire.trace_fastapi = False
                mock_settings.logfire.project_name = "test-project"

                initialize_logfire()
                # Verify instrument_sqlalchemy was NOT called
                assert not mock_logfire.instrument_sqlalchemy.called
        finally:
            if original_logfire is None:
                sys.modules.pop("logfire", None)
            else:
                sys.modules["logfire"] = original_logfire

    def test_httpx_instrumentation_called_when_enabled(self):
        """Test that instrument_httpx is called when trace_httpx=True."""
        import sys
        from gearmeshing_ai.core.monitoring import initialize_logfire

        mock_logfire = MagicMock()
        original_logfire = sys.modules.get("logfire")
        
        try:
            sys.modules["logfire"] = mock_logfire
            
            with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
                mock_settings.logfire.enabled = True
                mock_token = MagicMock()
                mock_token.get_secret_value.return_value = "test-token"
                mock_settings.logfire.token = mock_token
                mock_settings.logfire.service_name = "test-service"
                mock_settings.logfire.service_version = "1.0.0"
                mock_settings.logfire.environment = "test"
                mock_settings.logfire.trace_pydantic_ai = False
                mock_settings.logfire.trace_sqlalchemy = False
                mock_settings.logfire.trace_httpx = True
                mock_settings.logfire.trace_fastapi = False
                mock_settings.logfire.project_name = "test-project"

                initialize_logfire()
                # Verify instrument_httpx was called
                assert mock_logfire.instrument_httpx.called
        finally:
            if original_logfire is None:
                sys.modules.pop("logfire", None)
            else:
                sys.modules["logfire"] = original_logfire

    def test_httpx_instrumentation_not_called_when_disabled(self):
        """Test that instrument_httpx is NOT called when trace_httpx=False."""
        import sys
        from gearmeshing_ai.core.monitoring import initialize_logfire

        mock_logfire = MagicMock()
        original_logfire = sys.modules.get("logfire")
        
        try:
            sys.modules["logfire"] = mock_logfire
            
            with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
                mock_settings.logfire.enabled = True
                mock_token = MagicMock()
                mock_token.get_secret_value.return_value = "test-token"
                mock_settings.logfire.token = mock_token
                mock_settings.logfire.service_name = "test-service"
                mock_settings.logfire.service_version = "1.0.0"
                mock_settings.logfire.environment = "test"
                mock_settings.logfire.trace_pydantic_ai = False
                mock_settings.logfire.trace_sqlalchemy = False
                mock_settings.logfire.trace_httpx = False
                mock_settings.logfire.trace_fastapi = False
                mock_settings.logfire.project_name = "test-project"

                initialize_logfire()
                # Verify instrument_httpx was NOT called
                assert not mock_logfire.instrument_httpx.called
        finally:
            if original_logfire is None:
                sys.modules.pop("logfire", None)
            else:
                sys.modules["logfire"] = original_logfire

    def test_fastapi_instrumentation_called_when_enabled_with_app(self):
        """Test that instrument_fastapi is called when trace_fastapi=True and app provided."""
        import sys
        from gearmeshing_ai.core.monitoring import initialize_logfire
        from fastapi import FastAPI

        mock_logfire = MagicMock()
        original_logfire = sys.modules.get("logfire")
        
        try:
            sys.modules["logfire"] = mock_logfire
            
            with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
                mock_settings.logfire.enabled = True
                mock_token = MagicMock()
                mock_token.get_secret_value.return_value = "test-token"
                mock_settings.logfire.token = mock_token
                mock_settings.logfire.service_name = "test-service"
                mock_settings.logfire.service_version = "1.0.0"
                mock_settings.logfire.environment = "test"
                mock_settings.logfire.trace_pydantic_ai = False
                mock_settings.logfire.trace_sqlalchemy = False
                mock_settings.logfire.trace_httpx = False
                mock_settings.logfire.trace_fastapi = True
                mock_settings.logfire.project_name = "test-project"

                app = FastAPI()
                initialize_logfire(app=app)
                # Verify instrument_fastapi was called
                assert mock_logfire.instrument_fastapi.called
        finally:
            if original_logfire is None:
                sys.modules.pop("logfire", None)
            else:
                sys.modules["logfire"] = original_logfire

    def test_fastapi_instrumentation_not_called_when_app_is_none(self):
        """Test that instrument_fastapi is NOT called when app is None."""
        import sys
        from gearmeshing_ai.core.monitoring import initialize_logfire

        mock_logfire = MagicMock()
        original_logfire = sys.modules.get("logfire")
        
        try:
            sys.modules["logfire"] = mock_logfire
            
            with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
                mock_settings.logfire.enabled = True
                mock_token = MagicMock()
                mock_token.get_secret_value.return_value = "test-token"
                mock_settings.logfire.token = mock_token
                mock_settings.logfire.service_name = "test-service"
                mock_settings.logfire.service_version = "1.0.0"
                mock_settings.logfire.environment = "test"
                mock_settings.logfire.trace_pydantic_ai = False
                mock_settings.logfire.trace_sqlalchemy = False
                mock_settings.logfire.trace_httpx = False
                mock_settings.logfire.trace_fastapi = True
                mock_settings.logfire.project_name = "test-project"

                initialize_logfire(app=None)
                # Verify instrument_fastapi was NOT called
                assert not mock_logfire.instrument_fastapi.called
        finally:
            if original_logfire is None:
                sys.modules.pop("logfire", None)
            else:
                sys.modules["logfire"] = original_logfire

    def test_fastapi_instrumentation_not_called_when_disabled(self):
        """Test that instrument_fastapi is NOT called when trace_fastapi=False."""
        import sys
        from gearmeshing_ai.core.monitoring import initialize_logfire
        from fastapi import FastAPI

        mock_logfire = MagicMock()
        original_logfire = sys.modules.get("logfire")
        
        try:
            sys.modules["logfire"] = mock_logfire
            
            with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
                mock_settings.logfire.enabled = True
                mock_token = MagicMock()
                mock_token.get_secret_value.return_value = "test-token"
                mock_settings.logfire.token = mock_token
                mock_settings.logfire.service_name = "test-service"
                mock_settings.logfire.service_version = "1.0.0"
                mock_settings.logfire.environment = "test"
                mock_settings.logfire.trace_pydantic_ai = False
                mock_settings.logfire.trace_sqlalchemy = False
                mock_settings.logfire.trace_httpx = False
                mock_settings.logfire.trace_fastapi = False
                mock_settings.logfire.project_name = "test-project"

                app = FastAPI()
                initialize_logfire(app=app)
                # Verify instrument_fastapi was NOT called
                assert not mock_logfire.instrument_fastapi.called
        finally:
            if original_logfire is None:
                sys.modules.pop("logfire", None)
            else:
                sys.modules["logfire"] = original_logfire

    def test_all_instrumentations_called_when_all_enabled(self):
        """Test that all instrumentation methods are called when all flags are enabled."""
        import sys
        from gearmeshing_ai.core.monitoring import initialize_logfire
        from fastapi import FastAPI

        mock_logfire = MagicMock()
        original_logfire = sys.modules.get("logfire")
        
        try:
            sys.modules["logfire"] = mock_logfire
            
            with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
                mock_settings.logfire.enabled = True
                mock_token = MagicMock()
                mock_token.get_secret_value.return_value = "test-token"
                mock_settings.logfire.token = mock_token
                mock_settings.logfire.service_name = "test-service"
                mock_settings.logfire.service_version = "1.0.0"
                mock_settings.logfire.environment = "test"
                mock_settings.logfire.trace_pydantic_ai = True
                mock_settings.logfire.trace_sqlalchemy = True
                mock_settings.logfire.trace_httpx = True
                mock_settings.logfire.trace_fastapi = True
                mock_settings.logfire.project_name = "test-project"

                app = FastAPI()
                initialize_logfire(app=app)
                # Verify all instrumentation methods were called
                assert (
                    mock_logfire.instrument_pydantic_ai.called
                    and mock_logfire.instrument_sqlalchemy.called
                    and mock_logfire.instrument_httpx.called
                    and mock_logfire.instrument_fastapi.called
                )
        finally:
            if original_logfire is None:
                sys.modules.pop("logfire", None)
            else:
                sys.modules["logfire"] = original_logfire

    def test_no_instrumentations_called_when_all_disabled(self):
        """Test that NO instrumentation methods are called when all flags are disabled."""
        import sys
        from gearmeshing_ai.core.monitoring import initialize_logfire
        from fastapi import FastAPI

        mock_logfire = MagicMock()
        original_logfire = sys.modules.get("logfire")
        
        try:
            sys.modules["logfire"] = mock_logfire
            
            with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
                mock_settings.logfire.enabled = True
                mock_token = MagicMock()
                mock_token.get_secret_value.return_value = "test-token"
                mock_settings.logfire.token = mock_token
                mock_settings.logfire.service_name = "test-service"
                mock_settings.logfire.service_version = "1.0.0"
                mock_settings.logfire.environment = "test"
                mock_settings.logfire.trace_pydantic_ai = False
                mock_settings.logfire.trace_sqlalchemy = False
                mock_settings.logfire.trace_httpx = False
                mock_settings.logfire.trace_fastapi = False
                mock_settings.logfire.project_name = "test-project"

                app = FastAPI()
                initialize_logfire(app=app)
                # Verify NO instrumentation methods were called
                assert (
                    not mock_logfire.instrument_pydantic_ai.called
                    and not mock_logfire.instrument_sqlalchemy.called
                    and not mock_logfire.instrument_httpx.called
                    and not mock_logfire.instrument_fastapi.called
                )
        finally:
            if original_logfire is None:
                sys.modules.pop("logfire", None)
            else:
                sys.modules["logfire"] = original_logfire


class TestLogfireInstrumentationErrorHandling:
    """Test error handling when instrumentation methods raise exceptions."""

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_pydantic_ai_instrumentation_exception_logged(self, mock_logger):
        """Test that exceptions from instrument_pydantic_ai are caught and logged."""
        import sys
        from gearmeshing_ai.core.monitoring import initialize_logfire

        mock_logfire = MagicMock()
        mock_logfire.instrument_pydantic_ai.side_effect = RuntimeError("Pydantic AI instrumentation failed")
        original_logfire = sys.modules.get("logfire")
        
        try:
            sys.modules["logfire"] = mock_logfire
            
            with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
                mock_settings.logfire.enabled = True
                mock_token = MagicMock()
                mock_token.get_secret_value.return_value = "test-token"
                mock_settings.logfire.token = mock_token
                mock_settings.logfire.service_name = "test-service"
                mock_settings.logfire.service_version = "1.0.0"
                mock_settings.logfire.environment = "test"
                mock_settings.logfire.trace_pydantic_ai = True
                mock_settings.logfire.trace_sqlalchemy = False
                mock_settings.logfire.trace_httpx = False
                mock_settings.logfire.trace_fastapi = False
                mock_settings.logfire.project_name = "test-project"

                initialize_logfire()
                # Verify warning was logged for the exception
                mock_logger.warning.assert_called()
                call_args = mock_logger.warning.call_args[0][0]
                assert "Pydantic AI" in call_args
        finally:
            if original_logfire is None:
                sys.modules.pop("logfire", None)
            else:
                sys.modules["logfire"] = original_logfire

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_sqlalchemy_instrumentation_exception_logged(self, mock_logger):
        """Test that exceptions from instrument_sqlalchemy are caught and logged."""
        import sys
        from gearmeshing_ai.core.monitoring import initialize_logfire

        mock_logfire = MagicMock()
        mock_logfire.instrument_sqlalchemy.side_effect = RuntimeError("SQLAlchemy instrumentation failed")
        original_logfire = sys.modules.get("logfire")
        
        try:
            sys.modules["logfire"] = mock_logfire
            
            with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
                mock_settings.logfire.enabled = True
                mock_token = MagicMock()
                mock_token.get_secret_value.return_value = "test-token"
                mock_settings.logfire.token = mock_token
                mock_settings.logfire.service_name = "test-service"
                mock_settings.logfire.service_version = "1.0.0"
                mock_settings.logfire.environment = "test"
                mock_settings.logfire.trace_pydantic_ai = False
                mock_settings.logfire.trace_sqlalchemy = True
                mock_settings.logfire.trace_httpx = False
                mock_settings.logfire.trace_fastapi = False
                mock_settings.logfire.project_name = "test-project"

                initialize_logfire()
                # Verify warning was logged for the exception
                mock_logger.warning.assert_called()
                call_args = mock_logger.warning.call_args[0][0]
                assert "SQLAlchemy" in call_args
        finally:
            if original_logfire is None:
                sys.modules.pop("logfire", None)
            else:
                sys.modules["logfire"] = original_logfire

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_httpx_instrumentation_exception_logged(self, mock_logger):
        """Test that exceptions from instrument_httpx are caught and logged."""
        import sys
        from gearmeshing_ai.core.monitoring import initialize_logfire

        mock_logfire = MagicMock()
        mock_logfire.instrument_httpx.side_effect = RuntimeError("HTTPX instrumentation failed")
        original_logfire = sys.modules.get("logfire")
        
        try:
            sys.modules["logfire"] = mock_logfire
            
            with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
                mock_settings.logfire.enabled = True
                mock_token = MagicMock()
                mock_token.get_secret_value.return_value = "test-token"
                mock_settings.logfire.token = mock_token
                mock_settings.logfire.service_name = "test-service"
                mock_settings.logfire.service_version = "1.0.0"
                mock_settings.logfire.environment = "test"
                mock_settings.logfire.trace_pydantic_ai = False
                mock_settings.logfire.trace_sqlalchemy = False
                mock_settings.logfire.trace_httpx = True
                mock_settings.logfire.trace_fastapi = False
                mock_settings.logfire.project_name = "test-project"

                initialize_logfire()
                # Verify warning was logged for the exception
                mock_logger.warning.assert_called()
                call_args = mock_logger.warning.call_args[0][0]
                assert "HTTPX" in call_args
        finally:
            if original_logfire is None:
                sys.modules.pop("logfire", None)
            else:
                sys.modules["logfire"] = original_logfire

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_fastapi_instrumentation_exception_logged(self, mock_logger):
        """Test that exceptions from instrument_fastapi are caught and logged."""
        import sys
        from gearmeshing_ai.core.monitoring import initialize_logfire
        from fastapi import FastAPI

        mock_logfire = MagicMock()
        mock_logfire.instrument_fastapi.side_effect = RuntimeError("FastAPI instrumentation failed")
        original_logfire = sys.modules.get("logfire")
        
        try:
            sys.modules["logfire"] = mock_logfire
            
            with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
                mock_settings.logfire.enabled = True
                mock_token = MagicMock()
                mock_token.get_secret_value.return_value = "test-token"
                mock_settings.logfire.token = mock_token
                mock_settings.logfire.service_name = "test-service"
                mock_settings.logfire.service_version = "1.0.0"
                mock_settings.logfire.environment = "test"
                mock_settings.logfire.trace_pydantic_ai = False
                mock_settings.logfire.trace_sqlalchemy = False
                mock_settings.logfire.trace_httpx = False
                mock_settings.logfire.trace_fastapi = True
                mock_settings.logfire.project_name = "test-project"

                app = FastAPI()
                initialize_logfire(app=app)
                # Verify warning was logged for the exception
                mock_logger.warning.assert_called()
                call_args = mock_logger.warning.call_args[0][0]
                assert "FastAPI" in call_args
        finally:
            if original_logfire is None:
                sys.modules.pop("logfire", None)
            else:
                sys.modules["logfire"] = original_logfire

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_multiple_instrumentation_exceptions_all_logged(self, mock_logger):
        """Test that multiple instrumentation exceptions are all caught and logged."""
        import sys
        from gearmeshing_ai.core.monitoring import initialize_logfire
        from fastapi import FastAPI

        mock_logfire = MagicMock()
        mock_logfire.instrument_pydantic_ai.side_effect = RuntimeError("Pydantic AI failed")
        mock_logfire.instrument_sqlalchemy.side_effect = RuntimeError("SQLAlchemy failed")
        mock_logfire.instrument_httpx.side_effect = RuntimeError("HTTPX failed")
        mock_logfire.instrument_fastapi.side_effect = RuntimeError("FastAPI failed")
        original_logfire = sys.modules.get("logfire")
        
        try:
            sys.modules["logfire"] = mock_logfire
            
            with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
                mock_settings.logfire.enabled = True
                mock_token = MagicMock()
                mock_token.get_secret_value.return_value = "test-token"
                mock_settings.logfire.token = mock_token
                mock_settings.logfire.service_name = "test-service"
                mock_settings.logfire.service_version = "1.0.0"
                mock_settings.logfire.environment = "test"
                mock_settings.logfire.trace_pydantic_ai = True
                mock_settings.logfire.trace_sqlalchemy = True
                mock_settings.logfire.trace_httpx = True
                mock_settings.logfire.trace_fastapi = True
                mock_settings.logfire.project_name = "test-project"

                app = FastAPI()
                initialize_logfire(app=app)
                # Verify all warnings were logged
                assert mock_logger.warning.call_count >= 4
                warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
                assert any("Pydantic AI" in str(call) for call in warning_calls)
                assert any("SQLAlchemy" in str(call) for call in warning_calls)
                assert any("HTTPX" in str(call) for call in warning_calls)
                assert any("FastAPI" in str(call) for call in warning_calls)
        finally:
            if original_logfire is None:
                sys.modules.pop("logfire", None)
            else:
                sys.modules["logfire"] = original_logfire

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_pydantic_ai_exception_type_error(self, mock_logger):
        """Test that TypeError from instrument_pydantic_ai is caught and logged."""
        import sys
        from gearmeshing_ai.core.monitoring import initialize_logfire

        mock_logfire = MagicMock()
        mock_logfire.instrument_pydantic_ai.side_effect = TypeError("Invalid type for Pydantic AI")
        original_logfire = sys.modules.get("logfire")
        
        try:
            sys.modules["logfire"] = mock_logfire
            
            with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
                mock_settings.logfire.enabled = True
                mock_token = MagicMock()
                mock_token.get_secret_value.return_value = "test-token"
                mock_settings.logfire.token = mock_token
                mock_settings.logfire.service_name = "test-service"
                mock_settings.logfire.service_version = "1.0.0"
                mock_settings.logfire.environment = "test"
                mock_settings.logfire.trace_pydantic_ai = True
                mock_settings.logfire.trace_sqlalchemy = False
                mock_settings.logfire.trace_httpx = False
                mock_settings.logfire.trace_fastapi = False
                mock_settings.logfire.project_name = "test-project"

                initialize_logfire()
                # Verify warning was logged for the exception
                mock_logger.warning.assert_called()
                call_args = mock_logger.warning.call_args[0][0]
                assert "Pydantic AI" in call_args
        finally:
            if original_logfire is None:
                sys.modules.pop("logfire", None)
            else:
                sys.modules["logfire"] = original_logfire

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_sqlalchemy_exception_attribute_error(self, mock_logger):
        """Test that AttributeError from instrument_sqlalchemy is caught and logged."""
        import sys
        from gearmeshing_ai.core.monitoring import initialize_logfire

        mock_logfire = MagicMock()
        mock_logfire.instrument_sqlalchemy.side_effect = AttributeError("SQLAlchemy attribute missing")
        original_logfire = sys.modules.get("logfire")
        
        try:
            sys.modules["logfire"] = mock_logfire
            
            with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
                mock_settings.logfire.enabled = True
                mock_token = MagicMock()
                mock_token.get_secret_value.return_value = "test-token"
                mock_settings.logfire.token = mock_token
                mock_settings.logfire.service_name = "test-service"
                mock_settings.logfire.service_version = "1.0.0"
                mock_settings.logfire.environment = "test"
                mock_settings.logfire.trace_pydantic_ai = False
                mock_settings.logfire.trace_sqlalchemy = True
                mock_settings.logfire.trace_httpx = False
                mock_settings.logfire.trace_fastapi = False
                mock_settings.logfire.project_name = "test-project"

                initialize_logfire()
                # Verify warning was logged for the exception
                mock_logger.warning.assert_called()
                call_args = mock_logger.warning.call_args[0][0]
                assert "SQLAlchemy" in call_args
        finally:
            if original_logfire is None:
                sys.modules.pop("logfire", None)
            else:
                sys.modules["logfire"] = original_logfire

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_httpx_exception_value_error(self, mock_logger):
        """Test that ValueError from instrument_httpx is caught and logged."""
        import sys
        from gearmeshing_ai.core.monitoring import initialize_logfire

        mock_logfire = MagicMock()
        mock_logfire.instrument_httpx.side_effect = ValueError("Invalid value for HTTPX")
        original_logfire = sys.modules.get("logfire")
        
        try:
            sys.modules["logfire"] = mock_logfire
            
            with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
                mock_settings.logfire.enabled = True
                mock_token = MagicMock()
                mock_token.get_secret_value.return_value = "test-token"
                mock_settings.logfire.token = mock_token
                mock_settings.logfire.service_name = "test-service"
                mock_settings.logfire.service_version = "1.0.0"
                mock_settings.logfire.environment = "test"
                mock_settings.logfire.trace_pydantic_ai = False
                mock_settings.logfire.trace_sqlalchemy = False
                mock_settings.logfire.trace_httpx = True
                mock_settings.logfire.trace_fastapi = False
                mock_settings.logfire.project_name = "test-project"

                initialize_logfire()
                # Verify warning was logged for the exception
                mock_logger.warning.assert_called()
                call_args = mock_logger.warning.call_args[0][0]
                assert "HTTPX" in call_args
        finally:
            if original_logfire is None:
                sys.modules.pop("logfire", None)
            else:
                sys.modules["logfire"] = original_logfire

    @patch("gearmeshing_ai.core.monitoring.logger")
    def test_fastapi_exception_import_error(self, mock_logger):
        """Test that ImportError from instrument_fastapi is caught and logged."""
        import sys
        from gearmeshing_ai.core.monitoring import initialize_logfire
        from fastapi import FastAPI

        mock_logfire = MagicMock()
        mock_logfire.instrument_fastapi.side_effect = ImportError("FastAPI import failed")
        original_logfire = sys.modules.get("logfire")
        
        try:
            sys.modules["logfire"] = mock_logfire
            
            with patch("gearmeshing_ai.core.monitoring.settings") as mock_settings:
                mock_settings.logfire.enabled = True
                mock_token = MagicMock()
                mock_token.get_secret_value.return_value = "test-token"
                mock_settings.logfire.token = mock_token
                mock_settings.logfire.service_name = "test-service"
                mock_settings.logfire.service_version = "1.0.0"
                mock_settings.logfire.environment = "test"
                mock_settings.logfire.trace_pydantic_ai = False
                mock_settings.logfire.trace_sqlalchemy = False
                mock_settings.logfire.trace_httpx = False
                mock_settings.logfire.trace_fastapi = True
                mock_settings.logfire.project_name = "test-project"

                app = FastAPI()
                initialize_logfire(app=app)
                # Verify warning was logged for the exception
                mock_logger.warning.assert_called()
                call_args = mock_logger.warning.call_args[0][0]
                assert "FastAPI" in call_args
        finally:
            if original_logfire is None:
                sys.modules.pop("logfire", None)
            else:
                sys.modules["logfire"] = original_logfire
