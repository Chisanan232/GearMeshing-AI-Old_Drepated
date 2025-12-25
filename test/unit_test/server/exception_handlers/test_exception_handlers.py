"""
Unit tests for server exception handlers.

Tests cover global exception handling with various error types,
edge cases, and error scenarios.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from gearmeshing_ai.server.exception_handlers import setup_exception_handlers
from gearmeshing_ai.server.exception_handlers.global_handler import (
    global_exception_handler,
)


class TestGlobalExceptionHandler:
    """Test suite for global exception handler."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock request object."""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/api/v1/test"
        request.query_params = {}
        request.client = Mock()
        request.client.host = "127.0.0.1"
        return request

    @pytest.mark.asyncio
    async def test_exception_handler_logs_error(self, mock_request):
        """Test that exception handler logs errors."""
        exc = ValueError("Test error")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger") as mock_logger:
            response = await global_exception_handler(mock_request, exc)

            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            assert "Unhandled exception" in call_args[0][0]
            # Check extra dict for error type
            assert call_args[1]["extra"]["error_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_exception_handler_returns_500_status(self, mock_request):
        """Test that exception handler returns 500 status code."""
        exc = RuntimeError("Test error")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger"):
            response = await global_exception_handler(mock_request, exc)

            assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_exception_handler_returns_json_response(self, mock_request):
        """Test that exception handler returns JSON response."""
        exc = RuntimeError("Test error")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger"):
            response = await global_exception_handler(mock_request, exc)

            assert isinstance(response, JSONResponse)

    @pytest.mark.asyncio
    async def test_exception_handler_includes_error_id(self, mock_request):
        """Test that exception handler includes error ID in response."""
        exc = RuntimeError("Test error")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger"):
            response = await global_exception_handler(mock_request, exc)

            # Parse the response body
            import json

            body = json.loads(response.body.decode())
            assert "error_id" in body
            assert isinstance(body["error_id"], int)

    @pytest.mark.asyncio
    async def test_exception_handler_includes_error_type(self, mock_request):
        """Test that exception handler includes error type in response."""
        exc = ValueError("Test error")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger"):
            response = await global_exception_handler(mock_request, exc)

            import json

            body = json.loads(response.body.decode())
            assert "error_type" in body
            assert body["error_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_exception_handler_includes_detail_message(self, mock_request):
        """Test that exception handler includes detail message in response."""
        exc = RuntimeError("Test error")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger"):
            response = await global_exception_handler(mock_request, exc)

            import json

            body = json.loads(response.body.decode())
            assert "detail" in body
            assert body["detail"] == "Internal server error"

    @pytest.mark.asyncio
    async def test_exception_handler_logs_request_method(self, mock_request):
        """Test that exception handler logs request method."""
        exc = RuntimeError("Test error")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger") as mock_logger:
            await global_exception_handler(mock_request, exc)

            call_args = mock_logger.error.call_args
            extra = call_args[1]["extra"]
            assert extra["method"] == "GET"

    @pytest.mark.asyncio
    async def test_exception_handler_logs_request_path(self, mock_request):
        """Test that exception handler logs request path."""
        exc = RuntimeError("Test error")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger") as mock_logger:
            await global_exception_handler(mock_request, exc)

            call_args = mock_logger.error.call_args
            extra = call_args[1]["extra"]
            assert extra["path"] == "/api/v1/test"

    @pytest.mark.asyncio
    async def test_exception_handler_logs_client_ip(self, mock_request):
        """Test that exception handler logs client IP."""
        exc = RuntimeError("Test error")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger") as mock_logger:
            await global_exception_handler(mock_request, exc)

            call_args = mock_logger.error.call_args
            extra = call_args[1]["extra"]
            assert extra["client"] == "127.0.0.1"

    @pytest.mark.asyncio
    async def test_exception_handler_logs_error_type(self, mock_request):
        """Test that exception handler logs error type."""
        exc = TypeError("Test error")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger") as mock_logger:
            await global_exception_handler(mock_request, exc)

            call_args = mock_logger.error.call_args
            extra = call_args[1]["extra"]
            assert extra["error_type"] == "TypeError"

    @pytest.mark.asyncio
    async def test_exception_handler_logs_traceback(self, mock_request):
        """Test that exception handler logs traceback."""
        exc = RuntimeError("Test error")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger") as mock_logger:
            await global_exception_handler(mock_request, exc)

            call_args = mock_logger.error.call_args
            extra = call_args[1]["extra"]
            assert "traceback" in extra
            assert isinstance(extra["traceback"], str)

    @pytest.mark.asyncio
    async def test_exception_handler_logs_with_exc_info(self, mock_request):
        """Test that exception handler logs with exc_info=True."""
        exc = RuntimeError("Test error")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger") as mock_logger:
            await global_exception_handler(mock_request, exc)

            call_args = mock_logger.error.call_args
            assert call_args[1]["exc_info"] is True

    @pytest.mark.asyncio
    async def test_exception_handler_handles_missing_client(self, mock_request):
        """Test that exception handler handles missing client info."""
        mock_request.client = None
        exc = RuntimeError("Test error")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger") as mock_logger:
            response = await global_exception_handler(mock_request, exc)

            call_args = mock_logger.error.call_args
            extra = call_args[1]["extra"]
            assert extra["client"] == "unknown"

    @pytest.mark.asyncio
    async def test_exception_handler_handles_empty_query_params(self, mock_request):
        """Test that exception handler handles empty query parameters."""
        mock_request.query_params = {}
        exc = RuntimeError("Test error")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger") as mock_logger:
            await global_exception_handler(mock_request, exc)

            call_args = mock_logger.error.call_args
            extra = call_args[1]["extra"]
            assert extra["query_params"] == {}

    @pytest.mark.asyncio
    async def test_exception_handler_handles_query_params(self, mock_request):
        """Test that exception handler logs query parameters."""
        mock_request.query_params = {"key": "value", "foo": "bar"}
        exc = RuntimeError("Test error")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger") as mock_logger:
            await global_exception_handler(mock_request, exc)

            call_args = mock_logger.error.call_args
            extra = call_args[1]["extra"]
            assert "query_params" in extra

    @pytest.mark.asyncio
    async def test_exception_handler_handles_different_exception_types(self, mock_request):
        """Test that exception handler handles different exception types."""
        exceptions = [
            ValueError("value error"),
            TypeError("type error"),
            RuntimeError("runtime error"),
            KeyError("key error"),
            AttributeError("attribute error"),
            IndexError("index error"),
        ]

        for exc in exceptions:
            with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger") as mock_logger:
                response = await global_exception_handler(mock_request, exc)

                assert response.status_code == 500
                import json

                body = json.loads(response.body.decode())
                assert body["error_type"] == type(exc).__name__

    @pytest.mark.asyncio
    async def test_exception_handler_error_id_is_unique(self, mock_request):
        """Test that each exception gets a unique error ID."""
        exc1 = RuntimeError("Error 1")
        exc2 = RuntimeError("Error 2")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger"):
            response1 = await global_exception_handler(mock_request, exc1)
            response2 = await global_exception_handler(mock_request, exc2)

            import json

            body1 = json.loads(response1.body.decode())
            body2 = json.loads(response2.body.decode())

            # Error IDs should be different (based on object id)
            assert body1["error_id"] != body2["error_id"]

    @pytest.mark.asyncio
    async def test_exception_handler_logs_error_message(self, mock_request):
        """Test that exception handler logs the error message."""
        error_msg = "This is a specific error message"
        exc = RuntimeError(error_msg)

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger") as mock_logger:
            await global_exception_handler(mock_request, exc)

            call_args = mock_logger.error.call_args
            assert error_msg in call_args[0][0]


class TestSetupExceptionHandlers:
    """Test suite for setup_exception_handlers function."""

    def test_setup_exception_handlers_registers_handler(self):
        """Test that setup_exception_handlers registers the handler."""
        app = FastAPI()

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger"):
            setup_exception_handlers(app)

            # Check that exception handler is registered
            assert Exception in app.exception_handlers

    def test_setup_exception_handlers_logs_debug_message(self):
        """Test that setup_exception_handlers logs a debug message."""
        app = FastAPI()

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger") as mock_logger:
            setup_exception_handlers(app)

            mock_logger.debug.assert_called_once()
            call_args = mock_logger.debug.call_args
            assert "Exception handlers registered" in call_args[0][0]

    def test_setup_exception_handlers_with_existing_handlers(self):
        """Test that setup_exception_handlers works with existing handlers."""
        app = FastAPI()

        # Add a custom handler first
        @app.exception_handler(ValueError)
        async def value_error_handler(request, exc):
            return JSONResponse(status_code=400, content={"error": "value error"})

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger"):
            setup_exception_handlers(app)

            # Both handlers should be registered
            assert ValueError in app.exception_handlers
            assert Exception in app.exception_handlers


class TestExceptionHandlerWithDifferentErrors:
    """Test exception handler with different exception types."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock request object."""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/api/v1/test"
        request.query_params = {}
        request.client = Mock()
        request.client.host = "127.0.0.1"
        return request

    @pytest.mark.asyncio
    async def test_exception_handler_with_value_error(self, mock_request):
        """Test exception handler with ValueError."""
        exc = ValueError("Invalid value")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger"):
            response = await global_exception_handler(mock_request, exc)

            assert response.status_code == 500
            import json

            data = json.loads(response.body.decode())
            assert data["error_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_exception_handler_with_type_error(self, mock_request):
        """Test exception handler with TypeError."""
        exc = TypeError("Type mismatch")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger"):
            response = await global_exception_handler(mock_request, exc)

            assert response.status_code == 500
            import json

            data = json.loads(response.body.decode())
            assert data["error_type"] == "TypeError"

    @pytest.mark.asyncio
    async def test_exception_handler_with_key_error(self, mock_request):
        """Test exception handler with KeyError."""
        exc = KeyError("missing_key")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger"):
            response = await global_exception_handler(mock_request, exc)

            assert response.status_code == 500
            import json

            data = json.loads(response.body.decode())
            assert data["error_type"] == "KeyError"

    @pytest.mark.asyncio
    async def test_exception_handler_with_index_error(self, mock_request):
        """Test exception handler with IndexError."""
        exc = IndexError("list index out of range")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger"):
            response = await global_exception_handler(mock_request, exc)

            assert response.status_code == 500
            import json

            data = json.loads(response.body.decode())
            assert data["error_type"] == "IndexError"

    @pytest.mark.asyncio
    async def test_exception_handler_with_attribute_error(self, mock_request):
        """Test exception handler with AttributeError."""
        exc = AttributeError("missing_attribute")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger"):
            response = await global_exception_handler(mock_request, exc)

            assert response.status_code == 500
            import json

            data = json.loads(response.body.decode())
            assert data["error_type"] == "AttributeError"

    @pytest.mark.asyncio
    async def test_exception_handler_preserves_error_details(self, mock_request):
        """Test that exception handler preserves error details."""
        exc = RuntimeError("Specific error message")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger"):
            response = await global_exception_handler(mock_request, exc)

            import json

            data = json.loads(response.body.decode())
            assert "error_id" in data
            assert "error_type" in data
            assert "detail" in data

    @pytest.mark.asyncio
    async def test_exception_handler_response_is_json(self, mock_request):
        """Test that exception handler returns valid JSON."""
        exc = RuntimeError("Test error")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger"):
            response = await global_exception_handler(mock_request, exc)

            # Should be able to parse as JSON
            import json

            data = json.loads(response.body.decode())
            assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_exception_handler_with_nested_exception(self, mock_request):
        """Test exception handler with nested exceptions."""
        try:
            raise ValueError("Inner error")
        except ValueError as e:
            exc = RuntimeError("Outer error")

        with patch("gearmeshing_ai.server.exception_handlers.global_handler.logger"):
            response = await global_exception_handler(mock_request, exc)

            assert response.status_code == 500
            import json

            data = json.loads(response.body.decode())
            assert data["error_type"] == "RuntimeError"
