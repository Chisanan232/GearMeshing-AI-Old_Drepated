"""
Unit tests for Logfire middleware.

This test suite covers:
- Request/response processing
- Performance metrics collection
- Error handling and exception tracking
- Slow request detection
- Request context propagation
- Header injection
"""

import pytest
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi import FastAPI, Request
from starlette.responses import Response
from starlette.testclient import TestClient


class TestLogfireMiddlewareDispatch:
    """Test LogfireMiddleware.dispatch method."""

    @pytest.mark.asyncio
    async def test_middleware_processes_successful_request(self):
        """Test that middleware processes successful requests."""
        from gearmeshing_ai.server.middleware.logfire_middleware import LogfireMiddleware

        # Create mock request and call_next
        mock_request = AsyncMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/api/v1/test"
        mock_request.url.query = ""
        mock_request.state = MagicMock()

        mock_response = Response(content="test", status_code=200)

        async def mock_call_next(request):
            return mock_response

        middleware = LogfireMiddleware(app=AsyncMock())

        with patch("gearmeshing_ai.server.middleware.logfire_middleware.log_api_request") as mock_log:
            response = await middleware.dispatch(mock_request, mock_call_next)

            assert response.status_code == 200
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[1]["method"] == "GET"
            assert call_args[1]["path"] == "/api/v1/test"
            assert call_args[1]["status_code"] == 200

    @pytest.mark.asyncio
    async def test_middleware_measures_request_duration(self):
        """Test that middleware measures request duration."""
        from gearmeshing_ai.server.middleware.logfire_middleware import LogfireMiddleware

        mock_request = AsyncMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/runs"
        mock_request.url.query = ""
        mock_request.state = MagicMock()

        mock_response = Response(content="test", status_code=201)

        async def mock_call_next(request):
            # Simulate some processing time
            await AsyncMock()()
            return mock_response

        middleware = LogfireMiddleware(app=AsyncMock())

        with patch("gearmeshing_ai.server.middleware.logfire_middleware.log_api_request") as mock_log:
            response = await middleware.dispatch(mock_request, mock_call_next)

            mock_log.assert_called_once()
            call_args = mock_log.call_args
            # Duration should be a positive number
            assert call_args[1]["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_middleware_adds_process_time_header(self):
        """Test that middleware adds X-Process-Time header."""
        from gearmeshing_ai.server.middleware.logfire_middleware import LogfireMiddleware

        mock_request = AsyncMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/api/v1/test"
        mock_request.url.query = ""
        mock_request.state = MagicMock()

        mock_response = Response(content="test", status_code=200)

        async def mock_call_next(request):
            return mock_response

        middleware = LogfireMiddleware(app=AsyncMock())

        with patch("gearmeshing_ai.server.middleware.logfire_middleware.log_api_request"):
            response = await middleware.dispatch(mock_request, mock_call_next)

            assert "X-Process-Time" in response.headers
            # Should be a numeric string
            assert float(response.headers["X-Process-Time"]) >= 0

    @pytest.mark.asyncio
    async def test_middleware_detects_slow_requests(self):
        """Test that middleware detects and logs slow requests."""
        from gearmeshing_ai.server.middleware.logfire_middleware import LogfireMiddleware

        mock_request = AsyncMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/api/v1/slow"
        mock_request.url.query = ""
        mock_request.state = MagicMock()

        mock_response = Response(content="test", status_code=200)

        async def mock_call_next(request):
            # Simulate slow request (>1 second)
            await AsyncMock()()
            return mock_response

        middleware = LogfireMiddleware(app=AsyncMock())

        with patch("gearmeshing_ai.server.middleware.logfire_middleware.log_api_request"):
            with patch("gearmeshing_ai.server.middleware.logfire_middleware.logger") as mock_logger:
                with patch("gearmeshing_ai.server.middleware.logfire_middleware.time.time") as mock_time:
                    # Simulate 1.5 second duration
                    mock_time.side_effect = [0, 1.5]

                    response = await middleware.dispatch(mock_request, mock_call_next)

                    # Should log warning for slow request
                    mock_logger.warning.assert_called_once()
                    call_args = mock_logger.warning.call_args
                    assert "Slow API request" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_middleware_handles_request_exception(self):
        """Test that middleware handles exceptions during request processing."""
        from gearmeshing_ai.server.middleware.logfire_middleware import LogfireMiddleware

        mock_request = AsyncMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/api/v1/error"
        mock_request.url.query = ""
        mock_request.state = MagicMock()

        async def mock_call_next(request):
            raise ValueError("Test error")

        middleware = LogfireMiddleware(app=AsyncMock())

        with patch("gearmeshing_ai.server.middleware.logfire_middleware.log_api_request") as mock_log:
            with patch("gearmeshing_ai.server.middleware.logfire_middleware.logger") as mock_logger:
                with pytest.raises(ValueError):
                    await middleware.dispatch(mock_request, mock_call_next)

                # Should log error
                mock_logger.error.assert_called_once()
                # Should still log API request with 500 status
                mock_log.assert_called_once()
                call_args = mock_log.call_args
                assert call_args[1]["status_code"] == 500

    @pytest.mark.asyncio
    async def test_middleware_stores_request_context(self):
        """Test that middleware stores request context in request.state."""
        from gearmeshing_ai.server.middleware.logfire_middleware import LogfireMiddleware

        mock_request = AsyncMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/runs"
        mock_request.url.query = "?tenant_id=123"
        mock_request.state = MagicMock()

        mock_response = Response(content="test", status_code=201)

        async def mock_call_next(request):
            return mock_response

        middleware = LogfireMiddleware(app=AsyncMock())

        with patch("gearmeshing_ai.server.middleware.logfire_middleware.log_api_request"):
            response = await middleware.dispatch(mock_request, mock_call_next)

            # Check that request state was populated
            assert mock_request.state.start_time is not None
            assert mock_request.state.method == "POST"
            assert mock_request.state.path == "/api/v1/runs"

    @pytest.mark.asyncio
    async def test_middleware_logs_different_http_methods(self):
        """Test that middleware logs requests with different HTTP methods."""
        from gearmeshing_ai.server.middleware.logfire_middleware import LogfireMiddleware

        methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]

        for method in methods:
            mock_request = AsyncMock(spec=Request)
            mock_request.method = method
            mock_request.url.path = "/api/v1/test"
            mock_request.url.query = ""
            mock_request.state = MagicMock()

            mock_response = Response(content="test", status_code=200)

            async def mock_call_next(request):
                return mock_response

            middleware = LogfireMiddleware(app=AsyncMock())

            with patch("gearmeshing_ai.server.middleware.logfire_middleware.log_api_request") as mock_log:
                response = await middleware.dispatch(mock_request, mock_call_next)

                call_args = mock_log.call_args
                assert call_args[1]["method"] == method

    @pytest.mark.asyncio
    async def test_middleware_logs_different_status_codes(self):
        """Test that middleware logs requests with different status codes."""
        from gearmeshing_ai.server.middleware.logfire_middleware import LogfireMiddleware

        status_codes = [200, 201, 204, 400, 401, 403, 404, 500, 502, 503]

        for status_code in status_codes:
            mock_request = AsyncMock(spec=Request)
            mock_request.method = "GET"
            mock_request.url.path = "/api/v1/test"
            mock_request.url.query = ""
            mock_request.state = MagicMock()

            mock_response = Response(content="test", status_code=status_code)

            async def mock_call_next(request):
                return mock_response

            middleware = LogfireMiddleware(app=AsyncMock())

            with patch("gearmeshing_ai.server.middleware.logfire_middleware.log_api_request") as mock_log:
                response = await middleware.dispatch(mock_request, mock_call_next)

                call_args = mock_log.call_args
                assert call_args[1]["status_code"] == status_code

    @pytest.mark.asyncio
    async def test_middleware_handles_exception_with_logging(self):
        """Test that middleware logs exceptions with context."""
        from gearmeshing_ai.server.middleware.logfire_middleware import LogfireMiddleware

        mock_request = AsyncMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/runs"
        mock_request.url.query = ""
        mock_request.state = MagicMock()

        test_error = RuntimeError("Database connection failed")

        async def mock_call_next(request):
            raise test_error

        middleware = LogfireMiddleware(app=AsyncMock())

        with patch("gearmeshing_ai.server.middleware.logfire_middleware.log_api_request"):
            with patch("gearmeshing_ai.server.middleware.logfire_middleware.logger") as mock_logger:
                with pytest.raises(RuntimeError):
                    await middleware.dispatch(mock_request, mock_call_next)

                # Should log error with exc_info
                mock_logger.error.assert_called_once()
                call_args = mock_logger.error.call_args
                assert call_args[1]["exc_info"] is True

    @pytest.mark.asyncio
    async def test_middleware_preserves_response_content(self):
        """Test that middleware preserves response content."""
        from gearmeshing_ai.server.middleware.logfire_middleware import LogfireMiddleware

        mock_request = AsyncMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/api/v1/test"
        mock_request.url.query = ""
        mock_request.state = MagicMock()

        expected_content = b"test response content"
        mock_response = Response(content=expected_content, status_code=200)

        async def mock_call_next(request):
            return mock_response

        middleware = LogfireMiddleware(app=AsyncMock())

        with patch("gearmeshing_ai.server.middleware.logfire_middleware.log_api_request"):
            response = await middleware.dispatch(mock_request, mock_call_next)

            assert response.body == expected_content


class TestLogfireMiddlewareIntegration:
    """Integration tests for LogfireMiddleware with FastAPI."""

    def test_middleware_instantiation(self):
        """Test middleware can be instantiated with FastAPI app."""
        from gearmeshing_ai.server.middleware.logfire_middleware import LogfireMiddleware

        app = FastAPI()
        middleware = LogfireMiddleware(app=app)
        
        assert middleware is not None
        assert middleware.app == app


class TestLogfireMiddlewareEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_middleware_with_very_fast_request(self):
        """Test middleware with very fast request (near-zero duration)."""
        from gearmeshing_ai.server.middleware.logfire_middleware import LogfireMiddleware

        mock_request = AsyncMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/api/v1/fast"
        mock_request.url.query = ""
        mock_request.state = MagicMock()

        mock_response = Response(content="test", status_code=200)

        async def mock_call_next(request):
            return mock_response

        middleware = LogfireMiddleware(app=AsyncMock())

        with patch("gearmeshing_ai.server.middleware.logfire_middleware.log_api_request") as mock_log:
            response = await middleware.dispatch(mock_request, mock_call_next)

            mock_log.assert_called_once()
            call_args = mock_log.call_args
            # Duration should be >= 0
            assert call_args[1]["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_middleware_with_empty_path(self):
        """Test middleware with empty path."""
        from gearmeshing_ai.server.middleware.logfire_middleware import LogfireMiddleware

        mock_request = AsyncMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/"
        mock_request.url.query = ""
        mock_request.state = MagicMock()

        mock_response = Response(content="test", status_code=200)

        async def mock_call_next(request):
            return mock_response

        middleware = LogfireMiddleware(app=AsyncMock())

        with patch("gearmeshing_ai.server.middleware.logfire_middleware.log_api_request") as mock_log:
            response = await middleware.dispatch(mock_request, mock_call_next)

            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[1]["path"] == "/"

    @pytest.mark.asyncio
    async def test_middleware_with_special_characters_in_path(self):
        """Test middleware with special characters in path."""
        from gearmeshing_ai.server.middleware.logfire_middleware import LogfireMiddleware

        mock_request = AsyncMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/api/v1/test-endpoint_123"
        mock_request.url.query = ""
        mock_request.state = MagicMock()

        mock_response = Response(content="test", status_code=200)

        async def mock_call_next(request):
            return mock_response

        middleware = LogfireMiddleware(app=AsyncMock())

        with patch("gearmeshing_ai.server.middleware.logfire_middleware.log_api_request") as mock_log:
            response = await middleware.dispatch(mock_request, mock_call_next)

            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[1]["path"] == "/api/v1/test-endpoint_123"

    @pytest.mark.asyncio
    async def test_middleware_with_large_response(self):
        """Test middleware with large response body."""
        from gearmeshing_ai.server.middleware.logfire_middleware import LogfireMiddleware

        mock_request = AsyncMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/api/v1/large"
        mock_request.url.query = ""
        mock_request.state = MagicMock()

        # Create large response
        large_content = b"x" * 1000000  # 1MB
        mock_response = Response(content=large_content, status_code=200)

        async def mock_call_next(request):
            return mock_response

        middleware = LogfireMiddleware(app=AsyncMock())

        with patch("gearmeshing_ai.server.middleware.logfire_middleware.log_api_request") as mock_log:
            response = await middleware.dispatch(mock_request, mock_call_next)

            mock_log.assert_called_once()
            assert response.body == large_content
