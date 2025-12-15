"""Error types specific to the Gateway API layer.

Purpose:
- Provide typed exceptions thrown by `GatewayApiClient` and consumers of the
  Gateway REST API.
- Expose HTTP-oriented context (e.g., status code, error body) for diagnosis.

Usage:
- Catch `GatewayApiError` for general failures and inspect `status_code` or
  `details`.
- Catch `GatewayServerNotFoundError` when a server lookup by ID returns 404.
"""

from __future__ import annotations

from typing import Any, Optional


class GatewayApiError(Exception):
    """Base error for Gateway API failures.

    Args:
        message: Human-readable error description.
        status_code: Optional HTTP status code associated with the failure.
        details: Optional structured payload from the server (e.g., JSON body).
    """

    def __init__(self, message: str, *, status_code: Optional[int] = None, details: Optional[Any] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.details = details
