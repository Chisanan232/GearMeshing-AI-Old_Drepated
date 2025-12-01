from __future__ import annotations

from typing import Any, Optional


class GatewayApiError(Exception):
    def __init__(self, message: str, *, status_code: Optional[int] = None, details: Optional[Any] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.details = details


class GatewayServerNotFoundError(GatewayApiError):
    def __init__(self, server_id: str) -> None:
        super().__init__(f"Gateway server not found: {server_id}", status_code=404)
        self.server_id = server_id
