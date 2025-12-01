from __future__ import annotations

from typing import Any, Protocol


class HttpStreamTransport(Protocol):
    def send_request(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class SseTransport(Protocol):
    async def connect(self, path: str) -> None: ...
    async def close(self) -> None: ...


class StdioTransport(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
