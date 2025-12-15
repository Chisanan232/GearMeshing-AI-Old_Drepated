from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from .provider import PromptProvider


class HotReloadWrapper(PromptProvider):
    """Lightweight, thread-safe wrapper that calls `refresh` periodically.

    The wrapper delegates `get`/`version` to the inner provider but ensures
    `refresh` is invoked at most once per `interval_seconds` across threads.
    """

    def __init__(
        self,
        inner: PromptProvider,
        *,
        interval_seconds: float = 60.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._inner = inner
        self._interval = max(interval_seconds, 0.0)
        self._logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._last_refresh: float = 0.0

    def _maybe_refresh(self) -> None:
        if self._interval <= 0:
            return
        now = time.monotonic()
        if now - self._last_refresh < self._interval:
            return
        with self._lock:
            # Double-check under lock
            now = time.monotonic()
            if now - self._last_refresh < self._interval:
                return
            try:
                self._inner.refresh()
            except Exception as exc:  # pragma: no cover - defensive logging
                # Never log prompt content; only metadata.
                self._logger.warning(
                    "Prompt provider refresh failed; provider=%s version=%s error=%s",
                    type(self._inner).__name__,
                    self._safe_version(),
                    type(exc).__name__,
                )
            finally:
                self._last_refresh = time.monotonic()

    def _safe_version(self) -> str:
        try:
            return self._inner.version()
        except Exception:  # pragma: no cover - defensive
            return "<unknown>"

    def get(self, name: str, locale: str = "en", tenant: Optional[str] = None) -> str:
        self._maybe_refresh()
        return self._inner.get(name, locale=locale, tenant=tenant)

    def version(self) -> str:
        self._maybe_refresh()
        return self._inner.version()

    def refresh(self) -> None:
        # Explicit refresh always bypasses throttling.
        try:
            self._inner.refresh()
        finally:
            self._last_refresh = time.monotonic()
