from __future__ import annotations

import logging
import threading
import time

from typing import Dict, Optional

from .base import PromptProvider

# Minimal, non-sensitive builtin prompts so OSS deployments can run in
# "basic" mode without any commercial bundle. These are intentionally
# terse and generic.


_BUILTIN_PROMPTS: Dict[str, Dict[str, str]] = {
    "en": {
        "pm/system": "You are a pragmatic product manager. Focus on clear user outcomes, constraints, and trade-offs.",
        "dev/system": "You are a senior software engineer. Prefer small, safe changes, explicit assumptions, and tests.",
        "qa/system": "You are a meticulous QA engineer. Think in terms of edge cases, regressions, and observability.",
    }
}


class BuiltinPromptProvider(PromptProvider):
    """In-repo builtin prompts for basic/local usage.

    The version is derived from a static identifier so that callers can record
    which builtin prompt set was used in a given run.
    """

    def __init__(self, *, prompts: Optional[Dict[str, Dict[str, str]]] = None, version_id: str = "builtin-v1") -> None:
        self._prompts = prompts or _BUILTIN_PROMPTS
        self._version = version_id

    def get(self, name: str, locale: str = "en", tenant: Optional[str] = None) -> str:  # noqa: ARG002
        bucket = self._prompts.get(locale) or {}
        try:
            return bucket[name]
        except KeyError as exc:  # pragma: no cover - trivial branch
            raise KeyError(f"prompt not found: locale={locale!r} name={name!r}") from exc

    def version(self) -> str:
        return self._version

    def refresh(self) -> None:  # noqa: D401 - trivial no-op
        """Builtin provider has no external state to refresh."""

        return None


class StackedPromptProvider(PromptProvider):
    """Chains providers with fallback semantics.

    Typically used as `StackedPromptProvider(commercial, builtin)` so that
    commercial prompts override builtin ones, while preserving a working
    baseline when commercial bundles are unavailable.
    """

    def __init__(self, primary: PromptProvider, fallback: PromptProvider) -> None:
        self._primary = primary
        self._fallback = fallback

    def get(self, name: str, locale: str = "en", tenant: Optional[str] = None) -> str:
        try:
            return self._primary.get(name, locale=locale, tenant=tenant)
        except KeyError:
            return self._fallback.get(name, locale=locale, tenant=tenant)

    def version(self) -> str:
        return f"stacked:{self._primary.version()}+{self._fallback.version()}"

    def refresh(self) -> None:
        # Refresh both in order; callers that wrap this in a hot-reload wrapper
        # can still treat refresh as best-effort.
        self._primary.refresh()
        self._fallback.refresh()


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
