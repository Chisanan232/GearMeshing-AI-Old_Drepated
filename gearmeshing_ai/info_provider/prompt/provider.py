"""Concrete prompt provider implementations.

This module contains the built-in, non-sensitive prompt provider used by
open-source and local deployments, as well as composition helpers:

* :class:`BuiltinPromptProvider` – in-process, dictionary-backed provider
  with a small set of generic prompts.
* :class:`StackedPromptProvider` – combines two providers with fallback
  semantics (typically commercial over builtin).
* :class:`HotReloadWrapper` – wraps another provider to call ``refresh``
  periodically in a thread-safe way.

Higher-level code is expected to access these via the
``gearmeshing_ai.info_provider.prompt`` facade.
"""

from __future__ import annotations

import logging
import threading
import time

from typing import Dict, Optional

from .base import PromptProvider

# Minimal, non-sensitive builtin prompts so OSS deployments can run in
# "basic" mode without any commercial bundle. These are intentionally
# terse and generic. In commercial deployments these are usually
# overshadowed by a richer provider via StackedPromptProvider.


_BUILTIN_PROMPTS: Dict[str, Dict[str, str]] = {
    "en": {
        "pm/system": "You are a pragmatic product manager. Focus on clear user outcomes, constraints, and trade-offs.",
        "dev/system": "You are a senior software engineer. Prefer small, safe changes, explicit assumptions, and tests.",
        "qa/system": "You are a meticulous QA engineer. Think in terms of edge cases, regressions, and observability.",
    }
}


class BuiltinPromptProvider(PromptProvider):
    """In-repo builtin prompts for basic/local usage.

    This provider keeps a small dictionary of prompts in memory, keyed by
    locale and prompt name. It is intended to be safe to ship in the
    open-source repository:

    - No tenant-specific content.
    - No secrets or proprietary wording.

    The :meth:`version` string is a lightweight identifier (e.g. a migration
    label) that can be recorded alongside runs, allowing operators to
    correlate behavior with a given builtin prompt set.
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

    This helper composes two :class:`PromptProvider` implementations into a
    single provider. Resolution semantics:

    - ``get`` first asks ``primary``; if it raises ``KeyError`` the
      ``fallback`` provider is queried.
    - ``version`` returns a combined identifier
      (``"stacked:<primary>+<fallback>"``) so tooling can understand which
      providers contributed.
    - ``refresh`` forwards the call to both providers in order.

    Typical usage pattern::

        commercial = CommercialPromptProvider(...)
        builtin = BuiltinPromptProvider()
        provider = StackedPromptProvider(commercial, builtin)

    This ensures that commercial prompts take precedence when available,
    while still providing a fully functional fallback when the commercial
    provider is misconfigured or missing a particular key.
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
    """Lightweight, thread-safe wrapper that periodically refreshes a provider.

    Hot reload is useful for commercial providers that fetch prompt bundles
    from a remote source (e.g. HTTP+ETag, object storage, database) and need
    to pick up new versions without restarting the process.

    Design notes:

    - The wrapper calls ``inner.refresh()`` at most once per
      ``interval_seconds`` across all threads and requests.
    - ``get`` and ``version`` both trigger a background refresh check before
      delegating to the underlying provider.
    - Errors from ``refresh`` are logged in a redacted form; prompt text and
      error messages are never logged.

    This wrapper does not itself implement ETag or integrity validation; that
    is the responsibility of the underlying provider's :meth:`refresh`
    implementation.
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
        """Trigger a refresh on the underlying provider if the interval elapsed.

        This method is safe to call on every "read" operation. It uses
        ``time.monotonic`` and a double-checked lock to avoid calling
        :meth:`PromptProvider.refresh` more than necessary, and to ensure that
        only one thread performs the refresh at a time.
        """
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
        """Return the inner provider's version, guarding against errors.

        If :meth:`PromptProvider.version` raises for any reason, this returns
        ``"<unknown>"`` so that logs remain fully redacted while still
        conveying that the version was unavailable.
        """
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
        """Explicitly refresh the inner provider, bypassing throttling.

        Callers that expose administrative endpoints or management commands
        may want to force a refresh immediately (for example after a config
        change) rather than waiting for the interval-based background logic.
        This helper always calls :meth:`PromptProvider.refresh` and resets the
        internal timer so subsequent calls are throttled again.
        """
        try:
            self._inner.refresh()
        finally:
            self._last_refresh = time.monotonic()
