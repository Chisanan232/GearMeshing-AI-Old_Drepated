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
    """
    In-repo builtin prompts for basic/local usage.

    This provider keeps a small dictionary of prompts in memory, keyed by
    locale and prompt name. It serves as the baseline implementation for
    open-source usage and local development.

    Characteristics:
    - No tenant-specific content.
    - No secrets or proprietary wording.
    - Fast, in-memory lookup.

    The :meth:`version` string is a lightweight identifier (e.g. "builtin-v1")
    used for tracking which prompt set was active during a run.
    """

    def __init__(self, *, prompts: Optional[Dict[str, Dict[str, str]]] = None, version_id: str = "builtin-v1") -> None:
        """
        Initialize the builtin provider.

        Args:
            prompts: Optional dictionary of prompts to override defaults. Structure: {locale: {key: text}}.
            version_id: Identifier string for this prompt set version.
        """
        self._prompts = prompts or _BUILTIN_PROMPTS
        self._version = version_id

    def get(self, name: str, locale: str = "en", tenant: Optional[str] = None) -> str:  # noqa: ARG002
        """
        Retrieve a prompt text.

        Args:
            name: The prompt key (e.g., 'dev/system').
            locale: The requested locale (default: 'en').
            tenant: Ignored by this provider.

        Returns:
            The prompt text string.

        Raises:
            KeyError: If the prompt name is not found for the given locale.
        """
        bucket = self._prompts.get(locale) or {}
        try:
            return bucket[name]
        except KeyError as exc:  # pragma: no cover - trivial branch
            raise KeyError(f"prompt not found: locale={locale!r} name={name!r}") from exc

    def version(self) -> str:
        """Return the version identifier of the builtin prompts."""
        return self._version

    def refresh(self) -> None:  # noqa: D401 - trivial no-op
        """Builtin provider has no external state to refresh (no-op)."""
        return None


class StackedPromptProvider(PromptProvider):
    """
    Chains two providers with fallback semantics.

    This helper composes a primary (e.g., commercial) and a fallback (e.g., builtin)
    provider. It allows seamless degradation if the primary provider is missing keys
    or fails.

    Behavior:
    - ``get`` queries the primary first. If it raises ``KeyError``, the fallback is queried.
    - ``version`` returns a combined identifier ``"stacked:<primary>+<fallback>"``.
    - ``refresh`` refreshes both providers in order.
    """

    def __init__(self, primary: PromptProvider, fallback: PromptProvider) -> None:
        """
        Initialize the stacked provider.

        Args:
            primary: The preferred provider to query first.
            fallback: The backup provider to query on cache miss.
        """
        self._primary = primary
        self._fallback = fallback

    def get(self, name: str, locale: str = "en", tenant: Optional[str] = None) -> str:
        """
        Retrieve a prompt, trying primary then fallback.

        Args:
            name: Prompt key.
            locale: Language code.
            tenant: Tenant identifier.

        Returns:
            The prompt text.

        Raises:
            KeyError: If neither provider has the requested prompt.
        """
        try:
            return self._primary.get(name, locale=locale, tenant=tenant)
        except KeyError:
            return self._fallback.get(name, locale=locale, tenant=tenant)

    def version(self) -> str:
        """Return a composite version string reflecting both providers."""
        return f"stacked:{self._primary.version()}+{self._fallback.version()}"

    def refresh(self) -> None:
        """Refresh both the primary and fallback providers."""
        # Refresh both in order; callers that wrap this in a hot-reload wrapper
        # can still treat refresh as best-effort.
        self._primary.refresh()
        self._fallback.refresh()


class HotReloadWrapper(PromptProvider):
    """
    Lightweight, thread-safe wrapper that periodically refreshes a provider.

    Designed for providers that fetch prompts from remote sources (HTTP, S3, DB).
    It ensures ``refresh()`` is called at most once per ``interval_seconds``, regardless
    of read concurrency.

    Attributes:
        inner: The underlying provider instance.
        interval: Minimum seconds between refresh attempts.
    """

    def __init__(
        self,
        inner: PromptProvider,
        *,
        interval_seconds: float = 60.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the hot-reload wrapper.

        Args:
            inner: The provider to wrap and refresh.
            interval_seconds: Minimum time between refresh calls (default: 60s).
            logger: Optional logger for refresh errors (redacted).
        """
        self._inner = inner
        self._interval = max(interval_seconds, 0.0)
        self._logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._last_refresh: float = 0.0

    def _maybe_refresh(self) -> None:
        """
        Trigger a refresh on the underlying provider if the interval elapsed.

        This method uses double-checked locking to ensure thread safety and minimize contention.
        Errors during refresh are logged but swallowed to prevent read failures.
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
        """
        Return the inner provider's version, guarding against errors.

        Used for safe logging in case the inner provider is unstable.
        """
        try:
            return self._inner.version()
        except Exception:  # pragma: no cover - defensive
            return "<unknown>"

    def get(self, name: str, locale: str = "en", tenant: Optional[str] = None) -> str:
        """Retrieve a prompt, triggering a potential background refresh first."""
        self._maybe_refresh()
        return self._inner.get(name, locale=locale, tenant=tenant)

    def version(self) -> str:
        """Retrieve version, triggering a potential background refresh first."""
        self._maybe_refresh()
        return self._inner.version()

    def refresh(self) -> None:
        """
        Explicitly refresh the inner provider, bypassing throttling.

        Useful for administrative force-reloads. Resets the throttle timer.
        """
        try:
            self._inner.refresh()
        finally:
            self._last_refresh = time.monotonic()
