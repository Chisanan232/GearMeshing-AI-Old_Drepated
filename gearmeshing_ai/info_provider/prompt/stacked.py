from __future__ import annotations

from typing import Optional

from .base import PromptProvider


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
