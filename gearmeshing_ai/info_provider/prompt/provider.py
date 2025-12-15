from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class PromptProvider(Protocol):
    """Protocol for prompt providers.

    Implementations load prompts by stable key and locale and expose a
    lightweight `version` identifier. `refresh` is optional and should be a
    cheap no-op when unsupported.
    """

    def get(self, name: str, locale: str = "en", tenant: Optional[str] = None) -> str:
        """Return the prompt text for `name` in `locale`.

        Implementations should raise `KeyError` when the prompt is unknown.
        """

        ...

    def version(self) -> str:
        """Return a non-sensitive version identifier for this provider."""

        ...

    def refresh(self) -> None:
        """Refresh internal caches if supported (optional)."""

        ...
