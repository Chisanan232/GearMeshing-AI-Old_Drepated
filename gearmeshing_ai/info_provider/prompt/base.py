"""Core PromptProvider protocol used by the prompt subsystem.

This module defines :class:`PromptProvider`, a small, runtime-checkable
protocol that all prompt provider implementations are expected to follow.
Concrete providers live in :mod:`gearmeshing_ai.info_provider.prompt.provider`
and external/commercial providers are wired in via entry points.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class PromptProvider(Protocol):
    """Protocol for prompt providers used by GearMeshing-AI.

    The prompt provider abstraction sits between application code (API
    handlers, agents, workflows) and the concrete storage of prompt text. It
    is intentionally small so that different deployments can plug in
    alternative implementations without changing call sites.

    Key design points:

    - **Keyed access**: prompts are addressed by a stable string key
      (e.g. ``"pm/system"``) plus a locale and optional tenant id.
    - **Versioning**: ``version()`` returns a short, non-sensitive identifier
      that can be attached to run metadata for debugging and audits, without
      exposing prompt content.
    - **Reload**: ``refresh()`` is an optional hook for providers that support
      hot reload (e.g. ETag-based HTTP bundles). Implementations that do not
      support reload should implement it as a cheap no-op.

    Implementations must be side-effect free for ``get`` and should never log
    or expose prompt text through this interface.
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
