"""Provider loader utilities.

This module contains the logic that maps configuration (primarily the
``GEARMESH_PROMPT_PROVIDER`` environment variable) to a concrete
``PromptProvider`` implementation.

Two sources of providers are supported:

* The in-repo :class:`BuiltinPromptProvider`, always available as a safe
  fallback.
* External packages that register a factory via the
  ``gearmesh.prompt_providers`` entry-point group. This is how commercial
  prompt bundles are plugged into the system without modifying the OSS
  codebase.

The loader is deliberately defensive: failures to resolve or import a
commercial provider never prevent the system from starting; it simply falls
back to the builtin provider and emits a redacted warning.
"""

from __future__ import annotations

import logging
from importlib import metadata
from typing import Iterable

from .base import PromptProvider
from .provider import BuiltinPromptProvider

_LOGGER = logging.getLogger(__name__)


def _iter_entry_points(group: str) -> Iterable[metadata.EntryPoint]:
    """Return entry points for ``group`` across Python versions.

    ``importlib.metadata.entry_points`` changed shape across Python versions
    (from a simple mapping to an object with ``select``). This helper hides
    that detail so tests and the loader only need to reason about an
    iterable of entry points.

    The function is also used as an indirection point in tests so that
    behavior can be controlled without relying on the real environment.
    """

    try:
        eps = metadata.entry_points()
    except Exception:  # pragma: no cover - very defensive
        return []

    # Python 3.10+ provides the selectable interface.
    return eps.select(group=group)


def load_prompt_provider(builtin: PromptProvider | None = None) -> PromptProvider:
    """
    Load the configured :class:`PromptProvider` instance.

    Resolves the appropriate prompt provider based on the environment configuration.
    It supports a pluggable architecture via Python entry points, allowing commercial
    or custom prompt bundles to be injected without modifying the core codebase.

    Resolution algorithm:
    1. Read ``GEARMESHING_AI_PROMPT_PROVIDER`` from settings; default to ``"builtin"`` when unset.
    2. If the key is empty or ``"builtin"``, return a :class:`BuiltinPromptProvider` (or the ``builtin`` override).
    3. Otherwise, search the ``gearmesh.prompt_providers`` entry-point group for a matching name.
    4. If found, load and instantiate the provider factory.
    5. On any error (import failure, type mismatch), log a warning and fall back to builtin.

    This function is designed to be safe and resilient, ensuring the system always starts
    with at least the builtin prompts.

    Args:
        builtin: Optional override for the fallback/builtin provider (useful for testing).

    Returns:
        The resolved PromptProvider instance.
    """
    from gearmeshing_ai.server.core.config import settings

    provider_key = settings.gearmeshing_ai_prompt_provider or "builtin"
    base = builtin or BuiltinPromptProvider()

    if provider_key in {"", "builtin"}:
        return base

    for ep in _iter_entry_points("gearmesh.prompt_providers"):
        if ep.name != provider_key:
            continue
        try:
            factory = ep.load()
            provider = factory()
            if isinstance(provider, PromptProvider):
                return provider
            _LOGGER.warning(
                "PromptProviderLoader: entry point %s did not return a PromptProvider; falling back to builtin",
                ep.name,
            )
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.warning(
                "PromptProviderLoader: failed to load provider; name=%s entry_point=%s error=%s",
                provider_key,
                ep.name,
                type(exc).__name__,
            )
            break

    _LOGGER.warning(
        "PromptProviderLoader: using builtin provider after failing to resolve provider_key=%s",
        provider_key,
    )
    return base
