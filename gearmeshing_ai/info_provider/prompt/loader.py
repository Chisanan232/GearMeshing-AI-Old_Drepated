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
import os
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
    """Load the configured :class:`PromptProvider` instance.

    Resolution algorithm:

    1. Read ``GEARMESH_PROMPT_PROVIDER`` from the environment; default to
       ``"builtin"`` when unset.
    2. If the key is empty or ``"builtin"``, construct and return a
       :class:`BuiltinPromptProvider` (or the ``builtin`` override, if
       provided).
    3. Otherwise, search the ``gearmesh.prompt_providers`` entry-point group
       for a matching ``ep.name``. If found, call ``ep.load()`` to obtain a
       factory, then call the factory without arguments to build the
       provider.
    4. If the resulting object conforms to :class:`PromptProvider`, use it.
       Otherwise, log a warning and fall back to the builtin provider.
    5. Any import/initialization error results in a warning and a fallback to
       the builtin provider.

    This function never logs prompt text; warnings mention only the provider
    key, entry-point name and exception type.
    """

    provider_key = os.getenv("GEARMESH_PROMPT_PROVIDER") or "builtin"
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
