"""Model provider loader utilities.

This module contains the logic that maps configuration (primarily the
``GEARMESHING_AI_MODEL_PROVIDER`` environment variable) to a concrete
``ModelProvider`` implementation.

Two sources of providers are supported:

* The in-repo :class:`HardcodedModelProvider`, always available as a safe
  fallback.
* External packages that register a factory via the
  ``gearmesh.model_providers`` entry-point group. This is how custom
  model configuration bundles are plugged into the system without modifying the OSS
  codebase.

The loader is deliberately defensive: failures to resolve or import a
custom provider never prevent the system from starting; it simply falls
back to the hardcoded provider and emits a redacted warning.
"""

from __future__ import annotations

import logging
from importlib import metadata
from typing import Iterable

from .base import ModelProvider
from .provider import HardcodedModelProvider

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
    # Handle older Python versions that don't have select method.
    try:
        return eps.select(group=group)
    except (AttributeError, TypeError):  # pragma: no cover - defensive for older Python
        # Fallback for older Python versions - return empty list
        return []


def load_model_provider(builtin: ModelProvider | None = None) -> ModelProvider:
    """
    Load the configured :class:`ModelProvider` instance.

    Resolves the appropriate model provider based on the environment configuration.
    It supports a pluggable architecture via Python entry points, allowing custom
    or commercial model configuration bundles to be injected without modifying the core codebase.

    Resolution algorithm:
    1. Read ``GEARMESHING_AI_MODEL_PROVIDER`` from settings; default to ``"hardcoded"`` when unset.
    2. If the key is empty or ``"hardcoded"``, return a :class:`HardcodedModelProvider` (or the ``builtin`` override).
    3. Otherwise, search the ``gearmesh.model_providers`` entry-point group for a matching name.
    4. If found, load and instantiate the provider factory.
    5. On any error (import failure, type mismatch), log a warning and fall back to hardcoded.

    This function is designed to be safe and resilient, ensuring the system always starts
    with at least the hardcoded model configurations.

    Args:
        builtin: Optional override for the fallback/hardcoded provider (useful for testing).

    Returns:
        The resolved ModelProvider instance.
    """
    from gearmeshing_ai.server.core.config import settings

    provider_key = getattr(settings, 'gearmeshing_ai_model_provider', 'hardcoded')
    base = builtin or HardcodedModelProvider()

    if provider_key in {"", "hardcoded"}:
        return base

    for ep in _iter_entry_points("gearmesh.model_providers"):
        if ep.name != provider_key:
            continue
        try:
            factory = ep.load()
            provider = factory()
            if isinstance(provider, ModelProvider):
                return provider
            _LOGGER.warning(
                "ModelProviderLoader: entry point %s did not return a ModelProvider; falling back to hardcoded",
                ep.name,
            )
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.warning(
                "ModelProviderLoader: failed to load provider; name=%s entry_point=%s error=%s",
                provider_key,
                ep.name,
                type(exc).__name__,
            )
            break

    _LOGGER.warning(
        "ModelProviderLoader: using hardcoded provider after failing to resolve provider_key=%s",
        provider_key,
    )
    return base
