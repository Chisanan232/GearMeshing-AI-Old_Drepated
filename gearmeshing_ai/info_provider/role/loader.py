"""Role provider loader utilities.

This module contains the logic that maps configuration (primarily the
``GEARMESHING_AI_ROLE_PROVIDER`` environment variable) to a concrete
``RoleProvider`` implementation.

Two sources of providers are supported:

* The in-repo :class:`HardcodedRoleProvider`, always available as a safe
  fallback.
* Database-based provider for dynamic configuration.
* External packages that register a factory via the
  ``gearmesh.role_providers`` entry-point group. This is how commercial
  role configuration bundles are plugged into the system without modifying the OSS
  codebase.

The loader is deliberately defensive: failures to resolve or import a
commercial provider never prevent the system from starting; it simply falls
back to the hardcoded provider and emits a redacted warning.
"""

from __future__ import annotations

import logging
from importlib import metadata
from typing import Iterable

from .base import RoleProvider
from .static import HardcodedRoleProvider

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


def load_role_provider(builtin: RoleProvider | None = None) -> RoleProvider:
    """
    Load the configured :class:`RoleProvider` instance.

    Resolves the appropriate role provider based on the environment configuration.
    It supports a pluggable architecture via Python entry points, allowing custom
    or commercial role configuration bundles to be injected without modifying the core codebase.

    Resolution algorithm:
    1. Read ``GEARMESHING_AI_ROLE_PROVIDER`` from settings; default to ``"hardcoded"`` when unset.
    2. If the key is empty or ``"hardcoded"``, return a :class:`HardcodedRoleProvider` (or the ``builtin`` override).
    3. If the key is ``"database"``, return a database role provider (requires session).
    4. Otherwise, search the ``gearmesh.role_providers`` entry-point group for a matching name.
    5. If found, load and instantiate the provider factory.
    6. On any error (import failure, type mismatch), log a warning and fall back to hardcoded.

    This function is designed to be safe and resilient, ensuring the system always starts
    with at least the hardcoded role configurations.

    Args:
        builtin: Optional override for the fallback/hardcoded provider (useful for testing).

    Returns:
        The resolved RoleProvider instance.
    """
    from gearmeshing_ai.server.core.config import settings

    provider_key = getattr(settings, "gearmeshing_ai_role_provider", "hardcoded")
    base = builtin or HardcodedRoleProvider()

    if provider_key in {"", "hardcoded"}:
        return base

    # Handle database provider specially
    if provider_key == "database":
        try:
            # Database provider requires a session - this would typically be injected
            # at runtime when the application has a database connection
            _LOGGER.info("RoleProviderLoader: database provider requested, but session not available in loader")
            _LOGGER.warning("RoleProviderLoader: falling back to hardcoded provider (database requires session)")
            return base
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.warning(
                "RoleProviderLoader: failed to initialize database provider; error=%s",
                type(exc).__name__,
            )
            return base

    # Search entry points for custom providers
    for ep in _iter_entry_points("gearmesh.role_providers"):
        if ep.name != provider_key:
            continue
        try:
            factory = ep.load()
            provider = factory()
            if isinstance(provider, RoleProvider):
                return provider
            _LOGGER.warning(
                "RoleProviderLoader: entry point %s did not return a RoleProvider; falling back to hardcoded",
                ep.name,
            )
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.warning(
                "RoleProviderLoader: failed to load provider; name=%s entry_point=%s error=%s",
                provider_key,
                ep.name,
                type(exc).__name__,
            )
            break

    _LOGGER.warning(
        "RoleProviderLoader: using hardcoded provider after failing to resolve provider_key=%s",
        provider_key,
    )
    return base


def load_role_provider_with_session(session, builtin: RoleProvider | None = None) -> RoleProvider:
    """
    Load the configured :class:`RoleProvider` instance with database session support.

    This is a specialized version of load_role_provider that can initialize database
    providers when a session is available.

    Args:
        session: SQLModel database session for database provider.
        builtin: Optional override for the fallback/hardcoded provider.

    Returns:
        The resolved RoleProvider instance.
    """
    from gearmeshing_ai.server.core.config import settings

    provider_key = getattr(settings, "gearmeshing_ai_role_provider", "hardcoded")
    base = builtin or HardcodedRoleProvider()

    if provider_key in {"", "hardcoded"}:
        return base

    if provider_key == "database":
        try:
            from .database import get_database_role_provider
            return get_database_role_provider(session)
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.warning(
                "RoleProviderLoader: failed to initialize database provider; error=%s",
                type(exc).__name__,
            )
            return base

    # Fall back to the regular loader for entry point providers
    return load_role_provider(builtin)
