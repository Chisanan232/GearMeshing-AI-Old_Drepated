"""Core RoleProvider protocol used by the role subsystem.

This module defines :class:`RoleProvider`, a small, runtime-checkable
protocol that all role provider implementations are expected to follow.
Concrete providers live in :mod:`gearmeshing_ai.info_provider.role.provider`
and external/commercial providers are wired in via entry points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .models import RoleDefinition
else:
    # Runtime import to avoid circular dependency
    RoleDefinition = None


@runtime_checkable
class RoleProvider(Protocol):
    """Protocol for role providers used by GearMeshing-AI.

    The role provider abstraction sits between application code (API
    handlers, agents, workflows) and the concrete storage of role configurations.
    It is intentionally small so that different deployments can plug in
    alternative implementations without changing call sites.

    Key design points:

    - **Keyed access**: roles are addressed by a stable string
      (e.g. ``"dev"``, ``"planner"``) plus an optional tenant id.
    - **Versioning**: ``version()`` returns a short, non-sensitive identifier
      that can be attached to run metadata for debugging and audits.
    - **Reload**: ``refresh()`` is an optional hook for providers that support
      hot reload (e.g. database-based providers). Implementations that do not
      support reload should implement it as a cheap no-op.

    Implementations must be side-effect free for ``get`` and should never log
    or expose sensitive role information through this interface.
    """

    def get(self, role: str, tenant: Optional[str] = None) -> RoleDefinition:
        """Return the role definition for `role`.

        Implementations should raise `KeyError` when the role is unknown.
        """
        ...

    def list_roles(self, tenant: Optional[str] = None) -> list[str]:
        """Return a list of available role names.

        Args:
            tenant: Optional tenant identifier to filter by.

        Returns:
            List of role names available for the tenant.
        """
        ...

    def version(self) -> str:
        """Return a non-sensitive version identifier for this provider."""
        ...

    def refresh(self) -> None:
        """Refresh internal caches if supported (optional)."""
        ...
