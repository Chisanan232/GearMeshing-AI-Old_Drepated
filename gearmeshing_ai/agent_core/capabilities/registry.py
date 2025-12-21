from __future__ import annotations

"""Capability registry.

The registry maps a logical ``CapabilityName`` to an executable capability
implementation.

The runtime engine uses this registry to resolve ``ActionStep.capability``
values into concrete implementations.
"""

from typing import Dict

from ..schemas.domain import CapabilityName
from .base import Capability


class CapabilityRegistry:
    """
    In-memory mapping of capability names to implementations.

    This registry is the central lookup mechanism for resolving logical capability
    names (e.g., 'web_search') to executable code. The runtime engine relies on this
    registry to execute Action steps.

    Notes:
        - ``register`` overwrites any existing mapping for the capability name.
        - ``get`` will raise ``KeyError`` if the capability is missing.
    """

    def __init__(self) -> None:
        """Initialize an empty capability registry."""
        self._caps: Dict[CapabilityName, Capability] = {}

    def register(self, cap: Capability) -> None:
        """
        Register a capability implementation.

        Args:
            cap: The capability instance to register. It must expose a ``name`` attribute.
        """
        self._caps[cap.name] = cap

    def get(self, name: CapabilityName) -> Capability:
        """
        Retrieve a registered capability by name.

        Args:
            name: The logical name of the capability.

        Returns:
            The capability implementation.

        Raises:
            KeyError: If no capability is registered with the given name.
        """
        return self._caps[name]

    def has(self, name: CapabilityName) -> bool:
        """
        Check if a capability is registered.

        Args:
            name: The capability name to check.

        Returns:
            True if registered, False otherwise.
        """
        return name in self._caps
