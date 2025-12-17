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
    """In-memory mapping of capability names to implementations.

    Notes
    -----
    - ``register`` overwrites any existing mapping for the capability name.
    - ``get`` will raise ``KeyError`` if the capability is missing.
    """

    def __init__(self) -> None:
        self._caps: Dict[CapabilityName, Capability] = {}

    def register(self, cap: Capability) -> None:
        self._caps[cap.name] = cap

    def get(self, name: CapabilityName) -> Capability:
        return self._caps[name]

    def has(self, name: CapabilityName) -> bool:
        return name in self._caps
