from __future__ import annotations

from typing import Dict

from ..schemas.domain import CapabilityName
from .base import Capability


class CapabilityRegistry:
    def __init__(self) -> None:
        self._caps: Dict[CapabilityName, Capability] = {}

    def register(self, cap: Capability) -> None:
        self._caps[cap.name] = cap

    def get(self, name: CapabilityName) -> Capability:
        return self._caps[name]

    def has(self, name: CapabilityName) -> bool:
        return name in self._caps
