from __future__ import annotations

from dataclasses import dataclass

import pytest

from gearmeshing_ai.agent_core.capabilities.registry import CapabilityRegistry
from gearmeshing_ai.agent_core.schemas.domain import CapabilityName


@dataclass(frozen=True)
class _DummyCapability:
    name: CapabilityName


def test_registry_empty_has_false() -> None:
    reg = CapabilityRegistry()
    assert reg.has(CapabilityName.summarize) is False


def test_registry_get_missing_raises_keyerror() -> None:
    reg = CapabilityRegistry()
    with pytest.raises(KeyError):
        reg.get(CapabilityName.summarize)


def test_registry_register_then_get_returns_same_instance() -> None:
    reg = CapabilityRegistry()
    cap = _DummyCapability(name=CapabilityName.summarize)
    reg.register(cap)

    assert reg.has(CapabilityName.summarize) is True
    assert reg.get(CapabilityName.summarize) is cap


def test_registry_register_overwrites_existing_name() -> None:
    reg = CapabilityRegistry()
    cap1 = _DummyCapability(name=CapabilityName.summarize)
    cap2 = _DummyCapability(name=CapabilityName.summarize)

    reg.register(cap1)
    assert reg.get(CapabilityName.summarize) is cap1

    reg.register(cap2)
    assert reg.get(CapabilityName.summarize) is cap2
