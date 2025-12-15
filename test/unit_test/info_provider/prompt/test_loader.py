from __future__ import annotations

from types import SimpleNamespace
from typing import List

import pytest

from gearmeshing_ai.info_provider.prompt import (
    BuiltinPromptProvider,
    PromptProvider,
    load_prompt_provider,
)
from gearmeshing_ai.info_provider.prompt import loader as loader_mod


class _DummyProvider:
    def __init__(self) -> None:
        self._version = "dummy-v1"

    def get(self, name: str, locale: str = "en", tenant: str | None = None) -> str:
        if name == "pm/system":
            return "dummy"
        raise KeyError(name)

    def version(self) -> str:
        return self._version

    def refresh(self) -> None:
        return None


def _make_entry_point(name: str, obj) -> SimpleNamespace:
    return SimpleNamespace(name=name, load=lambda: obj)


def test_load_prompt_provider_defaults_to_builtin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEARMESH_PROMPT_PROVIDER", raising=False)
    provider = load_prompt_provider()
    assert isinstance(provider, BuiltinPromptProvider)


def test_load_prompt_provider_uses_builtin_when_key_is_builtin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEARMESH_PROMPT_PROVIDER", "builtin")
    provider = load_prompt_provider()
    assert isinstance(provider, BuiltinPromptProvider)


def test_load_prompt_provider_loads_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEARMESH_PROMPT_PROVIDER", "commercial")

    dummy = _DummyProvider()

    def _fake_iter(group: str) -> List[SimpleNamespace]:  # type: ignore[override]
        assert group == "gearmesh.prompt_providers"
        return [_make_entry_point("commercial", lambda: dummy)]

    monkeypatch.setattr(loader_mod, "_iter_entry_points", _fake_iter, raising=True)

    provider = load_prompt_provider()
    assert isinstance(provider, PromptProvider)
    assert provider.version() == "dummy-v1"
    assert provider.get("pm/system").lower() == "dummy"


def test_load_prompt_provider_falls_back_when_entry_point_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEARMESH_PROMPT_PROVIDER", "missing")

    def _fake_iter(group: str) -> List[SimpleNamespace]:  # type: ignore[override]
        assert group == "gearmesh.prompt_providers"
        return []

    monkeypatch.setattr(loader_mod, "_iter_entry_points", _fake_iter, raising=True)

    provider = load_prompt_provider()
    assert isinstance(provider, BuiltinPromptProvider)


def test_load_prompt_provider_falls_back_on_broken_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEARMESH_PROMPT_PROVIDER", "broken")

    def _broken_loader() -> None:
        raise RuntimeError("boom")

    def _fake_iter(group: str) -> List[SimpleNamespace]:  # type: ignore[override]
        assert group == "gearmesh.prompt_providers"
        return [_make_entry_point("broken", _broken_loader)]

    monkeypatch.setattr(loader_mod, "_iter_entry_points", _fake_iter, raising=True)

    provider = load_prompt_provider()
    # Must still return a working builtin provider
    assert isinstance(provider, BuiltinPromptProvider)
