from __future__ import annotations

import logging
import time
from typing import Dict

import pytest

from gearmeshing_ai.info_provider.prompt import HotReloadWrapper, PromptProvider


class _FakeProvider(PromptProvider):
    def __init__(self) -> None:
        self.refresh_count = 0
        self._version = "v1"

    def get(self, name: str, locale: str = "en", tenant: str | None = None) -> str:
        if name == "pm/system":
            return "pm-text"
        raise KeyError(name)

    def version(self) -> str:
        return self._version

    def refresh(self) -> None:
        self.refresh_count += 1


def test_hot_reload_wrapper_throttles_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _FakeProvider()
    # Use tiny interval and fixed time so we can control calls precisely
    t = 1000.0
    monkeypatch.setattr(time, "monotonic", lambda: t)
    wrapper = HotReloadWrapper(base, interval_seconds=10.0)

    # First call should trigger refresh
    assert wrapper.get("pm/system") == "pm-text"
    assert base.refresh_count == 1

    # Subsequent calls within interval should not trigger additional refresh
    monkeypatch.setattr(time, "monotonic", lambda: t + 1.0)
    _ = wrapper.get("pm/system")
    _ = wrapper.version()
    assert base.refresh_count == 1

    # After interval, refresh should run again
    monkeypatch.setattr(time, "monotonic", lambda: t + 20.0)
    _ = wrapper.get("pm/system")
    assert base.refresh_count == 2


def test_hot_reload_wrapper_logs_without_prompt_leak(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingProvider(PromptProvider):
        def __init__(self) -> None:
            self._version = "v-fail"

        def get(self, name: str, locale: str = "en", tenant: str | None = None) -> str:
            if name == "pm/system":
                return "SECRET-PROMPT-TEXT"
            raise KeyError(name)

        def version(self) -> str:
            return self._version

        def refresh(self) -> None:
            raise RuntimeError("refresh failed")

    base = _FailingProvider()
    # Ensure first call is beyond the interval so refresh is invoked
    monkeypatch.setattr(time, "monotonic", lambda: 10.0)
    wrapper = HotReloadWrapper(base, interval_seconds=1.0)

    caplog.set_level(logging.WARNING)
    # Trigger refresh via get
    with pytest.raises(KeyError):
        wrapper.get("unknown/key")

    # Ensure we logged a warning but did not leak prompt text
    messages = "\n".join(rec.getMessage() for rec in caplog.records)
    # At least one warning was emitted
    assert messages
    # Prompt text must never be logged
    assert "SECRET-PROMPT-TEXT" not in messages


def test_hot_reload_wrapper_zero_interval_never_auto_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _FakeProvider()
    # Regardless of time, interval <= 0 should disable auto refresh
    monkeypatch.setattr(time, "monotonic", lambda: 1000.0)
    wrapper = HotReloadWrapper(base, interval_seconds=0.0)

    # No auto-refresh on get/version
    _ = wrapper.get("pm/system")
    _ = wrapper.version()
    assert base.refresh_count == 0


def test_hot_reload_wrapper_double_check_inside_lock_can_skip_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Outer check passes but inner double-check inside the lock can still skip refresh.

    We simulate a (pathological) monotonic clock that moves backwards between the
    first and second call so that:
    - first call: now - last_refresh >= interval (passes outer check)
    - second call: now - last_refresh < interval (fails inner check and returns)
    This specifically covers reload.py lines 41-42.
    """

    base = _FakeProvider()
    calls = {"n": 0}

    def _fake_monotonic() -> float:
        calls["n"] += 1
        # First call returns 20.0 so outer check passes (20 >= 10)
        # Second call returns 5.0 so inner check fails (5 < 10)
        return 20.0 if calls["n"] == 1 else 5.0

    monkeypatch.setattr(time, "monotonic", _fake_monotonic)
    wrapper = HotReloadWrapper(base, interval_seconds=10.0)

    # Because of the double-check logic, refresh should not be called
    with pytest.raises(KeyError):
        wrapper.get("unknown/key")

    assert base.refresh_count == 0


def test_hot_reload_wrapper_safe_version_handles_version_error(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    class _BadVersionProvider(PromptProvider):
        def get(self, name: str, locale: str = "en", tenant: str | None = None) -> str:  # noqa: ARG002
            raise KeyError(name)

        def version(self) -> str:
            raise RuntimeError("broken-version")

        def refresh(self) -> None:
            raise RuntimeError("refresh-error")

    base = _BadVersionProvider()
    # Ensure first call is beyond the interval so refresh is invoked,
    # and _safe_version needs to handle version() exceptions.
    monkeypatch.setattr(time, "monotonic", lambda: 10.0)
    wrapper = HotReloadWrapper(base, interval_seconds=1.0)

    caplog.set_level(logging.WARNING)
    with pytest.raises(KeyError):
        wrapper.get("pm/system")

    messages = "\n".join(rec.getMessage() for rec in caplog.records)
    # We should still log something, but with version=<unknown>
    assert "<unknown>" in messages
    assert "broken-version" not in messages


def test_hot_reload_wrapper_explicit_refresh_bypasses_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _FakeProvider()
    # First call happens well after initial _last_refresh=0 so it triggers refresh
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    wrapper = HotReloadWrapper(base, interval_seconds=60.0)

    _ = wrapper.get("pm/system")
    assert base.refresh_count == 1

    # Explicit refresh should still run even within interval
    monkeypatch.setattr(time, "monotonic", lambda: 110.0)
    wrapper.refresh()
    assert base.refresh_count == 2
