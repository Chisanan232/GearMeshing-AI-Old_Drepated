from __future__ import annotations

from gearmeshing_ai.info_provider.prompt import (
    BuiltinPromptProvider,
    StackedPromptProvider,
)


def test_builtin_prompt_provider_returns_known_prompts() -> None:
    provider = BuiltinPromptProvider()
    pm = provider.get("pm/system", locale="en")
    dev = provider.get("dev/system", locale="en")

    assert "product manager" in pm.lower()
    assert "software engineer" in dev.lower()


def test_builtin_prompt_provider_raises_for_unknown_prompt() -> None:
    provider = BuiltinPromptProvider()
    try:
        provider.get("unknown/key", locale="en")
    except KeyError as exc:
        # Error message should not leak prompt content; only metadata.
        msg = str(exc)
        assert "unknown/key" in msg
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected KeyError for unknown prompt")


def test_stacked_prompt_provider_falls_back_to_builtin() -> None:
    class _Primary(BuiltinPromptProvider):
        def get(self, name: str, locale: str = "en", tenant: str | None = None) -> str:  # type: ignore[override]
            if name == "pm/system":
                return "primary-pm"
            raise KeyError(name)

    builtin = BuiltinPromptProvider()
    stacked = StackedPromptProvider(_Primary(), builtin)

    assert stacked.get("pm/system") == "primary-pm"
    # Unknown in primary should fall back to builtin
    qa_text = stacked.get("qa/system")
    assert "qa" in qa_text.lower()


def test_stacked_prompt_provider_version_includes_both() -> None:
    primary = BuiltinPromptProvider(version_id="p1")
    fallback = BuiltinPromptProvider(version_id="b1")
    stacked = StackedPromptProvider(primary, fallback)

    v = stacked.version()
    assert "p1" in v and "b1" in v


def test_stacked_prompt_provider_refresh_calls_both() -> None:
    class _P(BuiltinPromptProvider):
        def __init__(self) -> None:
            super().__init__()
            self.refreshed = 0

        def refresh(self) -> None:  # type: ignore[override]
            self.refreshed += 1

    class _F(BuiltinPromptProvider):
        def __init__(self) -> None:
            super().__init__()
            self.refreshed = 0

        def refresh(self) -> None:  # type: ignore[override]
            self.refreshed += 1

    primary = _P()
    fallback = _F()
    stacked = StackedPromptProvider(primary, fallback)

    stacked.refresh()

    assert primary.refreshed == 1
    assert fallback.refreshed == 1
