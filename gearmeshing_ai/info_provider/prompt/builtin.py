from __future__ import annotations

from typing import Dict, Optional

from .provider import PromptProvider

# Minimal, non-sensitive builtin prompts so OSS deployments can run in
# "basic" mode without any commercial bundle. These are intentionally
# terse and generic.


_BUILTIN_PROMPTS: Dict[str, Dict[str, str]] = {
    "en": {
        "pm/system": "You are a pragmatic product manager. Focus on clear user outcomes, constraints, and trade-offs.",
        "dev/system": "You are a senior software engineer. Prefer small, safe changes, explicit assumptions, and tests.",
        "qa/system": "You are a meticulous QA engineer. Think in terms of edge cases, regressions, and observability.",
    }
}


class BuiltinPromptProvider(PromptProvider):
    """In-repo builtin prompts for basic/local usage.

    The version is derived from a static identifier so that callers can record
    which builtin prompt set was used in a given run.
    """

    def __init__(self, *, prompts: Optional[Dict[str, Dict[str, str]]] = None, version_id: str = "builtin-v1") -> None:
        self._prompts = prompts or _BUILTIN_PROMPTS
        self._version = version_id

    def get(self, name: str, locale: str = "en", tenant: Optional[str] = None) -> str:  # noqa: ARG002
        bucket = self._prompts.get(locale) or {}
        try:
            return bucket[name]
        except KeyError as exc:  # pragma: no cover - trivial branch
            raise KeyError(f"prompt not found: locale={locale!r} name={name!r}") from exc

    def version(self) -> str:
        return self._version

    def refresh(self) -> None:  # noqa: D401 - trivial no-op
        """Builtin provider has no external state to refresh."""

        return None
