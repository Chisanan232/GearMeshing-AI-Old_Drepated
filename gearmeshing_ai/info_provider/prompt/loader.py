from __future__ import annotations

import logging
import os
from typing import Iterable

from importlib import metadata

from .provider import BuiltinPromptProvider
from .base import PromptProvider


_LOGGER = logging.getLogger(__name__)


def _iter_entry_points(group: str) -> Iterable[metadata.EntryPoint]:  # type: ignore[name-defined]
    try:
        eps = metadata.entry_points()
    except Exception:  # pragma: no cover - very defensive
        return []

    # Python 3.10+ provides the selectable interface.
    select = getattr(eps, "select", None)
    if callable(select):
        return select(group=group)
    return eps.get(group, [])  # type: ignore[no-any-return]


def load_prompt_provider(builtin: PromptProvider | None = None) -> PromptProvider:
    """Load the configured `PromptProvider`.

    Behavior:
    - Reads `GEARMESH_PROMPT_PROVIDER` env var.
    - If unset or `"builtin"`, returns `BuiltinPromptProvider`.
    - Otherwise, tries to load a provider from entry points group
      `gearmesh.prompt_providers` with that name.
    - On any failure, logs redacted metadata and falls back to builtin.
    """

    provider_key = os.getenv("GEARMESH_PROMPT_PROVIDER") or "builtin"
    base = builtin or BuiltinPromptProvider()

    if provider_key in {"", "builtin"}:
        return base

    for ep in _iter_entry_points("gearmesh.prompt_providers"):
        if ep.name != provider_key:
            continue
        try:
            factory = ep.load()
            provider = factory()  # type: ignore[call-arg]
            if isinstance(provider, PromptProvider):
                return provider
            _LOGGER.warning(
                "PromptProviderLoader: entry point %s did not return a PromptProvider; falling back to builtin",
                ep.name,
            )
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.warning(
                "PromptProviderLoader: failed to load provider; name=%s entry_point=%s error=%s",
                provider_key,
                ep.name,
                type(exc).__name__,
            )
            break

    _LOGGER.warning(
        "PromptProviderLoader: using builtin provider after failing to resolve provider_key=%s",
        provider_key,
    )
    return base
