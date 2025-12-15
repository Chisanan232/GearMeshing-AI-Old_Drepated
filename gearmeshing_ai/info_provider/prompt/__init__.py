"""Prompt provider facade.

This subpackage defines the public surface for GearMeshing-AI's prompt
provider abstraction. It re-exports the key types that callers are expected
to use:

- ``PromptProvider`` – protocol describing the minimal provider API.
- ``BuiltinPromptProvider`` – in-repo, non-sensitive prompts for OSS/basic use.
- ``StackedPromptProvider`` – composition helper for commercial+builtin stacks.
- ``HotReloadWrapper`` – optional wrapper that periodically refreshes a
  provider instance in a thread-safe, non-blocking way.
- ``load_prompt_provider`` – helper that selects a provider based on
  configuration and Python entry points.

Higher layers (API server, agents) should import from this module rather than
individual implementation files to keep the integration surface stable.
"""

from .provider import BuiltinPromptProvider, StackedPromptProvider, HotReloadWrapper
from .loader import load_prompt_provider
from .base import PromptProvider

__all__ = [
    "PromptProvider",
    "BuiltinPromptProvider",
    "StackedPromptProvider",
    "HotReloadWrapper",
    "load_prompt_provider",
]
