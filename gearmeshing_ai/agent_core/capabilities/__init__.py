"""Capability registry and capability execution pipeline.

 A *capability* is the execution unit for Action steps.

 - The planner emits ``ActionStep`` items with a logical capability name.
 - The runtime resolves that capability name through ``CapabilityRegistry``.
 - The engine executes the capability with a ``CapabilityContext`` that includes
   the current run, policy instance, and runtime dependencies.

 Capabilities are intentionally decoupled from planning:

 - Thought steps never resolve capabilities.
 - Action steps must pass through policy checks (allow/deny, safety validation,
   approval gating) before capability execution.

 This package exports:

 - ``Capability``: protocol for async capability execution.
 - ``CapabilityRegistry``: name → capability implementation mapping.
 - ``CapabilityContext``/``CapabilityResult``: execution input/output models.
 """

from .base import Capability, CapabilityContext, CapabilityResult
from .registry import CapabilityRegistry

__all__ = [
    "Capability",
    "CapabilityContext",
    "CapabilityResult",
    "CapabilityRegistry",
]
