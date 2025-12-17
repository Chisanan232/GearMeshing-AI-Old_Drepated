from __future__ import annotations

"""Capability protocol and execution data models.

A capability is the concrete execution unit for Action steps.

The runtime engine resolves ``ActionStep.capability`` through a
``CapabilityRegistry`` and executes the implementation with a
``CapabilityContext``.

Capabilities should:

- be deterministic with respect to their inputs as much as possible,
- return structured outputs in ``CapabilityResult.output``,
- avoid performing policy decisions themselves (policy is enforced by the
  engine before invocation).
"""

from dataclasses import dataclass
from typing import Any, Dict, Protocol

from ..policy.global_policy import GlobalPolicy
from ..schemas.domain import AgentRun, CapabilityName


@dataclass(frozen=True)
class CapabilityContext:
    """Execution context passed to capability implementations.

    Attributes
    ----------
    run:
        The current ``AgentRun`` being executed.
    policy:
        The configured ``GlobalPolicy`` instance for the run.
    deps:
        Runtime dependencies (repositories and registries) bundled in
        ``EngineDeps``.
    """

    run: AgentRun
    policy: GlobalPolicy
    deps: Any


@dataclass(frozen=True)
class CapabilityResult:
    """Structured capability execution result."""

    ok: bool
    output: Dict[str, Any]


class Capability(Protocol):
    """Protocol for capability implementations."""

    name: CapabilityName

    async def execute(self, ctx: CapabilityContext, *, args: Dict[str, Any]) -> CapabilityResult: ...
