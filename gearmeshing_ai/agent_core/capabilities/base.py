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
from typing import TYPE_CHECKING, Any, Dict, Protocol

from gearmeshing_ai.core.models.domain import AgentRun
from gearmeshing_ai.info_provider import CapabilityName

from ..policy.global_policy import GlobalPolicy

if TYPE_CHECKING:
    from ..runtime.models import EngineDeps


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
    deps: "EngineDeps"


@dataclass(frozen=True)
class CapabilityResult:
    """
    Structured execution result from a capability.

    Attributes:
        ok: True if the execution was successful, False if it failed.
        output: A dictionary containing results or error details.
    """

    ok: bool
    output: Dict[str, Any]


class Capability(Protocol):
    """
    Protocol definition for capability implementations.

    A capability represents a discrete unit of functionality (tool) that an agent
    can invoke. It must define a unique name and an execution method.
    """

    name: CapabilityName

    async def execute(self, ctx: CapabilityContext, *, args: Dict[str, Any]) -> CapabilityResult:
        """
        Execute the capability with the given context and arguments.

        Args:
            ctx: The runtime context for execution (includes dependencies, policy, run state).
            args: A dictionary of arguments specific to this capability.

        Returns:
            CapabilityResult: The outcome of the execution.
        """
        ...
