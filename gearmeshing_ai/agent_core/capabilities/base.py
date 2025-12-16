from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol

from ..policy.global_policy import GlobalPolicy
from ..schemas.domain import AgentRun, CapabilityName


@dataclass(frozen=True)
class CapabilityContext:
    run: AgentRun
    policy: GlobalPolicy
    deps: Any


@dataclass(frozen=True)
class CapabilityResult:
    ok: bool
    output: Dict[str, Any]


class Capability(Protocol):
    name: CapabilityName

    async def execute(self, ctx: CapabilityContext, *, args: Dict[str, Any]) -> CapabilityResult: ...
