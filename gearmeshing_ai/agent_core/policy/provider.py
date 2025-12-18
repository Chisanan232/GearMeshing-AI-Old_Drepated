from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol

from ..schemas.domain import AgentRun
from .models import PolicyConfig


class PolicyProvider(Protocol):
    """Resolve the effective PolicyConfig for a given run.

    Implementations can incorporate tenant/workspace overrides and versioning.
    """

    def get(self, run: AgentRun) -> PolicyConfig: ...


@dataclass(frozen=True)
class StaticPolicyProvider(PolicyProvider):
    """PolicyProvider backed by an in-memory mapping.

    Lookup order:

    1) tenant override (if tenant_id present)
    2) workspace override (if workspace_id present)
    3) default

    Returned configs are deep-copied to prevent accidental mutation.
    """

    default: PolicyConfig
    by_tenant: Dict[str, PolicyConfig] | None = None
    by_workspace: Dict[str, PolicyConfig] | None = None

    def get(self, run: AgentRun) -> PolicyConfig:
        if run.tenant_id and self.by_tenant and run.tenant_id in self.by_tenant:
            return self.by_tenant[run.tenant_id].model_copy(deep=True)
        if run.workspace_id and self.by_workspace and run.workspace_id in self.by_workspace:
            return self.by_workspace[run.workspace_id].model_copy(deep=True)
        return self.default.model_copy(deep=True)
