from __future__ import annotations

import pytest

from gearmeshing_ai.info_provider import (
    DEFAULT_ROLE_PROVIDER,
    StaticAgentRoleProvider,
    AgentRole,
)


@pytest.mark.parametrize("role", list(AgentRole))
def test_default_role_provider_supports_all_roles(role: AgentRole) -> None:
    d = DEFAULT_ROLE_PROVIDER.get(role)
    assert d.role == role
    assert d.cognitive.system_prompt_key


def test_static_role_provider_rejects_unknown_role() -> None:
    p = StaticAgentRoleProvider(definitions={})
    with pytest.raises(Exception):
        p.get("missing")
