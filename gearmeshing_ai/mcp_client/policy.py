from __future__ import annotations

from typing import Dict, Optional

from pydantic import Field

from .errors import ToolAccessDeniedError
from .schemas.base import BaseSchema


class ToolPolicy(BaseSchema):
    """
    Access policy for agents interacting with MCP servers and tools.
    """

    allowed_servers: Optional[set[str]] = Field(
        default=None,
        description=("If provided, agent may only access these server IDs. If None, all servers are allowed."),
    )
    allowed_tools: Optional[set[str]] = Field(
        default=None,
        description=(
            "If provided, agent may only call tools whose names are included. If None, all tools are allowed."
        ),
    )
    read_only: bool = Field(
        default=False,
        description=("If True, tool calls that are considered mutating should be blocked by the client layer."),
    )


PolicyMap = Dict[str, ToolPolicy]


def enforce_policy(*, agent_id: Optional[str], server_id: str, tool_name: str, policies: Optional[PolicyMap]) -> None:
    """Raise ToolAccessDeniedError if the policy prohibits access.

    This function is intentionally simple for now; mutating vs read-only is left to the caller to enforce.
    """
    if not agent_id or not policies:
        return
    policy = policies.get(agent_id)
    if policy is None:
        return

    if policy.allowed_servers is not None and server_id not in policy.allowed_servers:
        raise ToolAccessDeniedError(agent_id, server_id, tool_name)

    if policy.allowed_tools is not None and tool_name not in policy.allowed_tools:
        raise ToolAccessDeniedError(agent_id, server_id, tool_name)
