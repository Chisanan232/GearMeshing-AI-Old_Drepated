"""Policy models and enforcement helpers for MCP client.

Defines `ToolPolicy` for per-agent access policies and `enforce_policy` for
basic server/tool allow-list checks. Read-only enforcement is performed by the
client facades using additional context from listed tools and name heuristics.
"""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import Field

from .errors import ToolAccessDeniedError
from .schemas.base import BaseSchema


class ToolPolicy(BaseSchema):
    """Access policy for agents interacting with MCP servers and tools.

    Usage guidelines:
    - `allowed_servers`: If set, the agent can only access those server IDs.
    - `allowed_tools`: If set, the agent can only call tools in this set.
    - `read_only`: When True, mutating tool calls should be blocked by clients.
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
    """Raise `ToolAccessDeniedError` if agent's policy disallows the operation.

    Performs allow-list checks for server and tool names. Enforcement of
    read-only semantics (i.e., blocking mutating tools) is deliberately left to
    the caller because it may require tool metadata or heuristics.

    Args:
        agent_id: Optional agent identifier whose policy should be applied.
        server_id: Target server identifier.
        tool_name: Tool identifier (used for tool allow-list checks).
        policies: Optional map of agent IDs to `ToolPolicy`.

    Returns:
        None. This function either returns normally or raises an error.

    Raises:
        ToolAccessDeniedError: If the agent's policy disallows the server or tool.
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
