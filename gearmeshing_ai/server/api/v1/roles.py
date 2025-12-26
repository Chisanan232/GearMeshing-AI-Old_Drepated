"""
Roles API Endpoints.

This module provides read-only access to available agent roles.
Role discovery allows clients to understand which agent roles are available
for use in agent runs and configurations.

Note: This module only provides read-only endpoints. Role configuration
and customization are managed through agent configurations.
"""
from fastapi import APIRouter
from typing import List

from gearmeshing_ai.server.services.deps import OrchestratorDep

router = APIRouter()

@router.get(
    "/",
    response_model=List[str],
    summary="List Available Roles",
    description="Retrieve a list of all available agent roles that can be used in agent runs.",
    response_description="A list of role identifiers (e.g., 'planner', 'dev', 'qa').",
    responses={
        200: {"description": "List of available roles retrieved successfully"},
    },
)
async def list_roles(orchestrator: OrchestratorDep) -> List[str]:
    """
    List available agent roles.

    Returns the set of standard roles that agents can assume. These roles define
    the agent's behavior, capabilities, and system prompt. Use these role identifiers
    when creating agent runs or configuring agent behavior.

    Available roles typically include:
    - **planner**: Strategic planning and task decomposition
    - **dev**: Development and implementation tasks
    - **qa**: Quality assurance and testing
    - **reviewer**: Code and work review

    To customize role behavior for your tenant, use the Agent Configuration API
    to create tenant-specific configurations for each role.
    """
    return await orchestrator.list_available_roles()
