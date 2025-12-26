"""
Policy Management Endpoints.

This module handles configuration of tenant-specific agent policies,
such as autonomy levels, budget limits, and allowed tools.
"""

from fastapi import APIRouter, HTTPException

from gearmeshing_ai.server.schemas import PolicyResponse, PolicyUpdate
from gearmeshing_ai.server.services.deps import OrchestratorDep

router = APIRouter()


@router.get(
    "/{tenant_id}",
    response_model=PolicyResponse,
    summary="Get Tenant Policy",
    description="Retrieve the current policy configuration for a specific tenant.",
    response_description="The policy configuration object.",
    responses={404: {"description": "Policy not found for tenant"}},
)
async def get_policy(tenant_id: str, orchestrator: OrchestratorDep) -> PolicyResponse:
    """
    Get tenant policy.

    Returns the active policy settings for the tenant, such as allowed tools,
    resource limits, and autonomy constraints.
    """
    policy_config = await orchestrator.get_policy(tenant_id)
    if not policy_config:
        raise HTTPException(status_code=404, detail="Policy not found for tenant")

    return PolicyResponse(tenant_id=tenant_id, config=policy_config)


@router.put(
    "/{tenant_id}",
    response_model=PolicyResponse,
    summary="Update Tenant Policy",
    description="Update or create the policy configuration for a specific tenant.",
    response_description="The updated policy object.",
)
async def update_policy(tenant_id: str, policy_in: PolicyUpdate, orchestrator: OrchestratorDep) -> PolicyResponse:
    """
    Update tenant policy.

    Merges the provided configuration into the tenant's existing policy.
    If no policy exists, a new one is created.
    """
    current = await orchestrator.get_policy(tenant_id)
    new_config = policy_in.config

    if current:
        # Merge: use model_copy with update to properly merge nested models
        new_config = current.model_copy(update=new_config.model_dump(exclude_unset=True), deep=True)

    updated_config = await orchestrator.update_policy(tenant_id, new_config)

    return PolicyResponse(tenant_id=tenant_id, config=updated_config)
