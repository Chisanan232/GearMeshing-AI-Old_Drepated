"""
API endpoints for managing AI agent configurations.

Provides CRUD operations for agent configurations stored in the database,
replacing the YAML-based configuration system.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from gearmeshing_ai.core.logging_config import get_logger
from gearmeshing_ai.core.database import get_session
from gearmeshing_ai.core.database.entities.agent_configs import AgentConfig
from gearmeshing_ai.core.models.io.agent_configs import (
    AgentConfigCreate,
    AgentConfigRead,
    AgentConfigUpdate,
)

logger = get_logger(__name__)

router = APIRouter(tags=["agent-configs"])


@router.post(
    "",
    response_model=AgentConfigRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create Agent Configuration",
    description="Create a new agent configuration for a specific role and tenant. Configurations define agent behavior, system prompts, and model parameters.",
    response_description="The created agent configuration object with auto-generated ID.",
    responses={
        201: {"description": "Agent configuration created successfully"},
        400: {"description": "Invalid configuration data"},
    },
)
async def create_agent_config(
    config: AgentConfigCreate,
    session: AsyncSession = Depends(get_session),
) -> AgentConfigRead:
    """
    Create a new agent configuration.

    Stores a new agent configuration in the database. Each configuration is associated
    with a specific role (e.g., 'planner', 'dev') and optionally a tenant. Configurations
    can be activated/deactivated without deletion.

    - **role_name**: The agent role identifier (e.g., 'planner', 'dev', 'reviewer').
    - **tenant_id**: Optional tenant identifier for tenant-specific configurations.
    - **system_prompt**: The system prompt for the agent.
    - **model**: The LLM model to use (e.g., 'gpt-4', 'claude-3').
    - **temperature**: Model temperature for response variability (0.0-2.0).
    - **max_tokens**: Maximum tokens for model responses.
    - **is_active**: Whether this configuration is currently active.
    """
    db_config = AgentConfig.model_validate(config)
    session.add(db_config)
    await session.commit()
    await session.refresh(db_config)
    return AgentConfigRead.model_validate(db_config)


@router.get(
    "/role/{role_name}",
    response_model=AgentConfigRead,
    summary="Get Agent Configuration by Role",
    description="Retrieve the active agent configuration for a specific role, with optional tenant-specific override.",
    response_description="The agent configuration object for the requested role.",
    responses={
        200: {"description": "Agent configuration found"},
        404: {"description": "No active configuration found for the role"},
    },
)
async def get_agent_config_by_role(
    role_name: str,
    tenant_id: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
) -> AgentConfigRead:
    """
    Get agent configuration by role name.

    Retrieves the active configuration for a specific agent role. If a tenant_id is provided,
    returns the tenant-specific configuration if available; otherwise falls back to the
    default (global) configuration for that role.

    The lookup priority is:
    1. Tenant-specific active configuration
    2. Default (global) active configuration

    - **role_name**: The agent role to retrieve (e.g., 'planner', 'dev').
    - **tenant_id**: Optional tenant identifier for tenant-specific lookup.
    """
    # Try tenant-specific config first
    if tenant_id:
        statement = select(AgentConfig).where(
            (AgentConfig.role_name == role_name)
            & (AgentConfig.tenant_id == tenant_id)
            & (AgentConfig.is_active == True)
        )
        result = await session.execute(statement)
        config = result.scalars().first()
        if config:
            return AgentConfigRead.model_validate(config)

    # Fall back to default (no tenant) config
    statement = select(AgentConfig).where(
        (AgentConfig.role_name == role_name) & (AgentConfig.tenant_id == None) & (AgentConfig.is_active == True)
    )
    result = await session.execute(statement)
    config = result.scalars().first()
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent config for role '{role_name}' not found",
        )
    return AgentConfigRead.model_validate(config)


@router.get(
    "/{config_id}",
    response_model=AgentConfigRead,
    summary="Get Agent Configuration by ID",
    description="Retrieve a specific agent configuration by its unique identifier.",
    response_description="The agent configuration object.",
    responses={
        200: {"description": "Agent configuration found"},
        404: {"description": "Configuration not found"},
    },
)
async def get_agent_config(
    config_id: int,
    session: AsyncSession = Depends(get_session),
) -> AgentConfigRead:
    """
    Get agent configuration by ID.

    Retrieves a specific agent configuration using its database ID.
    This is useful for direct access when you already know the configuration ID.

    - **config_id**: The unique identifier of the agent configuration.
    """
    config = await session.get(AgentConfig, config_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent config {config_id} not found",
        )
    return AgentConfigRead.model_validate(config)


@router.get(
    "",
    response_model=list[AgentConfigRead],
    summary="List Agent Configurations",
    description="Retrieve a list of agent configurations, optionally filtered by tenant and active status.",
    response_description="A list of agent configuration objects.",
    responses={
        200: {"description": "List of configurations retrieved successfully"},
    },
)
async def list_agent_configs(
    tenant_id: Optional[str] = None,
    active_only: bool = True,
    session: AsyncSession = Depends(get_session),
) -> list[AgentConfigRead]:
    """
    List agent configurations.

    Retrieves all agent configurations with optional filtering. By default, only active
    configurations are returned. You can filter by tenant_id to get tenant-specific
    configurations or include inactive ones.

    - **tenant_id**: Optional filter for a specific tenant. If not provided, returns all configurations.
    - **active_only**: If True (default), only returns active configurations. Set to False to include inactive ones.
    """
    try:
        statement = select(AgentConfig)
        if tenant_id:
            statement = statement.where(AgentConfig.tenant_id == tenant_id)
        if active_only:
            statement = statement.where(AgentConfig.is_active == True)
        result = await session.execute(statement)
        configs = result.scalars().all()
        logger.debug(
            f"Retrieved {len(configs)} agent configurations (tenant_id={tenant_id}, active_only={active_only})"
        )
        return [AgentConfigRead.model_validate(config) for config in configs]
    except Exception as e:
        logger.error(f"Failed to list agent configurations: {str(e)}", exc_info=True)
        raise


@router.patch(
    "/{config_id}",
    response_model=AgentConfigRead,
    summary="Update Agent Configuration",
    description="Partially update an existing agent configuration. Only provided fields are updated.",
    response_description="The updated agent configuration object.",
    responses={
        200: {"description": "Agent configuration updated successfully"},
        404: {"description": "Configuration not found"},
    },
)
async def update_agent_config(
    config_id: int,
    config_update: AgentConfigUpdate,
    session: AsyncSession = Depends(get_session),
) -> AgentConfigRead:
    """
    Update agent configuration.

    Performs a partial update on an existing agent configuration. Only the fields
    provided in the request body are updated; other fields remain unchanged.
    This is useful for updating specific properties like system_prompt, model,
    or temperature without affecting other settings.

    - **config_id**: The unique identifier of the configuration to update.
    - **config_update**: The fields to update (all fields are optional).
    """
    config = await session.get(AgentConfig, config_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent config {config_id} not found",
        )

    update_data = config_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(config, key, value)

    session.add(config)
    await session.commit()
    await session.refresh(config)
    return AgentConfigRead.model_validate(config)


@router.delete(
    "/{config_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Agent Configuration",
    description="Permanently delete an agent configuration from the database.",
    responses={
        204: {"description": "Agent configuration deleted successfully"},
        404: {"description": "Configuration not found"},
    },
)
async def delete_agent_config(
    config_id: int,
    session: AsyncSession = Depends(get_session),
) -> None:
    """
    Delete agent configuration.

    Permanently removes an agent configuration from the database. This operation
    cannot be undone. Consider deactivating the configuration instead if you might
    need it later (set is_active to False).

    - **config_id**: The unique identifier of the configuration to delete.
    """
    config = await session.get(AgentConfig, config_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent config {config_id} not found",
        )
    await session.delete(config)
    await session.commit()
