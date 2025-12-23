from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Protocol

from ..repos.interfaces import PolicyRepository
from ..schemas.domain import AgentRun
from .models import PolicyConfig

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class DatabasePolicyProvider(PolicyProvider):
    """PolicyProvider backed by a PolicyRepository (persistence layer).

    Loads tenant-specific policy configurations from the database.

    Lookup order:

    1) tenant-specific policy from database (if tenant_id present)
    2) fallback to default policy

    Returned configs are deep-copied to prevent accidental mutation.
    """

    policy_repository: PolicyRepository
    default: PolicyConfig

    def get(self, run: AgentRun) -> PolicyConfig:
        """Resolve policy config for a run, loading from database if available.

        Args:
            run: The agent run for which to resolve policy.

        Returns:
            The resolved PolicyConfig.
        """
        if run.tenant_id:
            try:
                import asyncio

                # Get the policy config from the database
                config_dict = asyncio.run(self.policy_repository.get(run.tenant_id))
                if config_dict:
                    try:
                        # Parse the config dict into a PolicyConfig object
                        policy_config = PolicyConfig.model_validate(config_dict)
                        logger.debug(f"Loaded policy config from database for tenant '{run.tenant_id}'")
                        return policy_config
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse policy config for tenant '{run.tenant_id}': {e}. "
                            "Falling back to default policy."
                        )
            except Exception as e:
                logger.warning(
                    f"Failed to load policy config from database for tenant '{run.tenant_id}': {e}. "
                    "Falling back to default policy."
                )

        return self.default.model_copy(deep=True)


async def async_get_policy_config(
    policy_repository: PolicyRepository,
    run: AgentRun,
    default: PolicyConfig,
) -> PolicyConfig:
    """Asynchronously resolve policy config for a run from the database.

    This is the async version of DatabasePolicyProvider.get() for use in async contexts.

    Args:
        policy_repository: The policy repository to load from.
        run: The agent run for which to resolve policy.
        default: The default policy config to use as fallback.

    Returns:
        The resolved PolicyConfig.
    """
    if run.tenant_id:
        try:
            config_dict = await policy_repository.get(run.tenant_id)
            if config_dict:
                try:
                    policy_config = PolicyConfig.model_validate(config_dict)
                    logger.debug(f"Loaded policy config from database for tenant '{run.tenant_id}'")
                    return policy_config
                except Exception as e:
                    logger.warning(
                        f"Failed to parse policy config for tenant '{run.tenant_id}': {e}. "
                        "Falling back to default policy."
                    )
        except Exception as e:
            logger.warning(
                f"Failed to load policy config from database for tenant '{run.tenant_id}': {e}. "
                "Falling back to default policy."
            )

    return default.model_copy(deep=True)
