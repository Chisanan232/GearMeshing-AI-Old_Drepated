from __future__ import annotations

"""Convenience factories for wiring the agent core.

This module contains small helpers to build the default capability registry
and instantiate an ``AgentEngine`` from a ``PolicyConfig``.

The intent is to keep application wiring and tests concise, while still
allowing advanced deployments to provide their own registry, policy
configuration, and dependency bundles.
"""

from .capabilities.builtin import (
    CodeExecutionCapability,
    CodegenCapability,
    ShellExecCapability,
    SummarizeCapability,
    WebFetchCapability,
    WebSearchCapability,
)
from .capabilities.registry import CapabilityRegistry
from .policy.global_policy import GlobalPolicy
from .policy.models import PolicyConfig
from .policy.provider import PolicyProvider
from .runtime import EngineDeps
from .runtime.engine import AgentEngine
from .schemas.domain import AgentRole, AgentRun
from .service import AgentService, AgentServiceDeps
from .agent_registry import AgentRegistry
from .role_provider import AgentRoleProvider, DEFAULT_ROLE_PROVIDER


def build_default_registry() -> CapabilityRegistry:
    """Build the default ``CapabilityRegistry``.

    The default registry includes the built-in capabilities shipped with the
    repository (summarization, web search/fetch, shell execution, etc.).
    """
    reg = CapabilityRegistry()
    reg.register(SummarizeCapability())
    reg.register(WebSearchCapability())
    reg.register(WebFetchCapability())
    reg.register(ShellExecCapability())
    reg.register(CodeExecutionCapability())
    reg.register(CodegenCapability())
    return reg


def build_engine(*, policy_config: PolicyConfig, deps: EngineDeps) -> AgentEngine:
    """Construct an ``AgentEngine`` from config and dependencies."""
    policy = GlobalPolicy(policy_config)
    return AgentEngine(policy=policy, deps=deps)


def build_agent_registry(
    *,
    base_policy_config: PolicyConfig,
    deps: AgentServiceDeps,
    role_provider: AgentRoleProvider = DEFAULT_ROLE_PROVIDER,
    policy_provider: PolicyProvider | None = None,
) -> AgentRegistry:
    reg = AgentRegistry()

    def _make_factory(_role: str):
        def _factory(run: AgentRun) -> AgentService:
            cfg = (
                policy_provider.get(run).model_copy(deep=True)
                if policy_provider is not None
                else base_policy_config.model_copy(deep=True)
            )
            cfg.autonomy_profile = run.autonomy_profile

            role_def = role_provider.get(_role)

            role_caps = set(role_def.permissions.allowed_capabilities)
            if cfg.tool_policy.allowed_capabilities is None:
                cfg.tool_policy.allowed_capabilities = role_caps
            else:
                cfg.tool_policy.allowed_capabilities = set(cfg.tool_policy.allowed_capabilities).intersection(role_caps)

            role_tools = set(role_def.permissions.allowed_tools)
            if role_tools:
                if cfg.tool_policy.allowed_tools is None:
                    cfg.tool_policy.allowed_tools = role_tools
                else:
                    cfg.tool_policy.allowed_tools = set(cfg.tool_policy.allowed_tools).intersection(role_tools)

            return AgentService(policy_config=cfg, deps=deps, policy_provider=policy_provider)

        return _factory

    for r in AgentRole:
        reg.register(r.value, _make_factory(r.value))

    return reg
