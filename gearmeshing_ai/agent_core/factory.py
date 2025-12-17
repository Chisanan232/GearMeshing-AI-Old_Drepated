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
from .runtime import EngineDeps
from .runtime.engine import AgentEngine
from .schemas.domain import AgentRole, AgentRun
from .service import AgentService, AgentServiceDeps
from .agent_registry import AgentRegistry


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


def build_agent_registry(*, base_policy_config: PolicyConfig, deps: AgentServiceDeps) -> AgentRegistry:
    reg = AgentRegistry()

    def _make_factory(_role: str):
        def _factory(run: AgentRun) -> AgentService:
            cfg = base_policy_config.model_copy(deep=True)
            cfg.autonomy_profile = run.autonomy_profile
            return AgentService(policy_config=cfg, deps=deps)

        return _factory

    for r in AgentRole:
        reg.register(r.value, _make_factory(r.value))

    return reg
