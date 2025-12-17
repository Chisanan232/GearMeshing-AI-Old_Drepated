from __future__ import annotations

from ..info_provider.prompt import PromptProvider

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
from .runtime.engine import AgentEngine
from .runtime import EngineDeps


def build_default_registry() -> CapabilityRegistry:
    reg = CapabilityRegistry()
    reg.register(SummarizeCapability())
    reg.register(WebSearchCapability())
    reg.register(WebFetchCapability())
    reg.register(ShellExecCapability())
    reg.register(CodeExecutionCapability())
    reg.register(CodegenCapability())
    return reg


def build_engine(*, policy_config: PolicyConfig, deps: EngineDeps) -> AgentEngine:
    policy = GlobalPolicy(policy_config)
    return AgentEngine(policy=policy, deps=deps)
