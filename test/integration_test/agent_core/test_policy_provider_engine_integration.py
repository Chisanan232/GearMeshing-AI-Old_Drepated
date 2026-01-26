"""Integration tests for policy provider with AgentEngine and AgentService."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gearmeshing_ai.agent_core.policy.global_policy import GlobalPolicy
from gearmeshing_ai.agent_core.policy.models import (
    ApprovalPolicy,
    PolicyConfig,
    SafetyPolicy,
    ToolPolicy,
)
from gearmeshing_ai.agent_core.policy.provider import (
    DatabasePolicyProvider,
    StaticPolicyProvider,
)
from gearmeshing_ai.agent_core.repos.interfaces import PolicyRepository
from gearmeshing_ai.agent_core.runtime.models import EngineDeps
from gearmeshing_ai.agent_core.schemas.domain import (
    AgentRun,
    AgentRunStatus,
    AutonomyProfile,
    RiskLevel,
)
from gearmeshing_ai.info_provider import CapabilityName
from gearmeshing_ai.agent_core.service import AgentService, AgentServiceDeps


class TestPolicyProviderWithGlobalPolicy:
    """Integration tests for policy provider with GlobalPolicy."""

    @pytest.fixture
    def default_policy(self):
        """Create a default policy config."""
        return PolicyConfig(
            version="policy-v1",
            autonomy_profile=AutonomyProfile.balanced,
            tool_policy=ToolPolicy(
                allowed_capabilities=None,
                allowed_tools=None,
                blocked_tools=set(),
                allowed_mcp_servers=None,
                blocked_mcp_servers=set(),
            ),
            approval_policy=ApprovalPolicy(),
            safety_policy=SafetyPolicy(),
        )

    @pytest.mark.asyncio
    async def test_global_policy_with_static_provider(self, default_policy):
        """Test GlobalPolicy works with StaticPolicyProvider."""
        provider = StaticPolicyProvider(default=default_policy)

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id="acme-corp",
        )

        policy_config = await provider.get(run)
        global_policy = GlobalPolicy(policy_config)

        # Test policy decision
        decision = global_policy.decide(
            capability=CapabilityName.web_search,
            args={"query": "test"},
        )

        assert decision.block is False
        assert decision.risk == RiskLevel.low

    @pytest.mark.asyncio
    async def test_global_policy_with_database_provider(self, default_policy):
        """Test GlobalPolicy works with DatabasePolicyProvider."""
        mock_repo = AsyncMock(spec=PolicyRepository)
        mock_repo.get.return_value = None
        provider = DatabasePolicyProvider(
            policy_repository=mock_repo,
            default=default_policy,
        )

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id="acme-corp",
        )

        policy_config = await provider.get(run)

        global_policy = GlobalPolicy(policy_config)

        # Test policy decision
        decision = global_policy.decide(
            capability=CapabilityName.shell_exec,
            args={"command": "ls"},
        )

        assert decision.block is False
        assert decision.risk == RiskLevel.high

    @pytest.mark.asyncio
    async def test_global_policy_blocks_capability(self, default_policy):
        """Test GlobalPolicy blocks disallowed capabilities."""
        restricted_policy = default_policy.model_copy(deep=True)
        restricted_policy.tool_policy.allowed_capabilities = {CapabilityName.web_search}

        provider = StaticPolicyProvider(default=restricted_policy)

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
        )

        policy_config = await provider.get(run)
        global_policy = GlobalPolicy(policy_config)

        # Test policy blocks shell_exec
        decision = global_policy.decide(
            capability=CapabilityName.shell_exec,
            args={"command": "ls"},
        )

        assert decision.block is True
        assert "not allowed" in decision.block_reason

    @pytest.mark.asyncio
    async def test_global_policy_requires_approval_for_high_risk(self, default_policy):
        """Test GlobalPolicy requires approval for high-risk actions."""
        strict_policy = default_policy.model_copy(deep=True)
        strict_policy.autonomy_profile = AutonomyProfile.strict

        provider = StaticPolicyProvider(default=strict_policy)

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
        )

        policy_config = await provider.get(run)
        global_policy = GlobalPolicy(policy_config)

        # Test policy requires approval for high-risk action
        decision = global_policy.decide(
            capability=CapabilityName.shell_exec,
            args={"command": "ls"},
        )

        assert decision.require_approval is True
        assert decision.risk == RiskLevel.high


class TestPolicyProviderWithAgentService:
    """Integration tests for policy provider with AgentService."""

    @pytest.fixture
    def default_policy(self):
        """Create a default policy config."""
        return PolicyConfig(
            version="policy-v1",
            autonomy_profile=AutonomyProfile.balanced,
            tool_policy=ToolPolicy(),
            approval_policy=ApprovalPolicy(),
            safety_policy=SafetyPolicy(),
        )

    @pytest.fixture
    def mock_engine_deps(self):
        """Create mock engine dependencies."""
        return MagicMock(spec=EngineDeps)

    @pytest.fixture
    def mock_planner(self):
        """Create mock planner."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_agent_service_with_static_policy_provider(self, default_policy, mock_engine_deps, mock_planner):
        """Test AgentService with StaticPolicyProvider."""
        provider = StaticPolicyProvider(default=default_policy)

        service_deps = AgentServiceDeps(
            engine_deps=mock_engine_deps,
            planner=mock_planner,
        )

        service = AgentService(
            policy_config=default_policy,
            deps=service_deps,
            policy_provider=provider,
        )

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id="acme-corp",
        )

        # Test policy resolution
        resolved_policy = await service._policy_for_run(run)

        assert resolved_policy.autonomy_profile == AutonomyProfile.balanced
        assert resolved_policy.version == "policy-v1"

    @pytest.mark.asyncio
    async def test_agent_service_with_database_policy_provider(self, default_policy, mock_engine_deps, mock_planner):
        """Test AgentService with DatabasePolicyProvider."""
        mock_repo = AsyncMock(spec=PolicyRepository)
        mock_repo.get.return_value = default_policy.model_dump(mode="json")
        provider = DatabasePolicyProvider(
            policy_repository=mock_repo,
            default=default_policy,
        )

        service_deps = AgentServiceDeps(
            engine_deps=mock_engine_deps,
            planner=mock_planner,
        )

        service = AgentService(
            policy_config=default_policy,
            deps=service_deps,
            policy_provider=provider,
        )

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id="acme-corp",
        )

        resolved_policy = await service._policy_for_run(run)

        assert resolved_policy is not None
        assert resolved_policy.version == "policy-v1"

    @pytest.mark.asyncio
    async def test_agent_service_policy_for_run_applies_autonomy_profile(
        self, default_policy, mock_engine_deps, mock_planner
    ):
        """Test AgentService applies run-specific autonomy profile."""
        provider = StaticPolicyProvider(default=default_policy)

        service_deps = AgentServiceDeps(
            engine_deps=mock_engine_deps,
            planner=mock_planner,
        )

        service = AgentService(
            policy_config=default_policy,
            deps=service_deps,
            policy_provider=provider,
        )

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            autonomy_profile=AutonomyProfile.strict,
        )

        resolved_policy = await service._policy_for_run(run)

        # Autonomy profile from run should override provider's policy
        assert resolved_policy.autonomy_profile == AutonomyProfile.strict

    @pytest.mark.asyncio
    async def test_agent_service_without_policy_provider(self, default_policy, mock_engine_deps, mock_planner):
        """Test AgentService uses base policy when no provider is given."""
        service_deps = AgentServiceDeps(
            engine_deps=mock_engine_deps,
            planner=mock_planner,
        )

        service = AgentService(
            policy_config=default_policy,
            deps=service_deps,
            policy_provider=None,
        )

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
        )

        resolved_policy = await service._policy_for_run(run)

        assert resolved_policy.version == "policy-v1"
        assert resolved_policy.autonomy_profile == AutonomyProfile.balanced


class TestPolicyProviderTenantIsolation:
    """Integration tests for tenant isolation with policy provider."""

    @pytest.fixture
    def default_policy(self):
        """Create a default policy config."""
        return PolicyConfig(
            version="policy-v1",
            autonomy_profile=AutonomyProfile.balanced,
            tool_policy=ToolPolicy(),
            approval_policy=ApprovalPolicy(),
            safety_policy=SafetyPolicy(),
        )

    @pytest.mark.asyncio
    async def test_different_tenants_get_different_policies(self, default_policy):
        """Test different tenants get different policies from provider."""
        acme_policy = default_policy.model_copy(deep=True)
        acme_policy.autonomy_profile = AutonomyProfile.strict

        globex_policy = default_policy.model_copy(deep=True)
        globex_policy.autonomy_profile = AutonomyProfile.unrestricted

        provider = StaticPolicyProvider(
            default=default_policy,
            by_tenant={
                "acme-corp": acme_policy,
                "globex-corp": globex_policy,
            },
        )

        acme_run = AgentRun(
            id="acme-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id="acme-corp",
        )

        globex_run = AgentRun(
            id="globex-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id="globex-corp",
        )

        acme_config = await provider.get(acme_run)
        globex_config = await provider.get(globex_run)

        assert acme_config.autonomy_profile == AutonomyProfile.strict
        assert globex_config.autonomy_profile == AutonomyProfile.unrestricted

    @pytest.mark.asyncio
    async def test_database_provider_isolates_tenants(self, default_policy):
        """Test DatabasePolicyProvider isolates tenant policies."""
        mock_repo = AsyncMock(spec=PolicyRepository)

        acme_policy_dict = {
            "version": "policy-v1",
            "autonomy_profile": "strict",
            "tool_policy": {
                "allowed_capabilities": None,
                "allowed_tools": None,
                "blocked_tools": [],
                "allowed_mcp_servers": None,
                "blocked_mcp_servers": [],
            },
            "approval_policy": {
                "require_for_risk_at_or_above": "medium",
                "approval_ttl_seconds": 900.0,
                "tool_risk_overrides": {},
                "tool_risk_kinds": {},
            },
            "safety_policy": {
                "block_prompt_injection": True,
                "redact_secrets": True,
                "max_tool_args_bytes": 64000,
            },
            "budget_policy": {"max_total_tokens": None},
        }

        globex_policy_dict = {
            "version": "policy-v1",
            "autonomy_profile": "unrestricted",
            "tool_policy": {
                "allowed_capabilities": None,
                "allowed_tools": None,
                "blocked_tools": [],
                "allowed_mcp_servers": None,
                "blocked_mcp_servers": [],
            },
            "approval_policy": {
                "require_for_risk_at_or_above": "high",
                "approval_ttl_seconds": 900.0,
                "tool_risk_overrides": {},
                "tool_risk_kinds": {},
            },
            "safety_policy": {
                "block_prompt_injection": False,
                "redact_secrets": False,
                "max_tool_args_bytes": 64000,
            },
            "budget_policy": {"max_total_tokens": None},
        }

        async def mock_get(tenant_id):
            if tenant_id == "acme-corp":
                return acme_policy_dict
            elif tenant_id == "globex-corp":
                return globex_policy_dict
            return None

        mock_repo.get = mock_get

        provider = DatabasePolicyProvider(
            policy_repository=mock_repo,
            default=default_policy,
        )

        acme_run = AgentRun(
            id="acme-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id="acme-corp",
        )

        globex_run = AgentRun(
            id="globex-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id="globex-corp",
        )

        acme_config = await provider.get(acme_run)
        globex_config = await provider.get(globex_run)

        assert acme_config.autonomy_profile == AutonomyProfile.strict
        assert globex_config.autonomy_profile == AutonomyProfile.unrestricted
        assert acme_config.safety_policy.redact_secrets is True
        assert globex_config.safety_policy.redact_secrets is False


class TestPolicyProviderWithRiskClassification:
    """Integration tests for policy provider with risk classification."""

    @pytest.fixture
    def default_policy(self):
        """Create a default policy config."""
        return PolicyConfig(
            version="policy-v1",
            autonomy_profile=AutonomyProfile.balanced,
            tool_policy=ToolPolicy(),
            approval_policy=ApprovalPolicy(),
            safety_policy=SafetyPolicy(),
        )

    @pytest.mark.asyncio
    async def test_policy_classifies_tool_risk_correctly(self, default_policy):
        """Test policy correctly classifies tool risk levels."""
        policy_with_overrides = default_policy.model_copy(deep=True)
        policy_with_overrides.approval_policy.tool_risk_overrides = {
            "web_search": RiskLevel.low,
            "delete_file": RiskLevel.high,
        }

        provider = StaticPolicyProvider(default=policy_with_overrides)

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
        )

        policy_config = await provider.get(run)
        global_policy = GlobalPolicy(policy_config)

        # Test risk classification with overrides
        web_search_risk = global_policy.classify_risk(
            CapabilityName.mcp_call,
            args={},
            logical_tool="web_search",
        )

        delete_file_risk = global_policy.classify_risk(
            CapabilityName.mcp_call,
            args={},
            logical_tool="delete_file",
        )

        assert web_search_risk == RiskLevel.low
        assert delete_file_risk == RiskLevel.high

    @pytest.mark.asyncio
    async def test_policy_approval_requirement_based_on_risk(self, default_policy):
        """Test policy determines approval requirement based on risk."""
        balanced_policy = default_policy.model_copy(deep=True)
        balanced_policy.autonomy_profile = AutonomyProfile.balanced
        balanced_policy.approval_policy.require_for_risk_at_or_above = RiskLevel.medium

        provider = StaticPolicyProvider(default=balanced_policy)

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
        )

        policy_config = await provider.get(run)
        global_policy = GlobalPolicy(policy_config)

        # Low risk should not require approval
        low_risk_decision = global_policy.decide(
            capability=CapabilityName.web_search,
            args={"query": "test"},
        )

        # High risk should require approval
        high_risk_decision = global_policy.decide(
            capability=CapabilityName.shell_exec,
            args={"command": "ls"},
        )

        assert low_risk_decision.require_approval is False
        assert high_risk_decision.require_approval is True


class TestPolicyProviderErrorHandling:
    """Integration tests for policy provider error handling."""

    @pytest.fixture
    def default_policy(self):
        """Create a default policy config."""
        return PolicyConfig(
            version="policy-v1",
            autonomy_profile=AutonomyProfile.balanced,
            tool_policy=ToolPolicy(),
            approval_policy=ApprovalPolicy(),
            safety_policy=SafetyPolicy(),
        )

    @pytest.mark.asyncio
    async def test_policy_provider_handles_missing_tenant_policy(self, default_policy):
        """Test policy provider gracefully handles missing tenant policy."""
        mock_repo = AsyncMock(spec=PolicyRepository)
        mock_repo.get.return_value = None

        provider = DatabasePolicyProvider(
            policy_repository=mock_repo,
            default=default_policy,
        )

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id="unknown-tenant",
        )

        policy_config = await provider.get(run)

        # Should fall back to default
        assert policy_config.version == "policy-v1"
        assert policy_config.autonomy_profile == AutonomyProfile.balanced

    @pytest.mark.asyncio
    async def test_policy_provider_handles_invalid_policy_config(self, default_policy):
        """Test policy provider handles invalid policy configuration."""
        mock_repo = AsyncMock(spec=PolicyRepository)
        mock_repo.get.return_value = {"invalid": "config"}

        provider = DatabasePolicyProvider(
            policy_repository=mock_repo,
            default=default_policy,
        )

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id="acme-corp",
        )

        policy_config = await provider.get(run)

        # Should fall back to default
        assert policy_config.version == "policy-v1"

    @pytest.mark.asyncio
    async def test_policy_provider_handles_repository_error(self, default_policy):
        """Test policy provider handles repository errors gracefully."""
        mock_repo = AsyncMock(spec=PolicyRepository)
        mock_repo.get.side_effect = RuntimeError("Database error")

        provider = DatabasePolicyProvider(
            policy_repository=mock_repo,
            default=default_policy,
        )

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id="acme-corp",
        )

        policy_config = await provider.get(run)

        # Should fall back to default
        assert policy_config.version == "policy-v1"
        assert policy_config.autonomy_profile == AutonomyProfile.balanced
