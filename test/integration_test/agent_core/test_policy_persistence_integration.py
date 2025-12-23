"""Integration tests for policy persistence layer integration with agent core."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gearmeshing_ai.agent_core.policy.models import (
    ApprovalPolicy,
    PolicyConfig,
    SafetyPolicy,
    ToolPolicy,
    ToolRiskKind,
)
from gearmeshing_ai.agent_core.policy.provider import (
    DatabasePolicyProvider,
    StaticPolicyProvider,
    async_get_policy_config,
)
from gearmeshing_ai.agent_core.repos.interfaces import PolicyRepository
from gearmeshing_ai.agent_core.schemas.domain import (
    AgentRun,
    AgentRunStatus,
    AutonomyProfile,
    RiskLevel,
)


class TestDatabasePolicyProvider:
    """Integration tests for DatabasePolicyProvider with persistence layer."""

    @pytest.fixture
    def mock_policy_repository(self):
        """Create a mock PolicyRepository."""
        return MagicMock(spec=PolicyRepository)

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

    def test_database_policy_provider_initialization(self, mock_policy_repository, default_policy):
        """Test DatabasePolicyProvider initializes with repository and default policy."""
        provider = DatabasePolicyProvider(
            policy_repository=mock_policy_repository,
            default=default_policy,
        )

        assert provider.policy_repository is mock_policy_repository
        assert provider.default is default_policy

    def test_database_policy_provider_returns_default_when_no_tenant(self, mock_policy_repository, default_policy):
        """Test provider returns default policy when run has no tenant_id."""
        provider = DatabasePolicyProvider(
            policy_repository=mock_policy_repository,
            default=default_policy,
        )

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id=None,
        )

        policy = provider.get(run)

        assert policy == default_policy
        mock_policy_repository.get.assert_not_called()

    def test_database_policy_provider_loads_tenant_policy(self, mock_policy_repository, default_policy):
        """Test provider loads tenant-specific policy from database."""
        tenant_policy_dict = {
            "version": "policy-v1",
            "autonomy_profile": "strict",
            "tool_policy": {
                "allowed_capabilities": ["mcp_call"],
                "allowed_tools": ["web_search"],
                "blocked_tools": [],
                "allowed_mcp_servers": None,
                "blocked_mcp_servers": [],
            },
            "approval_policy": {
                "require_for_risk_at_or_above": "low",
                "approval_ttl_seconds": 600.0,
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

        mock_policy_repository.get = MagicMock(return_value=tenant_policy_dict)

        provider = DatabasePolicyProvider(
            policy_repository=mock_policy_repository,
            default=default_policy,
        )

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id="acme-corp",
        )

        with patch("asyncio.run", return_value=tenant_policy_dict):
            policy = provider.get(run)

        assert policy.autonomy_profile == AutonomyProfile.strict
        assert "mcp_call" in policy.tool_policy.allowed_capabilities

    def test_database_policy_provider_fallback_on_missing_policy(self, mock_policy_repository, default_policy):
        """Test provider falls back to default when tenant policy not found."""
        mock_policy_repository.get = MagicMock(return_value=None)

        provider = DatabasePolicyProvider(
            policy_repository=mock_policy_repository,
            default=default_policy,
        )

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id="unknown-tenant",
        )

        with patch("asyncio.run", return_value=None):
            policy = provider.get(run)

        assert policy == default_policy

    def test_database_policy_provider_fallback_on_invalid_config(self, mock_policy_repository, default_policy):
        """Test provider falls back to default when policy config is invalid."""
        invalid_config = {"invalid": "config"}

        mock_policy_repository.get = MagicMock(return_value=invalid_config)

        provider = DatabasePolicyProvider(
            policy_repository=mock_policy_repository,
            default=default_policy,
        )

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id="acme-corp",
        )

        with patch("asyncio.run", return_value=invalid_config):
            policy = provider.get(run)

        assert policy == default_policy

    def test_database_policy_provider_fallback_on_repository_error(self, mock_policy_repository, default_policy):
        """Test provider falls back to default when repository raises error."""
        mock_policy_repository.get = MagicMock(side_effect=RuntimeError("Database error"))

        provider = DatabasePolicyProvider(
            policy_repository=mock_policy_repository,
            default=default_policy,
        )

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id="acme-corp",
        )

        with patch("asyncio.run", side_effect=RuntimeError("Database error")):
            policy = provider.get(run)

        assert policy == default_policy

    def test_database_policy_provider_deep_copies_policy(self, mock_policy_repository, default_policy):
        """Test provider returns deep copy of policy to prevent mutation."""
        mock_policy_repository.get = MagicMock(return_value=None)

        provider = DatabasePolicyProvider(
            policy_repository=mock_policy_repository,
            default=default_policy,
        )

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id=None,
        )

        policy1 = provider.get(run)
        policy2 = provider.get(run)

        # Should be equal but not the same object
        assert policy1 == policy2
        assert policy1 is not policy2


class TestAsyncGetPolicyConfig:
    """Integration tests for async_get_policy_config function."""

    @pytest.fixture
    def mock_policy_repository(self):
        """Create a mock PolicyRepository."""
        return MagicMock(spec=PolicyRepository)

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
    async def test_async_get_policy_config_loads_tenant_policy(self, mock_policy_repository, default_policy):
        """Test async function loads tenant-specific policy from database."""
        tenant_policy_dict = {
            "version": "policy-v1",
            "autonomy_profile": "strict",
            "tool_policy": {
                "allowed_capabilities": ["mcp_call"],
                "allowed_tools": [],
                "blocked_tools": [],
                "allowed_mcp_servers": None,
                "blocked_mcp_servers": [],
            },
            "approval_policy": {
                "require_for_risk_at_or_above": "low",
                "approval_ttl_seconds": 600.0,
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

        mock_policy_repository.get = AsyncMock(return_value=tenant_policy_dict)

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id="acme-corp",
        )

        policy = await async_get_policy_config(mock_policy_repository, run, default_policy)

        assert policy.autonomy_profile == AutonomyProfile.strict
        mock_policy_repository.get.assert_called_once_with("acme-corp")

    @pytest.mark.asyncio
    async def test_async_get_policy_config_returns_default_when_no_tenant(self, mock_policy_repository, default_policy):
        """Test async function returns default when run has no tenant_id."""
        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id=None,
        )

        policy = await async_get_policy_config(mock_policy_repository, run, default_policy)

        assert policy == default_policy
        mock_policy_repository.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_get_policy_config_fallback_on_missing_policy(self, mock_policy_repository, default_policy):
        """Test async function falls back to default when policy not found."""
        mock_policy_repository.get = AsyncMock(return_value=None)

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id="unknown-tenant",
        )

        policy = await async_get_policy_config(mock_policy_repository, run, default_policy)

        assert policy == default_policy

    @pytest.mark.asyncio
    async def test_async_get_policy_config_fallback_on_invalid_config(self, mock_policy_repository, default_policy):
        """Test async function falls back to default when config is invalid."""
        invalid_config = {"invalid": "config"}

        mock_policy_repository.get = AsyncMock(return_value=invalid_config)

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id="acme-corp",
        )

        policy = await async_get_policy_config(mock_policy_repository, run, default_policy)

        assert policy == default_policy

    @pytest.mark.asyncio
    async def test_async_get_policy_config_fallback_on_repository_error(self, mock_policy_repository, default_policy):
        """Test async function falls back to default when repository raises error."""
        mock_policy_repository.get = AsyncMock(side_effect=RuntimeError("Database error"))

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id="acme-corp",
        )

        policy = await async_get_policy_config(mock_policy_repository, run, default_policy)

        assert policy == default_policy

    @pytest.mark.asyncio
    async def test_async_get_policy_config_with_tool_risk_overrides(self, mock_policy_repository, default_policy):
        """Test async function loads policy with tool risk overrides."""
        tenant_policy_dict = {
            "version": "policy-v1",
            "autonomy_profile": "balanced",
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
                "tool_risk_overrides": {"web_search": "low", "delete_file": "high"},
                "tool_risk_kinds": {"read_file": "read", "write_file": "write"},
            },
            "safety_policy": {
                "block_prompt_injection": True,
                "redact_secrets": True,
                "max_tool_args_bytes": 64000,
            },
            "budget_policy": {"max_total_tokens": 100000},
        }

        mock_policy_repository.get = AsyncMock(return_value=tenant_policy_dict)

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id="acme-corp",
        )

        policy = await async_get_policy_config(mock_policy_repository, run, default_policy)

        assert policy.approval_policy.tool_risk_overrides["web_search"] == RiskLevel.low
        assert policy.approval_policy.tool_risk_overrides["delete_file"] == RiskLevel.high
        assert policy.approval_policy.tool_risk_kinds["read_file"] == ToolRiskKind.read
        assert policy.budget_policy.max_total_tokens == 100000


class TestStaticPolicyProvider:
    """Tests for StaticPolicyProvider (existing implementation)."""

    @pytest.fixture
    def default_policy(self):
        """Create a default policy config."""
        return PolicyConfig(
            version="policy-v1",
            autonomy_profile=AutonomyProfile.balanced,
        )

    @pytest.fixture
    def tenant_policy(self):
        """Create a tenant-specific policy config."""
        return PolicyConfig(
            version="policy-v1",
            autonomy_profile=AutonomyProfile.strict,
        )

    def test_static_policy_provider_returns_default(self, default_policy):
        """Test StaticPolicyProvider returns default when no overrides."""
        provider = StaticPolicyProvider(default=default_policy)

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id="acme-corp",
        )

        policy = provider.get(run)

        assert policy.autonomy_profile == AutonomyProfile.balanced

    def test_static_policy_provider_returns_tenant_override(self, default_policy, tenant_policy):
        """Test StaticPolicyProvider returns tenant override when available."""
        provider = StaticPolicyProvider(
            default=default_policy,
            by_tenant={"acme-corp": tenant_policy},
        )

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id="acme-corp",
        )

        policy = provider.get(run)

        assert policy.autonomy_profile == AutonomyProfile.strict

    def test_static_policy_provider_returns_workspace_override(self, default_policy, tenant_policy):
        """Test StaticPolicyProvider returns workspace override when available."""
        provider = StaticPolicyProvider(
            default=default_policy,
            by_workspace={"workspace-1": tenant_policy},
        )

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            workspace_id="workspace-1",
        )

        policy = provider.get(run)

        assert policy.autonomy_profile == AutonomyProfile.strict

    def test_static_policy_provider_tenant_takes_precedence(self, default_policy, tenant_policy):
        """Test StaticPolicyProvider prioritizes tenant over workspace override."""
        workspace_policy = PolicyConfig(
            version="policy-v1",
            autonomy_profile=AutonomyProfile.unrestricted,
        )

        provider = StaticPolicyProvider(
            default=default_policy,
            by_tenant={"acme-corp": tenant_policy},
            by_workspace={"workspace-1": workspace_policy},
        )

        run = AgentRun(
            id="test-run",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
            tenant_id="acme-corp",
            workspace_id="workspace-1",
        )

        policy = provider.get(run)

        # Tenant should take precedence
        assert policy.autonomy_profile == AutonomyProfile.strict
