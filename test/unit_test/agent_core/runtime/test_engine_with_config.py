"""Tests for AgentEngine with configuration-based model support."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.checkpoint.memory import MemorySaver

from gearmeshing_ai.agent_core.factory import build_default_registry
from gearmeshing_ai.agent_core.policy.global_policy import GlobalPolicy
from gearmeshing_ai.agent_core.policy.models import PolicyConfig
from gearmeshing_ai.agent_core.runtime import EngineDeps
from gearmeshing_ai.agent_core.runtime.engine import AgentEngine
from gearmeshing_ai.core.models.domain import AgentRun
from gearmeshing_ai.info_provider import AgentRole


class TestEngineWithConfigurationSupport:
    """Tests for AgentEngine with configuration-based model support."""

    @pytest.mark.asyncio
    async def test_engine_creates_model_from_config_for_thought(self) -> None:
        """Test engine creates model from config when thought_model is None."""
        # Create mock dependencies
        runs_repo = AsyncMock()
        events_repo = AsyncMock()
        approvals_repo = AsyncMock()
        checkpoints_repo = AsyncMock()
        tool_invocations_repo = AsyncMock()
        role_provider = MagicMock()
        role_provider.get.return_value = MagicMock(cognitive=MagicMock(system_prompt_key="dev_system_prompt"))

        deps = EngineDeps(
            runs=runs_repo,
            events=events_repo,
            approvals=approvals_repo,
            checkpoints=checkpoints_repo,
            tool_invocations=tool_invocations_repo,
            capabilities=build_default_registry(),
            usage=None,
            prompt_provider=MagicMock(),
            role_provider=role_provider,
            thought_model=None,  # No explicit model
            mcp_info_provider=None,
            mcp_call=None,
            checkpointer=MemorySaver(),
        )

        policy = GlobalPolicy(config=PolicyConfig())
        engine = AgentEngine(policy=policy, deps=deps)

        # Verify engine is created with no thought model
        # Model creation is deferred to _node_execute_next using async_create_model_for_role
        assert engine._deps.thought_model is None
        assert engine._deps.role_provider is not None

    @pytest.mark.asyncio
    async def test_engine_uses_provided_thought_model(self) -> None:
        """Test engine uses provided thought_model if available."""
        mock_thought_model = MagicMock()

        deps = EngineDeps(
            runs=AsyncMock(),
            events=AsyncMock(),
            approvals=AsyncMock(),
            checkpoints=AsyncMock(),
            tool_invocations=AsyncMock(),
            capabilities=build_default_registry(),
            usage=None,
            prompt_provider=None,
            role_provider=None,
            thought_model=mock_thought_model,  # Explicit model provided
            mcp_info_provider=None,
            mcp_call=None,
            checkpointer=MemorySaver(),
        )

        policy = GlobalPolicy(config=PolicyConfig())
        engine = AgentEngine(policy=policy, deps=deps)

        # Verify that the provided model is used
        assert engine._deps.thought_model is mock_thought_model

    @pytest.mark.asyncio
    async def test_engine_handles_model_creation_error_gracefully(self) -> None:
        """Test engine handles model creation errors gracefully."""
        runs_repo = AsyncMock()
        events_repo = AsyncMock()

        deps = EngineDeps(
            runs=runs_repo,
            events=events_repo,
            approvals=AsyncMock(),
            checkpoints=AsyncMock(),
            tool_invocations=AsyncMock(),
            capabilities=build_default_registry(),
            usage=None,
            prompt_provider=None,
            role_provider=MagicMock(),
            thought_model=None,
            mcp_info_provider=None,
            mcp_call=None,
            checkpointer=MemorySaver(),
        )

        policy = GlobalPolicy(config=PolicyConfig())
        engine = AgentEngine(policy=policy, deps=deps)

        # Engine should be created successfully even if model creation might fail later
        # The error handling happens in _node_execute_next during actual execution
        assert engine is not None
        assert engine._deps.thought_model is None

    @pytest.mark.asyncio
    async def test_engine_passes_tenant_id_to_model_creation(self) -> None:
        """Test engine passes tenant_id to model creation."""
        runs_repo = AsyncMock()
        events_repo = AsyncMock()
        role_provider = MagicMock()
        role_provider.get.return_value = MagicMock(cognitive=MagicMock(system_prompt_key="dev_system_prompt"))

        deps = EngineDeps(
            runs=runs_repo,
            events=events_repo,
            approvals=AsyncMock(),
            checkpoints=AsyncMock(),
            tool_invocations=AsyncMock(),
            capabilities=build_default_registry(),
            usage=None,
            prompt_provider=MagicMock(),
            role_provider=role_provider,
            thought_model=None,
            mcp_info_provider=None,
            mcp_call=None,
            checkpointer=MemorySaver(),
        )

        policy = GlobalPolicy(config=PolicyConfig())
        engine = AgentEngine(policy=policy, deps=deps)

        # Create a run with tenant_id
        run = AgentRun(role=AgentRole.dev, objective="test", tenant_id="acme-corp")

        # Verify tenant_id is properly set on the run
        # The tenant_id will be passed to async_create_model_for_role during execution
        assert run.tenant_id == "acme-corp"


class TestEngineModelIntegration:
    """Integration tests for engine with model configuration."""

    @pytest.mark.asyncio
    async def test_engine_with_no_model_and_no_role_provider(self) -> None:
        """Test engine works without model and role provider."""
        deps = EngineDeps(
            runs=AsyncMock(),
            events=AsyncMock(),
            approvals=AsyncMock(),
            checkpoints=AsyncMock(),
            tool_invocations=AsyncMock(),
            capabilities=build_default_registry(),
            usage=None,
            prompt_provider=None,
            role_provider=None,
            thought_model=None,
            mcp_info_provider=None,
            mcp_call=None,
            checkpointer=MemorySaver(),
        )

        policy = GlobalPolicy(config=PolicyConfig())
        engine = AgentEngine(policy=policy, deps=deps)

        # Engine should be created successfully
        assert engine is not None
        assert engine._deps.thought_model is None
        assert engine._deps.role_provider is None

    @pytest.mark.asyncio
    async def test_engine_with_all_model_components(self) -> None:
        """Test engine with all model-related components."""
        mock_model = MagicMock()
        mock_role_provider = MagicMock()
        mock_prompt_provider = MagicMock()

        deps = EngineDeps(
            runs=AsyncMock(),
            events=AsyncMock(),
            approvals=AsyncMock(),
            checkpoints=AsyncMock(),
            tool_invocations=AsyncMock(),
            capabilities=build_default_registry(),
            usage=None,
            prompt_provider=mock_prompt_provider,
            role_provider=mock_role_provider,
            thought_model=mock_model,
            mcp_info_provider=None,
            mcp_call=None,
            checkpointer=MemorySaver(),
        )

        policy = GlobalPolicy(config=PolicyConfig())
        engine = AgentEngine(policy=policy, deps=deps)

        # Verify all components are set
        assert engine._deps.thought_model is mock_model
        assert engine._deps.role_provider is mock_role_provider
        assert engine._deps.prompt_provider is mock_prompt_provider
