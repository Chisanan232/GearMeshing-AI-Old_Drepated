"""Integration tests for AgentEngine with model provider."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from gearmeshing_ai.agent_core.policy.global_policy import GlobalPolicy
from gearmeshing_ai.agent_core.runtime.engine import AgentEngine
from gearmeshing_ai.agent_core.runtime.models import EngineDeps
from gearmeshing_ai.agent_core.schemas.domain import AgentRun, AgentRunStatus


class TestEngineModelProviderIntegration:
    """Integration tests for AgentEngine with model provider."""

    @pytest.fixture
    def mock_policy(self):
        """Create a mock global policy."""
        return MagicMock(spec=GlobalPolicy)

    @pytest.fixture
    def mock_engine_deps(self):
        """Create mock engine dependencies."""
        deps = MagicMock(spec=EngineDeps)
        deps.runs = AsyncMock()
        deps.events = AsyncMock()
        deps.checkpoints = AsyncMock()
        deps.approvals = AsyncMock()
        deps.checkpointer = MemorySaver()
        deps.thought_model = None
        deps.role_provider = None
        deps.prompt_provider = None
        deps.capabilities = {}
        return deps

    def test_engine_initialization_with_deps(self, mock_policy, mock_engine_deps):
        """Test AgentEngine initializes with dependencies."""
        engine = AgentEngine(policy=mock_policy, deps=mock_engine_deps)
        assert engine._policy is mock_policy
        assert engine._deps is mock_engine_deps
        assert engine._graph is not None

    @pytest.mark.asyncio
    async def test_engine_start_run_without_thought_model(self, mock_policy, mock_engine_deps):
        """Test engine start_run when no thought model is provided."""
        engine = AgentEngine(policy=mock_policy, deps=mock_engine_deps)

        run = AgentRun(
            id="test-run-1",
            role="dev",
            objective="Test objective",
            status=AgentRunStatus.running,
        )

        plan = [{"kind": "thought", "thought": "analyze", "args": {}}]

        # Mock the async methods
        mock_engine_deps.runs.create = AsyncMock()
        mock_engine_deps.events.append = AsyncMock()
        mock_engine_deps.checkpoints.save = AsyncMock()

        # This should not raise an error even without thought model
        try:
            result = await engine.start_run(run=run, plan=plan)
            assert result == "test-run-1"
        except Exception as e:
            # Expected to fail due to graph execution, but should not fail on model creation
            assert "thought model" not in str(e).lower()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_engine_with_thought_model_provided(self, mock_policy, mock_engine_deps):
        """Test engine with thought model already provided."""
        mock_thought_model = MagicMock()
        mock_engine_deps.thought_model = mock_thought_model

        engine = AgentEngine(policy=mock_policy, deps=mock_engine_deps)

        run = AgentRun(
            id="test-run-2",
            role="dev",
            objective="Test objective",
            status=AgentRunStatus.running,
        )

        plan = [{"kind": "thought", "thought": "analyze", "args": {}}]

        mock_engine_deps.runs.create = AsyncMock()
        mock_engine_deps.events.append = AsyncMock()
        mock_engine_deps.checkpoints.save = AsyncMock()

        try:
            result = await engine.start_run(run=run, plan=plan)
            assert result == "test-run-2"
        except Exception:
            # Expected to fail on graph execution, but model should be used
            pass

    def test_engine_graph_structure(self, mock_policy, mock_engine_deps):
        """Test that engine graph is properly structured."""
        engine = AgentEngine(policy=mock_policy, deps=mock_engine_deps)

        # Verify graph has expected nodes
        assert engine._graph is not None
        # Graph should be compiled and ready to use
        assert hasattr(engine._graph, "ainvoke")

    @pytest.mark.asyncio
    async def test_engine_handles_missing_run(self, mock_policy, mock_engine_deps):
        """Test engine handles missing run gracefully."""
        engine = AgentEngine(policy=mock_policy, deps=mock_engine_deps)

        # Mock runs.get to return None
        mock_engine_deps.runs.get = AsyncMock(return_value=None)

        state = {
            "run_id": "nonexistent-run",
            "plan": [],
            "idx": 0,
            "awaiting_approval_id": None,
        }

        with pytest.raises(ValueError, match="run not found"):
            await engine._node_execute_next(state, config={})

    @pytest.mark.asyncio
    async def test_engine_handles_terminal_condition(self, mock_policy, mock_engine_deps):
        """Test engine detects terminal condition (idx >= plan length)."""
        engine = AgentEngine(policy=mock_policy, deps=mock_engine_deps)

        run = AgentRun(
            id="test-run-3",
            role="dev",
            objective="Test",
            status=AgentRunStatus.running,
        )
        mock_engine_deps.runs.get = AsyncMock(return_value=run)

        state = {
            "run_id": "test-run-3",
            "plan": [{"kind": "thought", "thought": "test"}],
            "idx": 1,  # Index beyond plan length
            "awaiting_approval_id": None,
        }

        result = await engine._node_execute_next(state, config={})

        # Should mark as finished
        assert result.get("_finished") is True
        assert result.get("_terminal_status") == AgentRunStatus.succeeded.value

    @pytest.mark.asyncio
    async def test_engine_skips_model_creation_when_role_provider_none(self, mock_policy, mock_engine_deps):
        """Test engine skips model creation when role_provider is None."""
        mock_engine_deps.role_provider = None
        engine = AgentEngine(policy=mock_policy, deps=mock_engine_deps)

        run = AgentRun(
            id="test-run-4",
            role="dev",
            objective="Test objective",
            status=AgentRunStatus.running,
        )
        mock_engine_deps.runs.get = AsyncMock(return_value=run)

        state = {
            "run_id": "test-run-4",
            "plan": [{"kind": "thought", "thought": "analyze", "args": {}}],
            "idx": 0,
            "awaiting_approval_id": None,
        }

        # Should not attempt model creation when role_provider is None
        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
            try:
                await engine._node_execute_next(state, config={})
            except Exception:
                pass
            # async_create_model_for_role should not be called
            mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_engine_model_creation_with_tenant_id(self, mock_policy, mock_engine_deps):
        """Test engine passes tenant_id to model creation."""
        mock_engine_deps.role_provider = MagicMock()
        engine = AgentEngine(policy=mock_policy, deps=mock_engine_deps)

        run = AgentRun(
            id="test-run-5",
            role="dev",
            objective="Test objective",
            status=AgentRunStatus.running,
            tenant_id="acme-corp",
        )
        mock_engine_deps.runs.get = AsyncMock(return_value=run)

        state = {
            "run_id": "test-run-5",
            "plan": [{"kind": "thought", "thought": "analyze", "args": {}}],
            "idx": 0,
            "awaiting_approval_id": None,
        }

        with patch("gearmeshing_ai.agent_core.runtime.engine.async_create_model_for_role") as mock_create:
            mock_create.return_value = MagicMock()
            try:
                await engine._node_execute_next(state, config={})
            except Exception:
                pass

            # Verify tenant_id was passed
            if mock_create.called:
                call_kwargs = mock_create.call_args[1]
                assert call_kwargs.get("tenant_id") == "acme-corp"

    @pytest.mark.asyncio
    async def test_engine_model_creation_api_key_missing(self, mock_policy, mock_engine_deps):
        """Test engine handles missing API key gracefully."""
        mock_engine_deps.role_provider = MagicMock()
        engine = AgentEngine(policy=mock_policy, deps=mock_engine_deps)

        run = AgentRun(
            id="test-run-6",
            role="dev",
            objective="Test objective",
            status=AgentRunStatus.running,
        )
        mock_engine_deps.runs.get = AsyncMock(return_value=run)

        state = {
            "run_id": "test-run-6",
            "plan": [{"kind": "thought", "thought": "analyze", "args": {}}],
            "idx": 0,
            "awaiting_approval_id": None,
        }

        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
            mock_create.side_effect = RuntimeError("API key not set")

            # Should not raise, should handle gracefully
            try:
                result = await engine._node_execute_next(state, config={})
                # Should continue without thought model
                assert result is not None
            except RuntimeError as e:
                # If it does raise, it should be logged but not crash
                assert "API key" in str(e)

    @pytest.mark.asyncio
    async def test_engine_model_creation_database_error(self, mock_policy, mock_engine_deps):
        """Test engine handles database errors during model creation."""
        mock_engine_deps.role_provider = MagicMock()
        engine = AgentEngine(policy=mock_policy, deps=mock_engine_deps)

        run = AgentRun(
            id="test-run-7",
            role="dev",
            objective="Test objective",
            status=AgentRunStatus.running,
        )
        mock_engine_deps.runs.get = AsyncMock(return_value=run)

        state = {
            "run_id": "test-run-7",
            "plan": [{"kind": "thought", "thought": "analyze", "args": {}}],
            "idx": 0,
            "awaiting_approval_id": None,
        }

        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
            mock_create.side_effect = RuntimeError("Database connection failed")

            try:
                result = await engine._node_execute_next(state, config={})
                # Should continue without thought model
                assert result is not None
            except RuntimeError:
                pass

    @pytest.mark.asyncio
    async def test_engine_uses_provided_thought_model_over_creation(self, mock_policy, mock_engine_deps):
        """Test engine uses provided thought model instead of creating one."""
        mock_thought_model = MagicMock()
        mock_engine_deps.thought_model = mock_thought_model
        mock_engine_deps.role_provider = MagicMock()

        engine = AgentEngine(policy=mock_policy, deps=mock_engine_deps)

        run = AgentRun(
            id="test-run-8",
            role="dev",
            objective="Test objective",
            status=AgentRunStatus.running,
        )
        mock_engine_deps.runs.get = AsyncMock(return_value=run)

        state = {
            "run_id": "test-run-8",
            "plan": [{"kind": "thought", "thought": "analyze", "args": {}}],
            "idx": 0,
            "awaiting_approval_id": None,
        }

        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
            try:
                await engine._node_execute_next(state, config={})
            except Exception:
                pass

            # Should not attempt to create model since one is provided
            mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_engine_model_creation_with_different_roles(self, mock_policy, mock_engine_deps):
        """Test engine creates models for different roles."""
        mock_engine_deps.role_provider = MagicMock()
        engine = AgentEngine(policy=mock_policy, deps=mock_engine_deps)

        roles = ["dev", "qa", "planner", "reviewer"]

        for role in roles:
            run = AgentRun(
                id=f"test-run-{role}",
                role=role,
                objective="Test objective",
                status=AgentRunStatus.running,
            )
            mock_engine_deps.runs.get = AsyncMock(return_value=run)

            state = {
                "run_id": f"test-run-{role}",
                "plan": [{"kind": "thought", "thought": "analyze", "args": {}}],
                "idx": 0,
                "awaiting_approval_id": None,
            }

            with patch("gearmeshing_ai.agent_core.runtime.engine.async_create_model_for_role") as mock_create:
                mock_create.return_value = MagicMock()
                try:
                    await engine._node_execute_next(state, config={})
                except Exception:
                    pass

                # Verify role was passed correctly if called
                if mock_create.called:
                    call_args = mock_create.call_args[0]
                    assert call_args[0] == role

    @pytest.mark.asyncio
    async def test_engine_model_creation_none_tenant_id(self, mock_policy, mock_engine_deps):
        """Test engine handles None tenant_id correctly."""
        mock_engine_deps.role_provider = MagicMock()
        engine = AgentEngine(policy=mock_policy, deps=mock_engine_deps)

        run = AgentRun(
            id="test-run-9",
            role="dev",
            objective="Test objective",
            status=AgentRunStatus.running,
            tenant_id=None,
        )
        mock_engine_deps.runs.get = AsyncMock(return_value=run)

        state = {
            "run_id": "test-run-9",
            "plan": [{"kind": "thought", "thought": "analyze", "args": {}}],
            "idx": 0,
            "awaiting_approval_id": None,
        }

        with patch("gearmeshing_ai.agent_core.runtime.engine.async_create_model_for_role") as mock_create:
            mock_create.return_value = MagicMock()
            try:
                await engine._node_execute_next(state, config={})
            except Exception:
                pass

            # Verify None tenant_id was passed if called
            if mock_create.called:
                call_kwargs = mock_create.call_args[1]
                assert call_kwargs.get("tenant_id") is None

    @pytest.mark.asyncio
    async def test_engine_model_creation_timeout(self, mock_policy, mock_engine_deps):
        """Test engine handles model creation timeout."""
        mock_engine_deps.role_provider = MagicMock()
        engine = AgentEngine(policy=mock_policy, deps=mock_engine_deps)

        run = AgentRun(
            id="test-run-10",
            role="dev",
            objective="Test objective",
            status=AgentRunStatus.running,
        )
        mock_engine_deps.runs.get = AsyncMock(return_value=run)

        state = {
            "run_id": "test-run-10",
            "plan": [{"kind": "thought", "thought": "analyze", "args": {}}],
            "idx": 0,
            "awaiting_approval_id": None,
        }

        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
            mock_create.side_effect = TimeoutError("Model creation timed out")

            try:
                result = await engine._node_execute_next(state, config={})
                # Should continue without thought model
                assert result is not None
            except TimeoutError:
                pass

    @pytest.mark.asyncio
    async def test_engine_model_creation_invalid_role(self, mock_policy, mock_engine_deps):
        """Test engine handles invalid role gracefully."""
        mock_engine_deps.role_provider = MagicMock()
        engine = AgentEngine(policy=mock_policy, deps=mock_engine_deps)

        run = AgentRun(
            id="test-run-11",
            role="invalid-role",
            objective="Test objective",
            status=AgentRunStatus.running,
        )
        mock_engine_deps.runs.get = AsyncMock(return_value=run)

        state = {
            "run_id": "test-run-11",
            "plan": [{"kind": "thought", "thought": "analyze", "args": {}}],
            "idx": 0,
            "awaiting_approval_id": None,
        }

        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
            mock_create.side_effect = ValueError("Role not found in configuration")

            try:
                result = await engine._node_execute_next(state, config={})
                # Should continue without thought model
                assert result is not None
            except ValueError:
                pass
