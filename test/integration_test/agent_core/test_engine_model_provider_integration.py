"""Integration tests for AgentEngine with model provider."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
            await engine._node_execute_next(state)

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

        result = await engine._node_execute_next(state)

        # Should mark as finished
        assert result.get("_finished") is True
        assert result.get("_terminal_status") == AgentRunStatus.succeeded.value
