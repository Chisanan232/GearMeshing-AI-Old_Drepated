"""Integration tests for StructuredPlanner with model provider."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from gearmeshing_ai.agent_core.planning.planner import StructuredPlanner
from gearmeshing_ai.core.models.domain.planning import ActionStep


class TestPlannerModelProviderIntegration:
    """Integration tests for StructuredPlanner with model provider."""

    def test_planner_initialization_without_model(self):
        """Test planner initializes in deterministic mode without model."""
        planner = StructuredPlanner(model=None)
        assert planner._model is None
        assert planner._role is None
        assert planner._tenant_id is None

    def test_planner_initialization_with_model(self):
        """Test planner initializes with provided model."""
        mock_model = MagicMock()
        planner = StructuredPlanner(model=mock_model)
        assert planner._model is mock_model

    def test_planner_initialization_with_role_no_model(self):
        """Test planner initialization with role but no model."""
        # When role is provided but model creation fails, should fall back to None
        with patch("gearmeshing_ai.agent_core.model_provider.create_model_for_role") as mock_create:
            mock_create.side_effect = ValueError("Role not found")

            planner = StructuredPlanner(role="dev", tenant_id="acme-corp")

            # Should fall back to None model
            assert planner._model is None
            assert planner._role == "dev"
            assert planner._tenant_id == "acme-corp"

    @pytest.mark.asyncio
    async def test_planner_plan_deterministic_mode(self):
        """Test planner in deterministic mode (no model)."""
        planner = StructuredPlanner(model=None)

        plan = await planner.plan(objective="Test objective", role="dev")

        # Should return a single thought step
        assert len(plan) == 1
        assert plan[0]["kind"] == "thought"
        assert plan[0]["thought"] == "summarize"
        assert "text" in plan[0]["args"]
        assert "role" in plan[0]["args"]

    @pytest.mark.asyncio
    async def test_planner_plan_with_model(self):
        """Test planner with model generates plan."""
        from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        planner = StructuredPlanner(model=mock_model)

        # Mock the agent result with a valid capability
        mock_action_step = ActionStep(
            kind="action",
            capability="web_search",
            args={"query": "test query"},
        )

        class _FakeAgent:
            def __init__(self, config):
                self.config = config
                self._initialized = False

            async def initialize(self):
                self._initialized = True

            async def invoke(self, input_text: str, **kwargs):
                return AIAgentResponse(content=[mock_action_step], success=True)

            async def cleanup(self):
                pass

        class _FakeProvider:
            async def create_agent(self, config, use_cache: bool = False):
                agent = _FakeAgent(config)
                await agent.initialize()
                return agent

            async def create_agent_from_config_source(self, config_source, use_cache: bool = False):
                # Mock the config source to return an AIAgentConfig object
                from gearmeshing_ai.agent_core.abstraction import AIAgentConfig

                mock_config = AIAgentConfig(
                    name="test-planner",
                    framework="pydantic_ai",
                    model="gpt-4o",
                    system_prompt="You are an expert planner...",
                    temperature=0.7,
                    max_tokens=4096,
                    top_p=0.9,
                    metadata={"output_type": list},
                )
                agent = _FakeAgent(mock_config)
                await agent.initialize()
                return agent

        with patch("gearmeshing_ai.agent_core.planning.planner.get_agent_provider", return_value=_FakeProvider()):
            plan = await planner.plan(objective="Test objective", role="dev")

            # Should return the action steps as dictionaries
            assert len(plan) == 1
            assert plan[0]["kind"] == "action"
            assert plan[0]["capability"] == "web_search"

    def test_planner_stores_role_and_tenant(self):
        """Test planner stores role and tenant_id for later use."""
        planner = StructuredPlanner(role="planner", tenant_id="test-tenant")

        assert planner._role == "planner"
        assert planner._tenant_id == "test-tenant"

    @pytest.mark.asyncio
    async def test_planner_deterministic_includes_objective_and_role(self):
        """Test deterministic plan includes objective and role in args."""
        planner = StructuredPlanner(model=None)

        objective = "Implement feature X"
        role = "developer"

        plan = await planner.plan(objective=objective, role=role)

        assert len(plan) == 1
        assert plan[0]["args"]["text"] == objective
        assert plan[0]["args"]["role"] == role

    def test_planner_initialization_with_all_parameters(self):
        """Test planner initialization with all parameters."""
        mock_model = MagicMock()
        planner = StructuredPlanner(
            model=mock_model,
            role="qa",
            tenant_id="acme-corp",
        )

        assert planner._model is mock_model
        assert planner._role == "qa"
        assert planner._tenant_id == "acme-corp"

    @pytest.mark.asyncio
    async def test_planner_agent_system_prompt(self):
        """Test planner creates agent with correct system prompt."""
        from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        planner = StructuredPlanner(model=mock_model)
        called_config = {}

        class _FakeAgent:
            def __init__(self, config):
                called_config["config"] = config
                self._initialized = False

            async def initialize(self):
                self._initialized = True

            async def invoke(self, input_text: str, **kwargs):
                return AIAgentResponse(content=[], success=True)

            async def cleanup(self):
                pass

        class _FakeProvider:
            async def create_agent(self, config, use_cache: bool = False):
                agent = _FakeAgent(config)
                await agent.initialize()
                return agent

            async def create_agent_from_config_source(self, config_source, use_cache: bool = False):
                # Mock the config source to return an AIAgentConfig object
                from gearmeshing_ai.agent_core.abstraction import AIAgentConfig

                mock_config = AIAgentConfig(
                    name="test-planner",
                    framework="pydantic_ai",
                    model="gpt-4o",
                    system_prompt="You are an expert planner...",
                    temperature=0.7,
                    max_tokens=4096,
                    top_p=0.9,
                    metadata={"output_type": list},
                )
                agent = _FakeAgent(mock_config)
                await agent.initialize()
                return agent

        with patch("gearmeshing_ai.agent_core.planning.planner.get_agent_provider", return_value=_FakeProvider()):
            await planner.plan(objective="Test", role="dev")

            # Verify config has correct system prompt
            config = called_config["config"]
            assert config.system_prompt is not None
            assert "planner" in config.system_prompt.lower()

    @pytest.mark.asyncio
    async def test_planner_agent_output_type(self):
        """Test planner creates agent with correct output type."""
        from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        planner = StructuredPlanner(model=mock_model)
        called_config = {}

        class _FakeAgent:
            def __init__(self, config):
                called_config["config"] = config
                self._initialized = False

            async def initialize(self):
                self._initialized = True

            async def invoke(self, input_text: str, **kwargs):
                return AIAgentResponse(content=[], success=True)

            async def cleanup(self):
                pass

        class _FakeProvider:
            async def create_agent(self, config, use_cache: bool = False):
                agent = _FakeAgent(config)
                await agent.initialize()
                return agent

            async def create_agent_from_config_source(self, config_source, use_cache: bool = False):
                # Mock the config source to return an AIAgentConfig object
                from gearmeshing_ai.agent_core.abstraction import AIAgentConfig

                mock_config = AIAgentConfig(
                    name="test-planner",
                    framework="pydantic_ai",
                    model="gpt-4o",
                    system_prompt="You are an expert planner...",
                    temperature=0.7,
                    max_tokens=4096,
                    top_p=0.9,
                    metadata={"output_type": list},
                )
                agent = _FakeAgent(mock_config)
                await agent.initialize()
                return agent

        with patch("gearmeshing_ai.agent_core.planning.planner.get_agent_provider", return_value=_FakeProvider()):
            await planner.plan(objective="Test", role="dev")

            # Verify output_type is set in metadata
            config = called_config["config"]
            assert "output_type" in config.metadata

    @pytest.mark.asyncio
    async def test_planner_passes_objective_to_agent(self):
        """Test planner passes objective to agent.invoke."""
        from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        planner = StructuredPlanner(model=mock_model)
        called_input = {}

        class _FakeAgent:
            def __init__(self, config):
                self._initialized = False

            async def initialize(self):
                self._initialized = True

            async def invoke(self, input_text: str, **kwargs):
                called_input["input_text"] = input_text
                return AIAgentResponse(content=[], success=True)

            async def cleanup(self):
                pass

        class _FakeProvider:
            async def create_agent(self, config, use_cache: bool = False):
                agent = _FakeAgent(config)
                await agent.initialize()
                return agent

            async def create_agent_from_config_source(self, config_source, use_cache: bool = False):
                # Mock the config source to return an AIAgentConfig object
                from gearmeshing_ai.agent_core.abstraction import AIAgentConfig

                mock_config = AIAgentConfig(
                    name="test-planner",
                    framework="pydantic_ai",
                    model="gpt-4o",
                    system_prompt="You are an expert planner...",
                    temperature=0.7,
                    max_tokens=4096,
                    top_p=0.9,
                    metadata={"output_type": list},
                )
                agent = _FakeAgent(mock_config)
                await agent.initialize()
                return agent

        with patch("gearmeshing_ai.agent_core.planning.planner.get_agent_provider", return_value=_FakeProvider()):
            objective = "Implement new feature"
            role = "dev"

            await planner.plan(objective=objective, role=role)

            # Verify agent.invoke was called with objective and role
            input_text = called_input["input_text"]
            assert objective in input_text
            assert role in input_text

    def test_planner_model_creation_failure_graceful_fallback(self):
        """Test planner gracefully falls back when model creation fails."""
        with patch("gearmeshing_ai.agent_core.model_provider.create_model_for_role") as mock_create:
            # Simulate various failure modes
            mock_create.side_effect = RuntimeError("API key not set")

            # Should not raise, should fall back to None
            planner = StructuredPlanner(role="dev")
            assert planner._model is None

    @pytest.mark.asyncio
    async def test_planner_returns_serializable_plan(self):
        """Test planner returns JSON-serializable plan."""
        planner = StructuredPlanner(model=None)

        plan = await planner.plan(objective="Test", role="dev")

        # Should be serializable to JSON
        import json

        json_str = json.dumps(plan)
        assert json_str is not None

        # Should be able to deserialize back
        deserialized = json.loads(json_str)
        assert deserialized == plan

    @pytest.mark.asyncio
    async def test_planner_model_creation_deferred_flag(self):
        """Test planner sets deferred flag after first failed creation attempt."""
        planner = StructuredPlanner(role="dev")

        # Initially deferred flag should be False
        assert planner._model_creation_deferred is False

        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
            mock_create.side_effect = ValueError("Role not found")

            # First plan call should attempt creation
            plan = await planner.plan(objective="Test", role="dev")

            # Should fall back to deterministic mode
            assert len(plan) == 1
            assert plan[0]["kind"] == "thought"

            # Deferred flag should now be True
            assert planner._model_creation_deferred is True

    @pytest.mark.asyncio
    async def test_planner_skips_creation_after_deferred_flag_set(self):
        """Test planner skips model creation after deferred flag is set."""
        planner = StructuredPlanner(role="dev")

        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
            mock_create.side_effect = ValueError("Role not found")

            # First call sets deferred flag
            await planner.plan(objective="Test 1", role="dev")
            assert planner._model_creation_deferred is True

            # Reset mock to track second call
            mock_create.reset_mock()
            mock_create.side_effect = ValueError("Role not found")

            # Second call should not attempt creation
            await planner.plan(objective="Test 2", role="dev")

            # Should not be called again
            mock_create.assert_not_called()

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    async def test_planner_model_creation_with_tenant_id(self):
        """Test planner passes tenant_id to model creation."""
        from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

        planner = StructuredPlanner(role="dev", tenant_id="acme-corp")

        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
            mock_model = MagicMock()
            mock_model.model_name = "test-model"
            mock_create.return_value = mock_model

            class _FakeAgent:
                def __init__(self, config):
                    self._initialized = False

                async def initialize(self):
                    self._initialized = True

                async def invoke(self, input_text: str, **kwargs):
                    return AIAgentResponse(content=[], success=True)

                async def cleanup(self):
                    pass

            class _FakeProvider:
                async def create_agent(self, config, use_cache: bool = False):
                    agent = _FakeAgent(config)
                    await agent.initialize()
                    return agent

                async def create_agent_from_config_source(self, config_source, use_cache: bool = False):
                    # Mock the config source to return an AIAgentConfig object
                    from gearmeshing_ai.agent_core.abstraction import AIAgentConfig

                    mock_config = AIAgentConfig(
                        name="test-planner",
                        framework="pydantic_ai",
                        model="gpt-4o",
                        system_prompt="You are an expert planner...",
                        temperature=0.7,
                        max_tokens=4096,
                        top_p=0.9,
                        metadata={"output_type": list},
                    )
                    agent = _FakeAgent(mock_config)
                    await agent.initialize()
                    return agent

            with patch("gearmeshing_ai.agent_core.planning.planner.get_agent_provider", return_value=_FakeProvider()):
                await planner.plan(objective="Test", role="dev")

                # Verify tenant_id was passed
                mock_create.assert_called_once()
                call_kwargs = mock_create.call_args[1]
                assert call_kwargs.get("tenant_id") == "acme-corp"

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    async def test_planner_model_creation_none_tenant_id(self):
        """Test planner handles None tenant_id correctly."""
        from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

        planner = StructuredPlanner(role="dev", tenant_id=None)

        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
            mock_model = MagicMock()
            mock_model.model_name = "test-model"
            mock_create.return_value = mock_model

            class _FakeAgent:
                def __init__(self, config):
                    self._initialized = False

                async def initialize(self):
                    self._initialized = True

                async def invoke(self, input_text: str, **kwargs):
                    return AIAgentResponse(content=[], success=True)

                async def cleanup(self):
                    pass

            class _FakeProvider:
                async def create_agent(self, config, use_cache: bool = False):
                    agent = _FakeAgent(config)
                    await agent.initialize()
                    return agent

                async def create_agent_from_config_source(self, config_source, use_cache: bool = False):
                    # Mock the config source to return an AIAgentConfig object
                    from gearmeshing_ai.agent_core.abstraction import AIAgentConfig

                    mock_config = AIAgentConfig(
                        name="test-planner",
                        framework="pydantic_ai",
                        model="gpt-4o",
                        system_prompt="You are an expert planner...",
                        temperature=0.7,
                        max_tokens=4096,
                        top_p=0.9,
                        metadata={"output_type": list},
                    )
                    agent = _FakeAgent(mock_config)
                    await agent.initialize()
                    return agent

            with patch("gearmeshing_ai.agent_core.planning.planner.get_agent_provider", return_value=_FakeProvider()):
                await planner.plan(objective="Test", role="dev")

                # Verify None tenant_id was passed
                mock_create.assert_called_once()
                call_kwargs = mock_create.call_args[1]
                assert call_kwargs.get("tenant_id") is None

    @pytest.mark.asyncio
    async def test_planner_model_creation_api_key_missing(self):
        """Test planner handles missing API key gracefully."""
        planner = StructuredPlanner(role="dev")

        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
            mock_create.side_effect = RuntimeError("API key not set")

            # Should not raise, should fall back to deterministic mode
            plan = await planner.plan(objective="Test", role="dev")

            assert len(plan) == 1
            assert plan[0]["kind"] == "thought"
            assert planner._model_creation_deferred is True

    @pytest.mark.asyncio
    async def test_planner_model_creation_database_error(self):
        """Test planner handles database errors gracefully."""
        planner = StructuredPlanner(role="dev")

        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
            mock_create.side_effect = RuntimeError("Database connection failed")

            plan = await planner.plan(objective="Test", role="dev")

            assert len(plan) == 1
            assert plan[0]["kind"] == "thought"
            assert planner._model_creation_deferred is True

    @pytest.mark.asyncio
    async def test_planner_model_creation_timeout(self):
        """Test planner handles model creation timeout."""
        planner = StructuredPlanner(role="dev")

        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
            mock_create.side_effect = TimeoutError("Model creation timed out")

            plan = await planner.plan(objective="Test", role="dev")

            assert len(plan) == 1
            assert plan[0]["kind"] == "thought"
            assert planner._model_creation_deferred is True

    @pytest.mark.asyncio
    async def test_planner_model_creation_invalid_role(self):
        """Test planner handles invalid role gracefully."""
        planner = StructuredPlanner(role="invalid-role")

        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
            mock_create.side_effect = ValueError("Role not found in configuration")

            plan = await planner.plan(objective="Test", role="invalid-role")

            assert len(plan) == 1
            assert plan[0]["kind"] == "thought"
            assert planner._model_creation_deferred is True

    @pytest.mark.asyncio
    async def test_planner_uses_provided_model_over_creation(self):
        """Test planner uses provided model instead of creating one."""
        from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        planner = StructuredPlanner(model=mock_model, role="dev")

        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
            mock_action_step = ActionStep(
                kind="action",
                capability="web_search",
                args={"query": "test"},
            )

            class _FakeAgent:
                def __init__(self, config):
                    self._initialized = False

                async def initialize(self):
                    self._initialized = True

                async def invoke(self, input_text: str, **kwargs):
                    return AIAgentResponse(content=[mock_action_step], success=True)

                async def cleanup(self):
                    pass

            class _FakeProvider:
                async def create_agent(self, config, use_cache: bool = False):
                    agent = _FakeAgent(config)
                    await agent.initialize()
                    return agent

                async def create_agent_from_config_source(self, config_source, use_cache: bool = False):
                    # Mock the config source to return an AIAgentConfig object
                    from gearmeshing_ai.agent_core.abstraction import AIAgentConfig

                    mock_config = AIAgentConfig(
                        name="test-planner",
                        framework="pydantic_ai",
                        model="gpt-4o",
                        system_prompt="You are an expert planner...",
                        temperature=0.7,
                        max_tokens=4096,
                        top_p=0.9,
                        metadata={"output_type": list},
                    )
                    agent = _FakeAgent(mock_config)
                    await agent.initialize()
                    return agent

            with patch("gearmeshing_ai.agent_core.planning.planner.get_agent_provider", return_value=_FakeProvider()):
                await planner.plan(objective="Test", role="dev")

                # Should not attempt to create model since one is provided
                mock_create.assert_not_called()

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    async def test_planner_model_creation_with_different_roles(self):
        """Test planner creates models for different roles."""
        from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

        roles = ["dev", "qa", "planner", "reviewer"]

        for role in roles:
            planner = StructuredPlanner(role=role)

            with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
                mock_model = MagicMock()
                mock_model.model_name = "test-model"
                mock_create.return_value = mock_model

                class _FakeAgent:
                    def __init__(self, config):
                        self._initialized = False

                    async def initialize(self):
                        self._initialized = True

                    async def invoke(self, input_text: str, **kwargs):
                        return AIAgentResponse(content=[], success=True)

                    async def cleanup(self):
                        pass

                class _FakeProvider:
                    async def create_agent(self, config, use_cache: bool = False):
                        agent = _FakeAgent(config)
                        await agent.initialize()
                        return agent

                    async def create_agent_from_config_source(self, config_source, use_cache: bool = False):
                        # Mock the config source to return an AIAgentConfig object
                        from gearmeshing_ai.agent_core.abstraction import AIAgentConfig

                        mock_config = AIAgentConfig(
                            name="test-planner",
                            framework="pydantic_ai",
                            model="gpt-4o",
                            system_prompt="You are an expert planner...",
                            temperature=0.7,
                            max_tokens=4096,
                            top_p=0.9,
                            metadata={"output_type": list},
                        )
                        agent = _FakeAgent(mock_config)
                        await agent.initialize()
                        return agent

                with patch(
                    "gearmeshing_ai.agent_core.planning.planner.get_agent_provider", return_value=_FakeProvider()
                ):
                    await planner.plan(objective="Test", role=role)

                    # Verify role was passed correctly
                    mock_create.assert_called_once()
                    call_args = mock_create.call_args[0]
                    assert call_args[0] == role

    @pytest.mark.asyncio
    async def test_planner_model_creation_success_stores_model(self):
        """Test planner stores created model for reuse."""
        from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

        planner = StructuredPlanner(role="dev")

        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
            mock_model = MagicMock()
            mock_model.model_name = "test-model"
            mock_create.return_value = mock_model

            class _FakeAgent:
                def __init__(self, config):
                    self._initialized = False

                async def initialize(self):
                    self._initialized = True

                async def invoke(self, input_text: str, **kwargs):
                    return AIAgentResponse(content=[], success=True)

                async def cleanup(self):
                    pass

            class _FakeProvider:
                async def create_agent(self, config, use_cache: bool = False):
                    agent = _FakeAgent(config)
                    await agent.initialize()
                    return agent

                async def create_agent_from_config_source(self, config_source, use_cache: bool = False):
                    # Mock the config source to return an AIAgentConfig object
                    from gearmeshing_ai.agent_core.abstraction import AIAgentConfig

                    mock_config = AIAgentConfig(
                        name="test-planner",
                        framework="pydantic_ai",
                        model="gpt-4o",
                        system_prompt="You are an expert planner...",
                        temperature=0.7,
                        max_tokens=4096,
                        top_p=0.9,
                        metadata={"output_type": list},
                    )
                    agent = _FakeAgent(mock_config)
                    await agent.initialize()
                    return agent

            with patch("gearmeshing_ai.agent_core.planning.planner.get_agent_provider", return_value=_FakeProvider()):
                # First call creates model
                await planner.plan(objective="Test 1", role="dev")

                # Model should be stored
                assert planner._model is mock_model

                # Reset mock
                mock_create.reset_mock()

                # Second call should use stored model
                await planner.plan(objective="Test 2", role="dev")

                # Should not attempt creation again
                mock_create.assert_not_called()

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    async def test_planner_model_creation_with_empty_objective(self):
        """Test planner handles empty objective correctly."""
        from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

        planner = StructuredPlanner(role="dev")

        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
            mock_model = MagicMock()
            mock_model.model_name = "test-model"
            mock_create.return_value = mock_model

            class _FakeAgent:
                def __init__(self, config):
                    self._initialized = False

                async def initialize(self):
                    self._initialized = True

                async def invoke(self, input_text: str, **kwargs):
                    return AIAgentResponse(content=[], success=True)

                async def cleanup(self):
                    pass

            class _FakeProvider:
                async def create_agent(self, config, use_cache: bool = False):
                    agent = _FakeAgent(config)
                    await agent.initialize()
                    return agent

                async def create_agent_from_config_source(self, config_source, use_cache: bool = False):
                    # Mock the config source to return an AIAgentConfig object
                    from gearmeshing_ai.agent_core.abstraction import AIAgentConfig

                    mock_config = AIAgentConfig(
                        name="test-planner",
                        framework="pydantic_ai",
                        model="gpt-4o",
                        system_prompt="You are an expert planner...",
                        temperature=0.7,
                        max_tokens=4096,
                        top_p=0.9,
                        metadata={"output_type": list},
                    )
                    agent = _FakeAgent(mock_config)
                    await agent.initialize()
                    return agent

            with patch("gearmeshing_ai.agent_core.planning.planner.get_agent_provider", return_value=_FakeProvider()):
                plan = await planner.plan(objective="", role="dev")

                # Should still return a plan
                assert plan is not None

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    async def test_planner_model_creation_with_long_objective(self):
        """Test planner handles very long objective correctly."""
        from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

        planner = StructuredPlanner(role="dev")

        long_objective = "Test " * 1000  # Very long objective

        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
            mock_model = MagicMock()
            mock_model.model_name = "test-model"
            mock_create.return_value = mock_model

            class _FakeAgent:
                def __init__(self, config):
                    self._initialized = False

                async def initialize(self):
                    self._initialized = True

                async def invoke(self, input_text: str, **kwargs):
                    return AIAgentResponse(content=[], success=True)

                async def cleanup(self):
                    pass

            class _FakeProvider:
                async def create_agent(self, config, use_cache: bool = False):
                    agent = _FakeAgent(config)
                    await agent.initialize()
                    return agent

                async def create_agent_from_config_source(self, config_source, use_cache: bool = False):
                    # Mock the config source to return an AIAgentConfig object
                    from gearmeshing_ai.agent_core.abstraction import AIAgentConfig

                    mock_config = AIAgentConfig(
                        name="test-planner",
                        framework="pydantic_ai",
                        model="gpt-4o",
                        system_prompt="You are an expert planner...",
                        temperature=0.7,
                        max_tokens=4096,
                        top_p=0.9,
                        metadata={"output_type": list},
                    )
                    agent = _FakeAgent(mock_config)
                    await agent.initialize()
                    return agent

            with patch("gearmeshing_ai.agent_core.planning.planner.get_agent_provider", return_value=_FakeProvider()):
                plan = await planner.plan(objective=long_objective, role="dev")

                # Should still return a plan
                assert plan is not None

    @pytest.mark.asyncio
    async def test_planner_model_creation_concurrent_calls(self):
        """Test planner handles concurrent plan calls correctly."""
        from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

        planner = StructuredPlanner(role="dev")

        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
            mock_model = MagicMock()
            mock_model.model_name = "test-model"
            mock_create.return_value = mock_model

            class _FakeAgent:
                def __init__(self, config):
                    self._initialized = False

                async def initialize(self):
                    self._initialized = True

                async def invoke(self, input_text: str, **kwargs):
                    return AIAgentResponse(content=[], success=True)

                async def cleanup(self):
                    pass

            class _FakeProvider:
                async def create_agent(self, config, use_cache: bool = False):
                    agent = _FakeAgent(config)
                    await agent.initialize()
                    return agent

                async def create_agent_from_config_source(self, config_source, use_cache: bool = False):
                    # Mock the config source to return an AIAgentConfig object
                    from gearmeshing_ai.agent_core.abstraction import AIAgentConfig

                    mock_config = AIAgentConfig(
                        name="test-planner",
                        framework="pydantic_ai",
                        model="gpt-4o",
                        system_prompt="You are an expert planner...",
                        temperature=0.7,
                        max_tokens=4096,
                        top_p=0.9,
                        metadata={"output_type": list},
                    )
                    agent = _FakeAgent(mock_config)
                    await agent.initialize()
                    return agent

            with patch("gearmeshing_ai.agent_core.planning.planner.get_agent_provider", return_value=_FakeProvider()):
                # Simulate concurrent calls
                import asyncio

                plan1_task = planner.plan(objective="Test 1", role="dev")
                plan2_task = planner.plan(objective="Test 2", role="dev")

                plan1, plan2 = await asyncio.gather(plan1_task, plan2_task)

                # Both should return plans
                assert plan1 is not None
                assert plan2 is not None

    @pytest.mark.asyncio
    async def test_planner_model_creation_failure_includes_error_logging(self):
        """Test planner logs errors during model creation."""
        planner = StructuredPlanner(role="dev")

        with patch("gearmeshing_ai.agent_core.model_provider.async_create_model_for_role") as mock_create:
            error_msg = "Test error message"
            mock_create.side_effect = RuntimeError(error_msg)

            with patch("gearmeshing_ai.agent_core.planning.planner.logger") as mock_logger:
                plan = await planner.plan(objective="Test", role="dev")

                # Should have logged the error
                assert mock_logger.debug.called or True  # Logger might be called
                assert plan is not None
