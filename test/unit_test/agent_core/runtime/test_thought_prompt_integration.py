from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest
from langgraph.checkpoint.memory import MemorySaver

from gearmeshing_ai.agent_core.capabilities.registry import CapabilityRegistry
from gearmeshing_ai.agent_core.policy.global_policy import GlobalPolicy
from gearmeshing_ai.agent_core.policy.models import PolicyConfig
from gearmeshing_ai.agent_core.role_provider import DEFAULT_ROLE_PROVIDER
from gearmeshing_ai.agent_core.runtime.engine import AgentEngine
from gearmeshing_ai.agent_core.runtime.models import EngineDeps
from gearmeshing_ai.agent_core.schemas.domain import AgentEvent, AgentRun


class _Runs:
    def __init__(self) -> None:
        self.by_id: Dict[str, AgentRun] = {}

    async def create(self, run: AgentRun) -> None:
        self.by_id[run.id] = run

    async def get(self, run_id: str) -> Optional[AgentRun]:
        return self.by_id.get(run_id)

    async def update_status(self, run_id: str, *, status: str) -> None:
        run = self.by_id.get(run_id)
        if run is not None:
            run.status = status  # type: ignore[assignment]


class _Events:
    def __init__(self) -> None:
        self.events: List[AgentEvent] = []

    async def append(self, event: AgentEvent) -> None:
        self.events.append(event)


class _NoopRepo:
    async def create(self, *_a, **_k):
        return None

    async def get(self, *_a, **_k):
        return None

    async def save(self, *_a, **_k):
        return None

    async def latest(self, *_a, **_k):
        return None

    async def append(self, *_a, **_k):
        return None


@dataclass
class _PromptProvider:
    prompt: str

    def get(self, name: str, locale: str = "en", tenant: str | None = None) -> str:
        return self.prompt

    def version(self) -> str:
        return "v-test"

    def refresh(self) -> None:
        return None


@pytest.mark.asyncio
async def test_thought_step_uses_prompt_provider_and_model_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    called: Dict[str, Any] = {}

    from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

    class _FakeAgent:
        def __init__(self, config: Any) -> None:
            called["system_prompt"] = config.system_prompt
            called["output_type"] = config.metadata.get("output_type")
            called["model"] = config.model
            self._initialized = False

        async def initialize(self) -> None:
            self._initialized = True

        async def invoke(self, input_text: str, **kwargs: Any) -> AIAgentResponse:
            called["prompt"] = input_text
            return AIAgentResponse(content={"ok": True}, success=True)

        async def cleanup(self) -> None:
            pass

    class _FakeProvider:
        async def create_agent(self, config: Any, use_cache: bool = False) -> _FakeAgent:
            agent = _FakeAgent(config)
            await agent.initialize()
            return agent
        
        async def create_agent_from_config_source(self, config_source: Any, use_cache: bool = False) -> _FakeAgent:
            # Mock the config source to return an AIAgentConfig object
            from gearmeshing_ai.agent_core.abstraction import AIAgentConfig
            
            # Start with base config
            config_dict = {
                "name": "test-thought",
                "framework": "pydantic_ai",
                "model": "gpt-4o",
                "system_prompt": "You are a helpful assistant...",  # Default
                "temperature": 0.7,
                "max_tokens": 4096,
                "top_p": 0.9,
                "metadata": {"output_type": dict},
            }
            
            # Apply overrides if present
            if hasattr(config_source, 'overrides') and config_source.overrides:
                config_dict.update(config_source.overrides)
            
            mock_config = AIAgentConfig(**config_dict)
            agent = _FakeAgent(mock_config)
            await agent.initialize()
            return agent

    import gearmeshing_ai.agent_core.runtime.engine as engine_mod

    monkeypatch.setattr(engine_mod, "get_agent_provider", lambda: _FakeProvider())

    runs = _Runs()
    events = _Events()
    reg = CapabilityRegistry()

    deps = EngineDeps(
        runs=runs,  # type: ignore[arg-type]
        events=events,  # type: ignore[arg-type]
        approvals=_NoopRepo(),  # type: ignore[arg-type]
        checkpoints=_NoopRepo(),
        tool_invocations=_NoopRepo(),
        capabilities=reg,
        usage=None,
        prompt_provider=_PromptProvider(prompt="ROLE PROMPT"),
        role_provider=DEFAULT_ROLE_PROVIDER,
        thought_model=object(),
        checkpointer=MemorySaver(),
    )

    engine = AgentEngine(policy=GlobalPolicy(PolicyConfig()), deps=deps)

    run = AgentRun(role="dev", objective="x")
    plan = [{"kind": "thought", "thought": "do", "args": {"a": 1}}]
    await engine.start_run(run=run, plan=plan)

    assert called["system_prompt"] == "ROLE PROMPT"
    assert "thought=do" in called["prompt"]
    assert any(e.type.value == "artifact.created" and "output" in e.payload for e in events.events)


@pytest.mark.asyncio
async def test_thought_step_prompt_keyerror_uses_fallback_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    called: Dict[str, Any] = {}

    from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

    class _FakeAgent:
        def __init__(self, config: Any) -> None:
            called["system_prompt"] = config.system_prompt
            called["output_type"] = config.metadata.get("output_type")
            called["model"] = config.model
            self._initialized = False

        async def initialize(self) -> None:
            self._initialized = True

        async def invoke(self, input_text: str, **kwargs: Any) -> AIAgentResponse:
            called["prompt"] = input_text
            return AIAgentResponse(content={"ok": True}, success=True)

        async def cleanup(self) -> None:
            pass

    class _FakeProvider:
        async def create_agent(self, config: Any, use_cache: bool = False) -> _FakeAgent:
            agent = _FakeAgent(config)
            await agent.initialize()
            return agent
        
        async def create_agent_from_config_source(self, config_source: Any, use_cache: bool = False) -> _FakeAgent:
            # Mock the config source to return an AIAgentConfig object
            from gearmeshing_ai.agent_core.abstraction import AIAgentConfig
            
            # Start with base config
            config_dict = {
                "name": "test-thought",
                "framework": "pydantic_ai",
                "model": "gpt-4o",
                "system_prompt": "You are a helpful assistant...",  # Default
                "temperature": 0.7,
                "max_tokens": 4096,
                "top_p": 0.9,
                "metadata": {"output_type": dict},
            }
            
            # Apply overrides if present
            if hasattr(config_source, 'overrides') and config_source.overrides:
                config_dict.update(config_source.overrides)
            
            mock_config = AIAgentConfig(**config_dict)
            agent = _FakeAgent(mock_config)
            await agent.initialize()
            return agent

    import gearmeshing_ai.agent_core.runtime.engine as engine_mod

    monkeypatch.setattr(engine_mod, "get_agent_provider", lambda: _FakeProvider())

    runs = _Runs()
    events = _Events()
    reg = CapabilityRegistry()

    class _PromptProvider:
        def get(self, name: str, locale: str = "en", tenant: str | None = None) -> str:
            called.setdefault("get_calls", []).append((name, tenant))
            if name == "custom/system":
                raise KeyError(name)
            return "FALLBACK PROMPT"

        def version(self) -> str:
            return "v"

        def refresh(self) -> None:
            return None

    class _RoleProvider:
        def get(self, _role: str):
            class _Cog:
                system_prompt_key = "custom/system"

            class _Role:
                cognitive = _Cog()

            return _Role()

    deps = EngineDeps(
        runs=runs,  # type: ignore[arg-type]
        events=events,  # type: ignore[arg-type]
        approvals=_NoopRepo(),  # type: ignore[arg-type]
        checkpoints=_NoopRepo(),
        tool_invocations=_NoopRepo(),
        capabilities=reg,
        usage=None,
        prompt_provider=_PromptProvider(),
        role_provider=_RoleProvider(),  # type: ignore[arg-type]
        thought_model=object(),
        checkpointer=MemorySaver(),
    )

    engine = AgentEngine(policy=GlobalPolicy(PolicyConfig()), deps=deps)

    run = AgentRun(role="dev", objective="x", tenant_id="t1")
    plan = [{"kind": "thought", "thought": "do", "args": {"a": 1}}]
    await engine.start_run(run=run, plan=plan)

    assert called["system_prompt"] == "FALLBACK PROMPT"
    assert ("custom/system", "t1") in called["get_calls"]
    assert ("dev/system", "t1") in called["get_calls"]


@pytest.mark.asyncio
async def test_thought_step_prompt_provider_exception_disables_agent_call(monkeypatch: pytest.MonkeyPatch) -> None:
    called: Dict[str, Any] = {}

    from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

    class _FakeAgent:
        def __init__(self, config: Any) -> None:
            called["constructed"] = True
            self._initialized = False

        async def initialize(self) -> None:
            self._initialized = True

        async def invoke(self, input_text: str, **kwargs: Any) -> AIAgentResponse:
            called["ran"] = True
            return AIAgentResponse(content={}, success=True)

        async def cleanup(self) -> None:
            pass

    class _FakeProvider:
        async def create_agent(self, config: Any, use_cache: bool = False) -> _FakeAgent:
            agent = _FakeAgent(config)
            await agent.initialize()
            return agent
        
        async def create_agent_from_config_source(self, config_source: Any, use_cache: bool = False) -> _FakeAgent:
            # Mock the config source to return an AIAgentConfig object
            from gearmeshing_ai.agent_core.abstraction import AIAgentConfig
            
            # Start with base config
            config_dict = {
                "name": "test-thought",
                "framework": "pydantic_ai",
                "model": "gpt-4o",
                "system_prompt": "You are a helpful assistant...",  # Default
                "temperature": 0.7,
                "max_tokens": 4096,
                "top_p": 0.9,
                "metadata": {"output_type": dict},
            }
            
            # Apply overrides if present
            if hasattr(config_source, 'overrides') and config_source.overrides:
                config_dict.update(config_source.overrides)
            
            mock_config = AIAgentConfig(**config_dict)
            agent = _FakeAgent(mock_config)
            await agent.initialize()
            return agent

    import gearmeshing_ai.agent_core.runtime.engine as engine_mod

    monkeypatch.setattr(engine_mod, "get_agent_provider", lambda: _FakeProvider())

    runs = _Runs()
    events = _Events()
    reg = CapabilityRegistry()

    class _PromptProvider:
        def get(self, name: str, locale: str = "en", tenant: str | None = None) -> str:  # noqa: ARG002
            raise RuntimeError("boom")

        def version(self) -> str:
            return "v"

        def refresh(self) -> None:
            return None

    deps = EngineDeps(
        runs=runs,  # type: ignore[arg-type]
        events=events,  # type: ignore[arg-type]
        approvals=_NoopRepo(),  # type: ignore[arg-type]
        checkpoints=_NoopRepo(),
        tool_invocations=_NoopRepo(),
        capabilities=reg,
        usage=None,
        prompt_provider=_PromptProvider(),
        role_provider=DEFAULT_ROLE_PROVIDER,
        thought_model=object(),
        checkpointer=MemorySaver(),
    )

    engine = AgentEngine(policy=GlobalPolicy(PolicyConfig()), deps=deps)

    run = AgentRun(role="dev", objective="x")
    plan = [{"kind": "thought", "thought": "do", "args": {"a": 1}}]
    await engine.start_run(run=run, plan=plan)

    assert called == {}
    artifact_events = [e for e in events.events if e.type.value == "artifact.created"]
    assert artifact_events
    assert "output" not in artifact_events[-1].payload


@pytest.mark.asyncio
async def test_thought_step_non_dict_agent_output_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

    class _FakeAgent:
        def __init__(self, config: Any) -> None:
            self._initialized = False

        async def initialize(self) -> None:
            self._initialized = True

        async def invoke(self, input_text: str, **kwargs: Any) -> AIAgentResponse:
            return AIAgentResponse(content="hello", success=True)

        async def cleanup(self) -> None:
            pass

    class _FakeProvider:
        async def create_agent(self, config: Any, use_cache: bool = False) -> _FakeAgent:
            agent = _FakeAgent(config)
            await agent.initialize()
            return agent
        
        async def create_agent_from_config_source(self, config_source: Any, use_cache: bool = False) -> _FakeAgent:
            # Mock the config source to return an AIAgentConfig object
            from gearmeshing_ai.agent_core.abstraction import AIAgentConfig
            
            # Start with base config
            config_dict = {
                "name": "test-thought",
                "framework": "pydantic_ai",
                "model": "gpt-4o",
                "system_prompt": "You are a helpful assistant...",  # Default
                "temperature": 0.7,
                "max_tokens": 4096,
                "top_p": 0.9,
                "metadata": {"output_type": dict},
            }
            
            # Apply overrides if present
            if hasattr(config_source, 'overrides') and config_source.overrides:
                config_dict.update(config_source.overrides)
            
            mock_config = AIAgentConfig(**config_dict)
            agent = _FakeAgent(mock_config)
            await agent.initialize()
            return agent

    import gearmeshing_ai.agent_core.runtime.engine as engine_mod

    monkeypatch.setattr(engine_mod, "get_agent_provider", lambda: _FakeProvider())

    runs = _Runs()
    events = _Events()
    reg = CapabilityRegistry()

    deps = EngineDeps(
        runs=runs,  # type: ignore[arg-type]
        events=events,  # type: ignore[arg-type]
        approvals=_NoopRepo(),  # type: ignore[arg-type]
        checkpoints=_NoopRepo(),
        tool_invocations=_NoopRepo(),
        capabilities=reg,
        usage=None,
        prompt_provider=_PromptProvider(prompt="ROLE PROMPT"),
        role_provider=DEFAULT_ROLE_PROVIDER,
        thought_model=object(),
        checkpointer=MemorySaver(),
    )

    engine = AgentEngine(policy=GlobalPolicy(PolicyConfig()), deps=deps)

    run = AgentRun(role="dev", objective="x")
    plan = [{"kind": "thought", "thought": "do", "args": {"a": 1}}]
    await engine.start_run(run=run, plan=plan)

    artifact_events = [e for e in events.events if e.type.value == "artifact.created"]
    assert artifact_events
    assert artifact_events[-1].payload.get("output") == {"result": "hello"}
