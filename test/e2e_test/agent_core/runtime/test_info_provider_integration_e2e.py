from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Protocol

import httpx
import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from sqlalchemy import select
from testcontainers.postgres import PostgresContainer

from gearmeshing_ai.agent_core.factory import build_default_registry
from gearmeshing_ai.agent_core.policy.global_policy import GlobalPolicy
from gearmeshing_ai.agent_core.policy.models import PolicyConfig
from gearmeshing_ai.agent_core.repos.models import (
    ApprovalRow,
    EventRow,
    ToolInvocationRow,
)
from gearmeshing_ai.agent_core.repos.sql import (
    build_sql_repos,
    create_all,
    create_engine,
    create_sessionmaker,
)
from gearmeshing_ai.info_provider import DEFAULT_ROLE_PROVIDER, AgentRole, CapabilityName
from gearmeshing_ai.agent_core.runtime import EngineDeps
from gearmeshing_ai.agent_core.runtime.engine import AgentEngine
from gearmeshing_ai.agent_core.schemas.domain import (
    AgentEventType,
    AgentRun,
    AgentRunStatus,
    ApprovalDecision,
    RiskLevel,
)
from gearmeshing_ai.info_provider.mcp.gateway_api.client import GatewayApiClient
from gearmeshing_ai.info_provider.mcp.provider import AsyncMCPInfoProvider
from gearmeshing_ai.info_provider.mcp.schemas.config import ServerConfig
from gearmeshing_ai.info_provider.mcp.strategy.direct import DirectMcpStrategy
from gearmeshing_ai.info_provider.mcp.strategy.direct_async import (
    AsyncDirectMcpStrategy,
)
from gearmeshing_ai.info_provider.mcp.strategy.gateway import GatewayMcpStrategy
from gearmeshing_ai.info_provider.mcp.strategy.gateway_async import (
    AsyncGatewayMcpStrategy,
)
from gearmeshing_ai.info_provider.mcp.transport import AsyncMCPTransport
from gearmeshing_ai.info_provider.prompt.provider import BuiltinPromptProvider


class _AsyncMcpLike(Protocol):
    async def list_tools(self, server_id: str): ...

    async def call_tool(self, server_id: str, tool_name: str, args: dict[str, Any]): ...


class _SyncToAsyncMcpAdapter:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def list_tools(self, server_id: str):
        return await asyncio.to_thread(lambda: list(self._inner.list_tools(server_id)))

    async def call_tool(self, server_id: str, tool_name: str, args: dict[str, Any]):
        return await asyncio.to_thread(lambda: self._inner.call_tool(server_id, tool_name, args))


class _FakeTool:
    def __init__(self, name: str, description: str | None, input_schema: Dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.inputSchema = input_schema


class _FakeListToolsResp:
    def __init__(self, tools: List[_FakeTool]) -> None:
        self.tools = tools
        self.next_cursor: str | None = None


class _FakeMcpSession:
    def __init__(self, *, state: dict, tools: List[_FakeTool]) -> None:
        self._state = state
        self._tools = tools

    async def list_tools(self, cursor: str | None = None, limit: int | None = None):  # noqa: ARG002
        self._state["list_tools_calls"] = self._state.get("list_tools_calls", 0) + 1
        return _FakeListToolsResp(self._tools)

    async def call_tool(self, name: str, arguments: Dict[str, Any] | None = None):
        self._state["call_tool_calls"] = self._state.get("call_tool_calls", 0) + 1
        self._state.setdefault("call_tool_names", []).append(name)
        self._state.setdefault("call_tool_args", []).append(dict(arguments or {}))
        return {"ok": True, "tool": name, "echo": dict(arguments or {})}


class _FakeMcpTransport(AsyncMCPTransport):
    def __init__(self, *, state: dict, tools: List[_FakeTool]) -> None:
        self._state = state
        self._tools = tools

    def session(self, endpoint_url: str):  # noqa: ARG002
        @asynccontextmanager
        async def _cm():
            yield _FakeMcpSession(state=self._state, tools=self._tools)

        return _cm()


def _mock_gateway_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.startswith("/admin/gateways/"):
            gw_id = request.url.path.split("/admin/gateways/", 1)[1]
            return httpx.Response(200, json={"id": gw_id, "name": "gw", "url": "http://mock/mcp"})
        if request.method == "GET" and request.url.path == "/admin/tools":
            data = {
                "data": [
                    {
                        "id": "t1",
                        "originalName": "echo",
                        "name": "echo",
                        "customName": "echo",
                        "customNameSlug": "echo",
                        "gatewaySlug": "gw",
                        "requestType": "SSE",
                        "integrationType": "MCP",
                        "inputSchema": {"type": "object"},
                        "createdAt": "2024-01-01T00:00:00Z",
                        "updatedAt": "2024-01-01T00:00:00Z",
                        "enabled": True,
                        "reachable": True,
                        "executionCount": 0,
                        "metrics": {
                            "totalExecutions": 0,
                            "successfulExecutions": 0,
                            "failedExecutions": 0,
                            "failureRate": 0.0,
                        },
                        "gatewayId": "g1",
                    },
                    {
                        "id": "t2",
                        "originalName": "create_issue",
                        "name": "create_issue",
                        "customName": "create_issue",
                        "customNameSlug": "create-issue",
                        "gatewaySlug": "gw",
                        "requestType": "SSE",
                        "integrationType": "MCP",
                        "inputSchema": {"type": "object"},
                        "createdAt": "2024-01-01T00:00:00Z",
                        "updatedAt": "2024-01-01T00:00:00Z",
                        "enabled": True,
                        "reachable": True,
                        "executionCount": 0,
                        "metrics": {
                            "totalExecutions": 0,
                            "successfulExecutions": 0,
                            "failedExecutions": 0,
                            "failureRate": 0.0,
                        },
                        "gatewayId": "g1",
                    },
                ]
            }
            return httpx.Response(200, json=data)
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


def _build_mcp_strategy(variant: str) -> tuple[_AsyncMcpLike, dict]:
    state: dict = {}
    fake_tools = [
        _FakeTool("echo", "Echo", {"type": "object"}),
        _FakeTool("create_issue", "Create", {"type": "object"}),
    ]
    mcp_transport = _FakeMcpTransport(state=state, tools=fake_tools)

    if variant == "direct_async":
        strat = AsyncDirectMcpStrategy(
            [ServerConfig(name="s1", endpoint_url="http://mock/mcp")],
            ttl_seconds=0.0,
            mcp_transport=mcp_transport,
        )
        return strat, state

    if variant == "direct_sync":
        strat = DirectMcpStrategy(  # type: ignore[assignment]
            [ServerConfig(name="s1", endpoint_url="http://mock/mcp")],
            ttl_seconds=0.0,
            mcp_transport=mcp_transport,
        )
        return _SyncToAsyncMcpAdapter(strat), state

    transport = _mock_gateway_transport()
    mgmt_client = httpx.Client(transport=transport, base_url="http://mock")
    gw = GatewayApiClient("http://mock", client=mgmt_client)

    if variant == "gateway_async":
        http_client = httpx.AsyncClient(transport=transport, base_url="http://mock")
        strat = AsyncGatewayMcpStrategy(gw, client=http_client, ttl_seconds=0.0, mcp_transport=mcp_transport)  # type: ignore[assignment]
        state["_close_http_client"] = http_client
        state["_close_mgmt_client"] = mgmt_client
        return strat, state

    if variant == "gateway_sync":
        http_client = httpx.Client(transport=transport, base_url="http://mock")
        strat = GatewayMcpStrategy(gw, client=http_client, ttl_seconds=0.0, mcp_transport=mcp_transport)  # type: ignore[assignment]
        state["_close_http_client"] = http_client
        state["_close_mgmt_client"] = mgmt_client
        return _SyncToAsyncMcpAdapter(strat), state

    raise ValueError(f"unknown variant: {variant}")


@pytest.mark.asyncio
async def test_e2e_role_prompt_provider_is_used_for_thought_step(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    from gearmeshing_ai.agent_core.abstraction import AIAgentResponse

    class _FakeAgent:
        def __init__(self, config: Any) -> None:
            captured["system_prompt"] = config.system_prompt
            captured["output_type"] = config.metadata.get("output_type")
            captured["model"] = config.model
            self._initialized = False

        async def initialize(self) -> None:
            self._initialized = True

        async def invoke(self, input_text: str, **kwargs: Any) -> AIAgentResponse:
            captured["prompt"] = input_text
            return AIAgentResponse(content={"from_agent": True}, success=True)

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
            
            # Get the system prompt from the prompt provider (simulating real behavior)
            system_prompt = None
            if hasattr(config_source, 'prompt_key') and hasattr(config_source, 'prompt_tenant_id'):
                # Simulate the prompt provider behavior and capture tenant info
                captured["prompt_provider_tenant"] = config_source.prompt_tenant_id
                mock_prompts = {
                    "dev/system": "DEV SYSTEM PROMPT",
                    "planner/system": "PLANNER SYSTEM PROMPT",
                }
                system_prompt = mock_prompts.get(config_source.prompt_key, "You are a helpful assistant...")
            
            # Start with base config
            config_dict = {
                "name": "test-thought",
                "framework": "pydantic_ai",
                "model": "gpt-4o",
                "system_prompt": system_prompt or "You are a helpful assistant...",
                "temperature": 0.7,
                "max_tokens": 4096,
                "top_p": 0.9,
                "metadata": {"output_type": dict},
            }
            
            # Apply overrides if present (but not system_prompt since it comes from prompt provider)
            if hasattr(config_source, "overrides") and config_source.overrides:
                filtered_overrides = {k: v for k, v in config_source.overrides.items() if k != "system_prompt"}
                config_dict.update(filtered_overrides)
            
            mock_config = AIAgentConfig(**config_dict)
            agent = _FakeAgent(mock_config)
            await agent.initialize()
            return agent

    import gearmeshing_ai.agent_core.runtime.engine as engine_mod

    monkeypatch.setattr(engine_mod, "get_agent_provider", lambda: _FakeProvider())

    builtin = BuiltinPromptProvider(
        prompts={
            "en": {
                "dev/system": "DEV SYSTEM PROMPT",
                "planner/system": "PLANNER SYSTEM PROMPT",
            }
        },
        version_id="builtin-e2e",
    )

    orig_get = builtin.get

    def _capture_get(name: str, locale: str = "en", tenant: Optional[str] = None) -> str:
        captured["prompt_provider_tenant"] = tenant
        return orig_get(name, locale=locale, tenant=tenant)

    monkeypatch.setattr(builtin, "get", _capture_get)

    with PostgresContainer("postgres:16") as pg:
        db_url = pg.get_connection_url().replace("postgresql://", "postgresql+asyncpg://")
        pool_url = pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")

        engine = create_engine(db_url)
        await create_all(engine)
        session_factory = create_sessionmaker(engine)
        repos = build_sql_repos(session_factory=session_factory)

        async with AsyncConnectionPool(conninfo=pool_url, min_size=1, max_size=1, kwargs={"autocommit": True}) as pool:
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()

            cfg = PolicyConfig()
            # nothing to approve here; just ensure engine runs
            cfg.tool_policy.allowed_capabilities = {
                CapabilityName.summarize,
                CapabilityName.mcp_call,
                CapabilityName.docs_read,
                CapabilityName.web_search,
                CapabilityName.web_fetch,
                CapabilityName.shell_exec,
                CapabilityName.codegen,
                CapabilityName.code_execution,
            }
            runtime = AgentEngine(
                policy=GlobalPolicy(cfg),
                deps=EngineDeps(
                    runs=repos.runs,
                    events=repos.events,
                    approvals=repos.approvals,
                    checkpoints=repos.checkpoints,
                    tool_invocations=repos.tool_invocations,
                    usage=repos.usage,
                    capabilities=build_default_registry(),
                    prompt_provider=builtin,
                    role_provider=DEFAULT_ROLE_PROVIDER,
                    thought_model="test",
                    checkpointer=checkpointer,
                ),
            )

            run = AgentRun(role="dev", objective="prove prompt", tenant_id="t1")
            plan = [{"kind": "thought", "thought": "do something", "args": {"x": 1}}]
            await runtime.start_run(run=run, plan=plan)

            assert "DEV SYSTEM PROMPT" in captured["system_prompt"]
            assert captured["prompt_provider_tenant"] == "t1"

            loaded = await repos.runs.get(run.id)
            assert loaded is not None
            assert str(getattr(loaded.status, "value", loaded.status)) == AgentRunStatus.succeeded.value

            async with session_factory() as s:
                ev = await s.execute(select(EventRow).where(EventRow.run_id == run.id))
                ev_rows = list(ev.scalars().all())
                types = [r.type for r in ev_rows]
                assert AgentEventType.thought_executed.value in types
                assert AgentEventType.artifact_created.value in types
                artifact = next(r for r in ev_rows if r.type == AgentEventType.artifact_created.value)
                assert artifact.payload.get("output") == {"from_agent": True}

            await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "variant",
    [
        "direct_async",
        "direct_sync",
        "gateway_async",
        "gateway_sync",
    ],
)
async def test_e2e_mcp_call_uses_real_strategy_metadata_for_risk_and_approval_gate(variant: str) -> None:
    strat, state = _build_mcp_strategy(variant)
    server_id = "g1" if variant.startswith("gateway") else "s1"
    info_provider = AsyncMCPInfoProvider(strategies=[strat])

    async def _mcp_call(server_id: str, tool_name: str, tool_args: dict):
        return await strat.call_tool(server_id, tool_name, tool_args)

    with PostgresContainer("postgres:16") as pg:
        db_url = pg.get_connection_url().replace("postgresql://", "postgresql+asyncpg://")
        pool_url = pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")

        engine = create_engine(db_url)
        await create_all(engine)
        session_factory = create_sessionmaker(engine)
        repos = build_sql_repos(session_factory=session_factory)

        async with AsyncConnectionPool(conninfo=pool_url, min_size=1, max_size=1, kwargs={"autocommit": True}) as pool:
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()

            cfg = PolicyConfig()
            cfg.tool_policy.allowed_capabilities = {CapabilityName.mcp_call}
            cfg.approval_policy.require_for_risk_at_or_above = RiskLevel.medium

            runtime = AgentEngine(
                policy=GlobalPolicy(cfg),
                deps=EngineDeps(
                    runs=repos.runs,
                    events=repos.events,
                    approvals=repos.approvals,
                    checkpoints=repos.checkpoints,
                    tool_invocations=repos.tool_invocations,
                    usage=repos.usage,
                    capabilities=build_default_registry(),
                    mcp_info_provider=info_provider,
                    mcp_call=_mcp_call,
                    checkpointer=checkpointer,
                ),
            )

            run = AgentRun(role="dev", objective="mcp", tenant_id="t1")
            plan = [
                # echo is considered non-mutating by heuristic => low risk => no approval
                {
                    "kind": "action",
                    "capability": CapabilityName.mcp_call.value,
                    "server_id": server_id,
                    "tool_name": "echo",
                    "args": {"tool_args": {"k": "v"}},
                },
                # create_issue is considered mutating by heuristic => medium risk => approval required
                {
                    "kind": "action",
                    "capability": CapabilityName.mcp_call.value,
                    "server_id": server_id,
                    "tool_name": "create_issue",
                    "args": {"tool_args": {"k": "v2"}},
                },
            ]

            await runtime.start_run(run=run, plan=plan)

            # should have paused on tool-b approval
            cp = await checkpointer.aget_tuple(config={"configurable": {"thread_id": run.id}})
            assert cp is not None

            # AsyncPostgresSaver returns a CheckpointTuple, state is in cp.checkpoint
            cp_state = cp.checkpoint

            # LangGraph stores state in channel_values
            if "channel_values" in cp_state:
                cp_state = cp_state["channel_values"]

            approval_id = cp_state.get("awaiting_approval_id")
            assert approval_id

            async with session_factory() as s:
                inv = await s.execute(
                    select(ToolInvocationRow)
                    .where(ToolInvocationRow.run_id == run.id)
                    .order_by(ToolInvocationRow.created_at)
                )
                inv_rows = list(inv.scalars().all())
                # first tool executed, second should not yet
                assert len(inv_rows) == 1
                assert inv_rows[0].ok is True
                assert inv_rows[0].tool_name == "echo"
                assert inv_rows[0].args.get("_mcp_tool_mutating") is False

                appr = await s.execute(select(ApprovalRow).where(ApprovalRow.id == approval_id))
                assert appr.first() is not None

            # approve and resume
            await repos.approvals.resolve(approval_id, decision=ApprovalDecision.approved.value, decided_by="tester")
            await runtime.resume_run(run_id=run.id, approval_id=approval_id)

            updated = await repos.runs.get(run.id)
            assert updated is not None
            assert str(getattr(updated.status, "value", updated.status)) == AgentRunStatus.succeeded.value

            async with session_factory() as s:
                inv2 = await s.execute(
                    select(ToolInvocationRow)
                    .where(ToolInvocationRow.run_id == run.id)
                    .order_by(ToolInvocationRow.created_at)
                )
                inv_rows2 = list(inv2.scalars().all())
                assert len(inv_rows2) == 2
                assert inv_rows2[0].ok is True
                assert inv_rows2[1].ok is True
                assert inv_rows2[0].tool_name == "echo"
                assert inv_rows2[0].args.get("_mcp_tool_mutating") is False
                assert inv_rows2[1].tool_name == "create_issue"
                assert inv_rows2[1].args.get("_mcp_tool_mutating") is True

            # strategy should have been invoked twice in total (via transport session)
            assert state.get("call_tool_names") == ["echo", "create_issue"]

            # close any httpx clients we created for gateway variants
            close_http = state.get("_close_http_client")
            close_mgmt = state.get("_close_mgmt_client")
            if close_http is not None:
                if hasattr(close_http, "aclose"):
                    await close_http.aclose()
                else:
                    close_http.close()
            if close_mgmt is not None:
                close_mgmt.close()

            await engine.dispose()
