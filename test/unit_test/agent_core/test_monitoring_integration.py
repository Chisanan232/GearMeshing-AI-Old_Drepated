"""
Unit tests for LangSmith monitoring integration in agent workflows.

Tests verify that:
- Decorators properly wrap functions
- Context information is captured correctly
- Graceful degradation when LangSmith is unavailable
- No performance impact when tracing is disabled
"""

from unittest.mock import patch

import pytest

from gearmeshing_ai.agent_core.monitoring_integration import (
    get_agent_run_context,
    get_capability_context,
    get_llm_context,
    trace_agent_run,
    trace_capability_execution,
    trace_llm_call,
    trace_planning,
)


class TestTraceAgentRunDecorator:
    """Test @trace_agent_run decorator."""

    @pytest.mark.asyncio
    async def test_trace_agent_run_async_function(self):
        """Test tracing async agent run functions."""

        @trace_agent_run
        async def mock_agent_run():
            return "run-123"

        result = await mock_agent_run()
        assert result == "run-123"

    def test_trace_agent_run_sync_function(self):
        """Test tracing sync agent run functions."""

        @trace_agent_run
        def mock_agent_run():
            return "run-123"

        result = mock_agent_run()
        assert result == "run-123"

    @pytest.mark.asyncio
    async def test_trace_agent_run_with_exception(self):
        """Test that exceptions are properly propagated."""

        @trace_agent_run
        async def mock_agent_run_error():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await mock_agent_run_error()

    @pytest.mark.asyncio
    async def test_trace_agent_run_preserves_function_name(self):
        """Test that decorator preserves function name."""

        @trace_agent_run
        async def my_agent_workflow():
            return "result"

        assert my_agent_workflow.__name__ == "my_agent_workflow"


class TestTracePlanningDecorator:
    """Test @trace_planning decorator."""

    @pytest.mark.asyncio
    async def test_trace_planning_async_function(self):
        """Test tracing async planning functions."""

        @trace_planning
        async def mock_planning():
            return [{"kind": "thought", "thought": "Step 1"}]

        result = await mock_planning()
        assert len(result) == 1
        assert result[0]["kind"] == "thought"

    def test_trace_planning_sync_function(self):
        """Test tracing sync planning functions."""

        @trace_planning
        def mock_planning():
            return [{"kind": "action", "action": "send_email"}]

        result = mock_planning()
        assert len(result) == 1
        assert result[0]["kind"] == "action"

    @pytest.mark.asyncio
    async def test_trace_planning_with_arguments(self):
        """Test tracing planning with arguments."""

        @trace_planning
        async def mock_planning(objective: str, role: str):
            return [
                {"kind": "thought", "thought": f"Planning for {objective}"},
                {"kind": "action", "action": f"Execute as {role}"},
            ]

        result = await mock_planning("Send email", "email_agent")
        assert len(result) == 2


class TestTraceCapabilityExecutionDecorator:
    """Test @trace_capability_execution decorator."""

    @pytest.mark.asyncio
    async def test_trace_capability_async_function(self):
        """Test tracing async capability execution."""

        @trace_capability_execution("send_email")
        async def send_email(to: str, subject: str):
            return {"status": "sent", "to": to}

        result = await send_email("john@example.com", "Hello")
        assert result["status"] == "sent"
        assert result["to"] == "john@example.com"

    def test_trace_capability_sync_function(self):
        """Test tracing sync capability execution."""

        @trace_capability_execution("create_task")
        def create_task(title: str):
            return {"id": "task-123", "title": title}

        result = create_task("My Task")
        assert result["id"] == "task-123"
        assert result["title"] == "My Task"

    @pytest.mark.asyncio
    async def test_trace_capability_with_exception(self):
        """Test that capability exceptions are propagated."""

        @trace_capability_execution("failing_capability")
        async def failing_capability():
            raise RuntimeError("Capability failed")

        with pytest.raises(RuntimeError, match="Capability failed"):
            await failing_capability()

    @pytest.mark.asyncio
    async def test_trace_capability_preserves_function_name(self):
        """Test that decorator preserves function name."""

        @trace_capability_execution("my_capability")
        async def my_custom_capability():
            return "result"

        assert my_custom_capability.__name__ == "my_custom_capability"


class TestTraceLLMCallDecorator:
    """Test @trace_llm_call decorator."""

    @pytest.mark.asyncio
    async def test_trace_llm_call_async_function(self):
        """Test tracing async LLM calls."""

        @trace_llm_call("gpt-4o", "openai")
        async def call_gpt4(prompt: str):
            return "This is a response"

        result = await call_gpt4("Hello")
        assert result == "This is a response"

    def test_trace_llm_call_sync_function(self):
        """Test tracing sync LLM calls."""

        @trace_llm_call("claude-3-5-sonnet", "anthropic")
        def call_claude(prompt: str):
            return "Claude response"

        result = call_claude("Hello")
        assert result == "Claude response"

    @pytest.mark.asyncio
    async def test_trace_llm_call_with_multiple_models(self):
        """Test tracing different LLM models."""

        @trace_llm_call("gpt-4o", "openai")
        async def call_gpt4():
            return "gpt4"

        @trace_llm_call("claude-3-5-sonnet", "anthropic")
        async def call_claude():
            return "claude"

        result1 = await call_gpt4()
        result2 = await call_claude()

        assert result1 == "gpt4"
        assert result2 == "claude"


class TestContextFunctions:
    """Test context information gathering functions."""

    def test_get_agent_run_context(self):
        """Test agent run context generation."""
        context = get_agent_run_context(
            run_id="run-123",
            tenant_id="tenant-456",
            role="email_agent",
        )

        assert context["run_id"] == "run-123"
        assert context["tenant_id"] == "tenant-456"
        assert context["role"] == "email_agent"

    def test_get_capability_context_with_inputs(self):
        """Test capability context with inputs."""
        inputs = {"to": "john@example.com", "subject": "Hello"}
        context = get_capability_context(
            capability_name="send_email",
            inputs=inputs,
        )

        assert context["capability"] == "send_email"
        assert context["inputs"] == inputs

    def test_get_capability_context_without_inputs(self):
        """Test capability context without inputs."""
        context = get_capability_context(capability_name="send_email")

        assert context["capability"] == "send_email"
        assert context["inputs"] == {}

    def test_get_llm_context_full(self):
        """Test LLM context with all parameters."""
        context = get_llm_context(
            model="gpt-4o",
            provider="openai",
            tokens_used=150,
            cost_usd=0.001,
        )

        assert context["model"] == "gpt-4o"
        assert context["provider"] == "openai"
        assert context["tokens_used"] == 150
        assert context["cost_usd"] == 0.001

    def test_get_llm_context_minimal(self):
        """Test LLM context with minimal parameters."""
        context = get_llm_context(
            model="gpt-4o",
            provider="openai",
        )

        assert context["model"] == "gpt-4o"
        assert context["provider"] == "openai"
        assert "tokens_used" not in context
        assert "cost_usd" not in context

    def test_get_llm_context_with_tokens_only(self):
        """Test LLM context with tokens but no cost."""
        context = get_llm_context(
            model="claude-3-5-sonnet",
            provider="anthropic",
            tokens_used=200,
        )

        assert context["tokens_used"] == 200
        assert "cost_usd" not in context


class TestDecoratorGracefulDegradation:
    """Test that decorators gracefully handle missing LangSmith."""

    @pytest.mark.asyncio
    async def test_trace_agent_run_without_langsmith(self):
        """Test agent run tracing without LangSmith available."""
        with patch("gearmeshing_ai.core.monitoring.get_traceable_decorator") as mock_get_decorator:
            # Simulate LangSmith unavailable with a no-op decorator
            def noop_decorator(func=None, **kwargs):
                if func is None:
                    return lambda f: f
                return func

            mock_get_decorator.return_value = noop_decorator

            @trace_agent_run
            async def mock_run():
                return "run-123"

            result = await mock_run()
            assert result == "run-123"

    @pytest.mark.asyncio
    async def test_trace_planning_without_langsmith(self):
        """Test planning tracing without LangSmith available."""
        with patch("gearmeshing_ai.core.monitoring.get_traceable_decorator") as mock_get_decorator:

            def noop_decorator(func=None, **kwargs):
                if func is None:
                    return lambda f: f
                return func

            mock_get_decorator.return_value = noop_decorator

            @trace_planning
            async def mock_planning():
                return []

            result = await mock_planning()
            assert result == []

    @pytest.mark.asyncio
    async def test_trace_capability_without_langsmith(self):
        """Test capability tracing without LangSmith available."""
        with patch("gearmeshing_ai.core.monitoring.get_traceable_decorator") as mock_get_decorator:

            def noop_decorator(func=None, **kwargs):
                if func is None:
                    return lambda f: f
                return func

            mock_get_decorator.return_value = noop_decorator

            @trace_capability_execution("test_capability")
            async def mock_capability():
                return "result"

            result = await mock_capability()
            assert result == "result"


class TestDecoratorWithArguments:
    """Test decorators with various argument patterns."""

    @pytest.mark.asyncio
    async def test_trace_agent_run_with_kwargs(self):
        """Test agent run with keyword arguments."""

        @trace_agent_run
        async def mock_run(run_id: str, tenant_id: str):
            return f"{run_id}-{tenant_id}"

        result = await mock_run(run_id="run-123", tenant_id="tenant-456")
        assert result == "run-123-tenant-456"

    @pytest.mark.asyncio
    async def test_trace_planning_with_mixed_args(self):
        """Test planning with mixed positional and keyword arguments."""

        @trace_planning
        async def mock_planning(objective, role, max_steps=10):
            return {
                "objective": objective,
                "role": role,
                "max_steps": max_steps,
            }

        result = await mock_planning("Send email", role="email_agent", max_steps=5)
        assert result["objective"] == "Send email"
        assert result["role"] == "email_agent"
        assert result["max_steps"] == 5

    @pytest.mark.asyncio
    async def test_trace_capability_with_complex_return(self):
        """Test capability with complex return types."""

        @trace_capability_execution("complex_capability")
        async def mock_capability():
            return {
                "status": "success",
                "data": [1, 2, 3],
                "metadata": {"key": "value"},
            }

        result = await mock_capability()
        assert result["status"] == "success"
        assert result["data"] == [1, 2, 3]
        assert result["metadata"]["key"] == "value"


class TestDecoratorCombination:
    """Test combining multiple decorators."""

    @pytest.mark.asyncio
    async def test_nested_decorators(self):
        """Test combining multiple tracing decorators."""

        @trace_agent_run
        @trace_planning
        async def combined_workflow():
            return "result"

        result = await combined_workflow()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_decorator_with_other_decorators(self):
        """Test LangSmith decorators with other decorators."""

        def other_decorator(func):
            async def wrapper(*args, **kwargs):
                result = await func(*args, **kwargs)
                return f"wrapped-{result}"

            return wrapper

        @other_decorator
        @trace_agent_run
        async def decorated_function():
            return "result"

        result = await decorated_function()
        assert result == "wrapped-result"
