"""
LangSmith monitoring integration utilities for agent workflows.

This module provides convenient utilities for integrating LangSmith tracing
into agent core workflows, including decorators and context managers for
tracing agent execution, planning, and capability execution.

Usage:
    from gearmeshing_ai.agent_core.monitoring_integration import trace_agent_run

    @trace_agent_run
    async def my_workflow(run: AgentRun) -> str:
        # Your workflow code here
        return run_id
"""

import logging
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def trace_agent_run(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to trace agent run execution with LangSmith.

    Automatically captures:
    - Agent run initialization
    - Planning phase
    - Execution phase
    - Errors and exceptions

    Usage:
        @trace_agent_run
        async def execute_agent_run(run: AgentRun) -> str:
            # Your implementation
            return run_id

    Args:
        func: The async function to trace

    Returns:
        Wrapped function with LangSmith tracing enabled
    """

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        from gearmeshing_ai.core.monitoring import get_traceable_decorator

        traceable = get_traceable_decorator()

        @traceable(name="agent_run", tags=["agent", "execution"])
        async def _traced_execution():
            return await func(*args, **kwargs)

        return await _traced_execution()

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        from gearmeshing_ai.core.monitoring import get_traceable_decorator

        traceable = get_traceable_decorator()

        @traceable(name="agent_run", tags=["agent", "execution"])
        def _traced_execution():
            return func(*args, **kwargs)

        return _traced_execution()

    # Return async wrapper if function is async, otherwise sync wrapper
    if hasattr(func, "__await__"):
        return async_wrapper
    return sync_wrapper


def trace_planning(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to trace planning phase execution with LangSmith.

    Captures:
    - Planning algorithm execution
    - Plan generation
    - Plan validation

    Usage:
        @trace_planning
        async def plan_agent_steps(objective: str, role: str) -> list:
            # Your planning implementation
            return plan

    Args:
        func: The async function to trace

    Returns:
        Wrapped function with LangSmith tracing enabled
    """

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        from gearmeshing_ai.core.monitoring import get_traceable_decorator

        traceable = get_traceable_decorator()

        @traceable(name="agent_planning", tags=["planning", "agent"])
        async def _traced_planning():
            return await func(*args, **kwargs)

        return await _traced_planning()

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        from gearmeshing_ai.core.monitoring import get_traceable_decorator

        traceable = get_traceable_decorator()

        @traceable(name="agent_planning", tags=["planning", "agent"])
        def _traced_planning():
            return func(*args, **kwargs)

        return _traced_planning()

    if hasattr(func, "__await__"):
        return async_wrapper
    return sync_wrapper


def trace_capability_execution(capability_name: str):
    """
    Decorator to trace capability execution with LangSmith.

    Captures:
    - Capability invocation
    - Input parameters
    - Output results
    - Execution time

    Usage:
        @trace_capability_execution("send_email")
        async def send_email_capability(to: str, subject: str, body: str):
            # Your capability implementation
            return result

    Args:
        capability_name: Name of the capability being executed

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            from gearmeshing_ai.core.monitoring import get_traceable_decorator

            traceable = get_traceable_decorator()

            @traceable(
                name=f"capability_{capability_name}",
                tags=["capability", capability_name],
            )
            async def _traced_capability():
                return await func(*args, **kwargs)

            return await _traced_capability()

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            from gearmeshing_ai.core.monitoring import get_traceable_decorator

            traceable = get_traceable_decorator()

            @traceable(
                name=f"capability_{capability_name}",
                tags=["capability", capability_name],
            )
            def _traced_capability():
                return func(*args, **kwargs)

            return _traced_capability()

        if hasattr(func, "__await__"):
            return async_wrapper
        return sync_wrapper

    return decorator


def trace_llm_call(model_name: str, provider: str):
    """
    Decorator to trace LLM calls with LangSmith.

    Captures:
    - Model name and provider
    - Prompt and completion
    - Token usage
    - Latency

    Usage:
        @trace_llm_call("gpt-4o", "openai")
        async def call_llm(prompt: str) -> str:
            # Your LLM call implementation
            return response

    Args:
        model_name: Name of the LLM model
        provider: Provider name (openai, anthropic, google)

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            from gearmeshing_ai.core.monitoring import get_traceable_decorator

            traceable = get_traceable_decorator()

            @traceable(
                name=f"llm_call_{model_name}",
                tags=["llm", provider, model_name],
            )
            async def _traced_llm_call():
                return await func(*args, **kwargs)

            return await _traced_llm_call()

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            from gearmeshing_ai.core.monitoring import get_traceable_decorator

            traceable = get_traceable_decorator()

            @traceable(
                name=f"llm_call_{model_name}",
                tags=["llm", provider, model_name],
            )
            def _traced_llm_call():
                return func(*args, **kwargs)

            return _traced_llm_call()

        if hasattr(func, "__await__"):
            return async_wrapper
        return sync_wrapper

    return decorator


def get_agent_run_context(run_id: str, tenant_id: str, role: str) -> dict:
    """
    Get context information for an agent run to include in LangSmith traces.

    Args:
        run_id: The unique run identifier
        tenant_id: The tenant identifier
        role: The agent role

    Returns:
        Dictionary with context information for tracing
    """
    return {
        "run_id": run_id,
        "tenant_id": tenant_id,
        "role": role,
    }


def get_capability_context(
    capability_name: str, inputs: Optional[dict] = None
) -> dict:
    """
    Get context information for a capability execution to include in traces.

    Args:
        capability_name: Name of the capability
        inputs: Input parameters to the capability

    Returns:
        Dictionary with context information for tracing
    """
    return {
        "capability": capability_name,
        "inputs": inputs or {},
    }


def get_llm_context(
    model: str,
    provider: str,
    tokens_used: Optional[int] = None,
    cost_usd: Optional[float] = None,
) -> dict:
    """
    Get context information for an LLM call to include in traces.

    Args:
        model: Model name
        provider: Provider name
        tokens_used: Total tokens used
        cost_usd: Cost in USD

    Returns:
        Dictionary with context information for tracing
    """
    context = {
        "model": model,
        "provider": provider,
    }
    if tokens_used is not None:
        context["tokens_used"] = tokens_used
    if cost_usd is not None:
        context["cost_usd"] = cost_usd
    return context
