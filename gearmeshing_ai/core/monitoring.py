"""
Monitoring and Tracing Configuration Module.

This module provides integration with Pydantic AI Logfire and LangSmith for comprehensive
monitoring and tracing of AI application operations, including:
- Agent execution traces (LangGraph/LangSmith)
- LLM model calls and performance metrics
- API endpoint tracing
- Database operation monitoring
- Error tracking and logging
- Performance metrics collection

Features:
- Automatic instrumentation of Pydantic AI operations via Logfire
- LangSmith integration for LangGraph agent tracing
- Structured logging with context
- Performance metrics and latency tracking
- Error and exception tracking
- Request/response tracing for API calls
"""

import logging
import os
from typing import Optional

from fastapi import FastAPI

from gearmeshing_ai.server.core.config import settings

logger = logging.getLogger(__name__)


def initialize_logfire(app: FastAPI | None = None) -> None:
    """
    Initialize Pydantic AI Logfire for monitoring and tracing.

    This function sets up Logfire with automatic instrumentation for:
    - Pydantic AI model calls
    - SQLAlchemy database operations
    - HTTPX HTTP requests
    - FastAPI endpoints

    Args:
        app: FastAPI application instance for FastAPI instrumentation (optional).
             If provided, enables automatic tracing of FastAPI endpoints.

    The initialization is conditional based on settings.logfire.enabled.
    """
    if not settings.logfire.enabled:
        logger.info("Logfire monitoring is disabled. Set LOGFIRE__ENABLED=true to enable.")
        return

    if not settings.logfire.token:
        logger.warning(
            "Logfire is enabled but LOGFIRE__TOKEN is not set. "
            "Monitoring will not work. Set LOGFIRE__TOKEN to enable Logfire."
        )
        return

    try:
        import logfire

        # Configure Logfire with project settings
        logfire.configure(
            token=settings.logfire.token,
            service_name=settings.logfire.service_name,
            service_version=settings.logfire.service_version,
            environment=settings.logfire.environment,
        )

        # Instrument Pydantic AI
        if settings.logfire.trace_pydantic_ai:
            try:
                logfire.instrument_pydantic_ai()
                logger.info("Logfire: Pydantic AI instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument Pydantic AI: {e}")

        # Instrument SQLAlchemy
        if settings.logfire.trace_sqlalchemy:
            try:
                logfire.instrument_sqlalchemy()
                logger.info("Logfire: SQLAlchemy instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument SQLAlchemy: {e}")

        # Instrument HTTPX
        if settings.logfire.trace_httpx:
            try:
                logfire.instrument_httpx()
                logger.info("Logfire: HTTPX instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument HTTPX: {e}")

        # Instrument FastAPI
        if settings.logfire.trace_fastapi:
            try:
                if app is not None:
                    logfire.instrument_fastapi(app=app)
                    logger.info("Logfire: FastAPI instrumentation enabled")
                else:
                    logger.debug("FastAPI app instance not provided, skipping FastAPI instrumentation")
            except Exception as e:
                logger.warning(f"Failed to instrument FastAPI: {e}")

        logger.info(
            f"Logfire monitoring initialized: "
            f"project={settings.logfire.project_name}, "
            f"environment={settings.logfire.environment}, "
            f"service={settings.logfire.service_name}"
        )

    except ImportError:
        logger.warning(
            "Logfire is enabled but 'logfire' package is not installed. Install it with: pip install logfire"
        )
    except Exception as e:
        logger.error(f"Failed to initialize Logfire: {e}", exc_info=True)


def initialize_langsmith() -> None:
    """
    Initialize LangSmith for LangGraph agent tracing and monitoring.

    This function sets up LangSmith integration for:
    - Automatic LangGraph agent execution tracing
    - LLM call tracking and cost monitoring
    - Agent run performance metrics
    - Error and exception tracking in agent workflows

    LangSmith uses environment variables for configuration:
    - LANGSMITH__TRACING: Enable/disable tracing (true/false)
    - LANGSMITH__API_KEY: API key for authentication
    - LANGSMITH__PROJECT: Project name for organizing traces
    - LANGSMITH__ENDPOINT: API endpoint (default: https://api.smith.langchain.com)

    When LANGSMITH__TRACING=true, LangGraph runs are automatically traced.
    """
    if not settings.langsmith.tracing:
        logger.debug("LangSmith tracing is disabled. Set LANGSMITH__TRACING=true to enable.")
        return

    if not settings.langsmith.api_key:
        logger.warning(
            "LangSmith tracing is enabled but LANGSMITH__API_KEY is not set. "
            "Tracing will not work. Set LANGSMITH__API_KEY to enable LangSmith."
        )
        return

    try:
        # Set LangSmith environment variables for automatic instrumentation
        # These must be set before LangGraph/LangChain operations are initialized
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith.api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith.project
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith.endpoint

        logger.info(
            f"LangSmith tracing initialized: " f"project={settings.langsmith.project}, " f"endpoint={settings.langsmith.endpoint}"
        )

    except Exception as e:
        logger.error(f"Failed to initialize LangSmith: {e}", exc_info=True)


def wrap_openai_client(client):
    """
    Wrap an OpenAI client for LangSmith tracing.

    This wraps OpenAI API calls so they are automatically traced in LangSmith,
    capturing prompts, responses, and token usage.

    Args:
        client: An OpenAI client instance

    Returns:
        Wrapped OpenAI client with tracing enabled, or original client if wrapping fails
    """
    try:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(client)
    except Exception as e:
        logger.debug(f"Could not wrap OpenAI client for LangSmith tracing: {e}")
        return client


def wrap_anthropic_client(client):
    """
    Wrap an Anthropic client for LangSmith tracing.

    This wraps Anthropic API calls so they are automatically traced in LangSmith,
    capturing prompts, responses, and token usage.

    Args:
        client: An Anthropic client instance

    Returns:
        Wrapped Anthropic client with tracing enabled, or original client if wrapping fails
    """
    try:
        from langsmith.wrappers import wrap_anthropic

        return wrap_anthropic(client)
    except Exception as e:
        logger.debug(f"Could not wrap Anthropic client for LangSmith tracing: {e}")
        return client


def get_traceable_decorator():
    """
    Get the LangSmith @traceable decorator for tracing functions.

    The @traceable decorator traces function execution, capturing inputs, outputs,
    and any nested LangGraph/LangChain operations.

    Returns:
        The traceable decorator function, or a no-op decorator if LangSmith is unavailable
    """
    try:
        from langsmith import traceable

        return traceable
    except Exception as e:
        logger.debug(f"Could not import LangSmith traceable decorator: {e}")

        # Return a no-op decorator
        def noop_decorator(func):
            return func

        return noop_decorator


def get_langsmith_client():
    """
    Get a LangSmith client for manual trace operations.

    Returns:
        LangSmith Client instance, or None if unavailable
    """
    try:
        from langsmith import Client

        return Client()
    except Exception as e:
        logger.debug(f"Could not create LangSmith client: {e}")
        return None


def get_logfire_context() -> Optional[dict]:
    """
    Get the current Logfire context for adding custom attributes.

    Returns:
        Dictionary with current Logfire context or None if not available.
    """
    try:
        import logfire

        return logfire.current_trace_context()
    except Exception:
        return None


def log_agent_run(run_id: str, tenant_id: str, objective: str, role: str) -> None:
    """
    Log the start of an agent run with context.

    Args:
        run_id: The unique identifier for the run
        tenant_id: The tenant identifier
        objective: The agent's objective
        role: The agent's role
    """
    try:
        import logfire

        logfire.info(
            "Agent run started",
            run_id=run_id,
            tenant_id=tenant_id,
            objective=objective,
            role=role,
        )
    except Exception:
        logger.debug(f"Could not log agent run to Logfire: run_id={run_id}")


def log_agent_completion(run_id: str, status: str, duration_ms: float) -> None:
    """
    Log the completion of an agent run.

    Args:
        run_id: The unique identifier for the run
        status: The completion status (succeeded, failed, cancelled)
        duration_ms: The duration of the run in milliseconds
    """
    try:
        import logfire

        logfire.info(
            "Agent run completed",
            run_id=run_id,
            status=status,
            duration_ms=duration_ms,
        )
    except Exception:
        logger.debug(f"Could not log agent completion to Logfire: run_id={run_id}")


def log_llm_call(model: str, tokens_used: int, cost_usd: Optional[float] = None) -> None:
    """
    Log an LLM model call with usage metrics.

    Args:
        model: The model name
        tokens_used: Total tokens used in the call
        cost_usd: The cost in USD (optional)
    """
    try:
        import logfire

        logfire.info(
            "LLM call completed",
            model=model,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
        )
    except Exception:
        logger.debug(f"Could not log LLM call to Logfire: model={model}")


def log_error(error_type: str, error_message: str, context: Optional[dict] = None) -> None:
    """
    Log an error with context for debugging.

    Args:
        error_type: Type of error
        error_message: Error message
        context: Additional context dictionary
    """
    try:
        import logfire

        logfire.error(
            f"{error_type}: {error_message}",
            **(context or {}),
        )
    except Exception:
        logger.debug(f"Could not log error to Logfire: {error_type}")


# Initialize monitoring systems on module import
initialize_logfire()
initialize_langsmith()
