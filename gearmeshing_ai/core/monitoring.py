"""
Monitoring and Tracing Configuration Module.

This module provides integration with Pydantic AI Logfire for comprehensive
monitoring and tracing of AI application operations, including:
- Agent execution traces
- LLM model calls and performance metrics
- API endpoint tracing
- Database operation monitoring
- Error tracking and logging
- Performance metrics collection

Features:
- Automatic instrumentation of Pydantic AI operations
- Structured logging with context
- Performance metrics and latency tracking
- Error and exception tracking
- Request/response tracing for API calls
"""

import logging
import os
from typing import Optional
from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Logfire configuration from environment
LOGFIRE_ENABLED = os.getenv("LOGFIRE_ENABLED", "false").lower() in ("true", "1", "yes")
LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN", "")
LOGFIRE_PROJECT_NAME = os.getenv("LOGFIRE_PROJECT_NAME", "gearmeshing-ai")
LOGFIRE_ENVIRONMENT = os.getenv("LOGFIRE_ENVIRONMENT", "development")
LOGFIRE_SERVICE_NAME = os.getenv("LOGFIRE_SERVICE_NAME", "gearmeshing-ai-server")
LOGFIRE_SERVICE_VERSION = os.getenv("LOGFIRE_SERVICE_VERSION", "0.0.0")

# Sampling configuration
LOGFIRE_SAMPLE_RATE = float(os.getenv("LOGFIRE_SAMPLE_RATE", "1.0"))
LOGFIRE_TRACE_SAMPLE_RATE = float(os.getenv("LOGFIRE_TRACE_SAMPLE_RATE", "1.0"))

# Feature flags
LOGFIRE_TRACE_PYDANTIC_AI = os.getenv("LOGFIRE_TRACE_PYDANTIC_AI", "true").lower() in ("true", "1", "yes")
LOGFIRE_TRACE_SQLALCHEMY = os.getenv("LOGFIRE_TRACE_SQLALCHEMY", "true").lower() in ("true", "1", "yes")
LOGFIRE_TRACE_HTTPX = os.getenv("LOGFIRE_TRACE_HTTPX", "true").lower() in ("true", "1", "yes")
LOGFIRE_TRACE_FASTAPI = os.getenv("LOGFIRE_TRACE_FASTAPI", "true").lower() in ("true", "1", "yes")


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

    The initialization is conditional based on LOGFIRE_ENABLED environment variable.
    """
    if not LOGFIRE_ENABLED:
        logger.info("Logfire monitoring is disabled. Set LOGFIRE_ENABLED=true to enable.")
        return

    if not LOGFIRE_TOKEN:
        logger.warning(
            "Logfire is enabled but LOGFIRE_TOKEN is not set. "
            "Monitoring will not work. Set LOGFIRE_TOKEN to enable Logfire."
        )
        return

    try:
        import logfire
        from logfire import SamplingOptions

        # Configure Logfire with project settings
        logfire.configure(
            token=LOGFIRE_TOKEN,
            service_name=LOGFIRE_SERVICE_NAME,
            service_version=LOGFIRE_SERVICE_VERSION,
            environment=LOGFIRE_ENVIRONMENT,
            sampling=SamplingOptions(
                head=LOGFIRE_SAMPLE_RATE,
                tail=LOGFIRE_TRACE_SAMPLE_RATE,
            ),
        )

        # Instrument Pydantic AI
        if LOGFIRE_TRACE_PYDANTIC_AI:
            try:
                logfire.instrument_pydantic_ai()
                logger.info("Logfire: Pydantic AI instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument Pydantic AI: {e}")

        # Instrument SQLAlchemy
        if LOGFIRE_TRACE_SQLALCHEMY:
            try:
                logfire.instrument_sqlalchemy()
                logger.info("Logfire: SQLAlchemy instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument SQLAlchemy: {e}")

        # Instrument HTTPX
        if LOGFIRE_TRACE_HTTPX:
            try:
                logfire.instrument_httpx()
                logger.info("Logfire: HTTPX instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument HTTPX: {e}")

        # Instrument FastAPI
        if LOGFIRE_TRACE_FASTAPI:
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
            f"project={LOGFIRE_PROJECT_NAME}, "
            f"environment={LOGFIRE_ENVIRONMENT}, "
            f"service={LOGFIRE_SERVICE_NAME}"
        )

    except ImportError:
        logger.warning(
            "Logfire is enabled but 'logfire' package is not installed. "
            "Install it with: pip install logfire"
        )
    except Exception as e:
        logger.error(f"Failed to initialize Logfire: {e}", exc_info=True)


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


def log_api_request(method: str, path: str, status_code: int, duration_ms: float) -> None:
    """
    Log an API request with performance metrics.

    Args:
        method: HTTP method
        path: Request path
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds
    """
    try:
        import logfire

        logfire.info(
            "API request completed",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration_ms,
        )
    except Exception:
        logger.debug(f"Could not log API request to Logfire: {method} {path}")


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


# Initialize Logfire on module import
initialize_logfire()
