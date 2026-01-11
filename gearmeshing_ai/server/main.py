"""
Main Application Entry Point.

This module initializes the FastAPI application, configures middleware (CORS),
and includes all API routers. It serves as the root of the web server.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from gearmeshing_ai.core.logging_config import get_logger, setup_logging
from gearmeshing_ai.core.monitoring import initialize_logfire
from gearmeshing_ai.server.middleware.logfire_middleware import LogfireMiddleware

from .api.v1 import (
    agent_configs,
    approvals,
    chat_sessions,
    health,
    policies,
    roles,
    runs,
    usage,
)
from .core import constant
from .core.config import settings
from .core.database import checkpointer_pool, init_db

# Initialize logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan events.

    Handles startup and shutdown events for the FastAPI application.
    This is the modern approach replacing the deprecated @app.on_event decorators.
    """
    # Startup
    try:
        logger.info("Starting up GearMeshing-AI Server...")

        if settings.enable_database:
            logger.info("Database connectivity enabled. Initializing database...")
            await init_db()
            logger.info("Database initialized successfully")

            # Initialize LangGraph Checkpointer Pool
            await checkpointer_pool.open()

            # Ensure Checkpointer Tables exist
            # Note: AsyncPostgresSaver.setup() creates indexes with CREATE INDEX CONCURRENTLY,
            # which requires autocommit mode. The psycopg_pool connection context manager
            # handles this automatically when used without explicit transaction control.
            async with checkpointer_pool.connection() as conn:
                # Set autocommit mode to allow CREATE INDEX CONCURRENTLY
                await conn.set_autocommit(True)
                checkpointer = AsyncPostgresSaver(conn)
                await checkpointer.setup()
            logger.info("LangGraph checkpointer initialized successfully")
        else:
            logger.info("Running in standalone mode without database connectivity (ENABLE_DATABASE=false)")

    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down GearMeshing-AI Server...")
    if settings.enable_database:
        await checkpointer_pool.close()


app = FastAPI(
    title=constant.PROJECT_NAME,
    description="""
    GearMeshing-AI Server API

    This API provides the backend services for the GearMeshing-AI autonomous agent platform.
    It supports managing agent runs, handling approvals, configuring policies, and streaming real-time events.
    """,
    version="1.0.0",
    openapi_url=f"{constant.API_V1_STR}/openapi.json",
    docs_url=f"{constant.API_V1_STR}/docs",
    redoc_url=f"{constant.API_V1_STR}/redoc",
    lifespan=lifespan,
)

# Initialize Logfire monitoring with FastAPI app instance
initialize_logfire(app=app)

# Add Logfire middleware for request tracing
app.add_middleware(LogfireMiddleware)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health.router, tags=["health"])
app.include_router(runs.router, prefix=f"{constant.API_V1_STR}/runs", tags=["runs"])
app.include_router(approvals.router, prefix=f"{constant.API_V1_STR}/runs", tags=["approvals"])
app.include_router(policies.router, prefix=f"{constant.API_V1_STR}/policies", tags=["policies"])
app.include_router(roles.router, prefix=f"{constant.API_V1_STR}/roles", tags=["roles"])
app.include_router(usage.router, prefix=f"{constant.API_V1_STR}/usage", tags=["usage"])
app.include_router(agent_configs.router, prefix=f"{constant.API_V1_STR}/agent-config")
app.include_router(chat_sessions.router, prefix=f"{constant.API_V1_STR}/chat-sessions")
