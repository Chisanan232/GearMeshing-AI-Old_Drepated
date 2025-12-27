"""
Main Application Entry Point.

This module initializes the FastAPI application, configures middleware (CORS),
and includes all API routers. It serves as the root of the web server.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gearmeshing_ai.core.logging_config import get_logger, setup_logging

from .api.v1 import (
    agent_configs,
    runs,
    health,
    policies,
    roles,
    usage,
)
from .core import constant
from .core.database import init_db

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
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)

    yield

    # Shutdown
    logger.info("Shutting down GearMeshing-AI Server...")


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
app.include_router(policies.router, prefix=f"{constant.API_V1_STR}/policies", tags=["policies"])
app.include_router(roles.router, prefix=f"{constant.API_V1_STR}/roles", tags=["roles"])
app.include_router(usage.router, prefix=f"{constant.API_V1_STR}/usage", tags=["usage"])
app.include_router(agent_configs.router, prefix=f"{constant.API_V1_STR}/agent-config")
