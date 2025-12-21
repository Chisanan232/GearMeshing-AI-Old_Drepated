"""
Main Application Entry Point.

This module initializes the FastAPI application, configures middleware (CORS),
and includes all API routers. It serves as the root of the web server.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .api.v1 import (
    health,
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="""
    GearMeshing-AI Server API

    This API provides the backend services for the GearMeshing-AI autonomous agent platform.
    It supports managing agent runs, handling approvals, configuring policies, and streaming real-time events.
    """,
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
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
