"""
Orchestrator Dependency.

Provides a singleton instance of the OrchestratorService for API endpoints.
"""
from typing import Annotated
from fastapi import Depends
from gearmeshing_ai.server.services.orchestrator import get_orchestrator, OrchestratorService

OrchestratorDep = Annotated[OrchestratorService, Depends(get_orchestrator)]
