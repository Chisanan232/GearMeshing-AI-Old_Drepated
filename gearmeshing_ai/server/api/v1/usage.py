"""
Usage Statistics Endpoints.

This module provides endpoints for querying aggregated token usage and
cost estimates associated with agent runs and tenants.
"""
from fastapi import APIRouter, Query
from sqlmodel import select, func
from typing import List, Optional
from datetime import datetime

from gearmeshing_ai.server.services.deps import OrchestratorDep
from gearmeshing_ai.agent_core.schemas.domain import UsageLedgerEntry

router = APIRouter()

@router.get(
    "/",
    summary="Get Usage Statistics",
    description="Retrieve aggregated token usage and cost statistics for a tenant.",
    response_description="Aggregated usage data."
)
async def get_usage(
    orchestrator: OrchestratorDep,
    tenant_id: str,
    from_date: Optional[datetime] = Query(None, alias="from", description="Start date for filtering usage."),
    to_date: Optional[datetime] = Query(None, alias="to", description="End date for filtering usage.")
):
    """
    Get aggregated usage.

    Calculates the total token consumption (prompt, completion, total) and estimated cost
    for a tenant within the specified date range.
    """
    entries = await orchestrator.list_usage(tenant_id=tenant_id, from_date=from_date, to_date=to_date)
    
    # Aggregate in code
    total_tokens = sum(e.total_tokens for e in entries)
    total_cost = sum(e.cost_usd or 0.0 for e in entries)
    
    return {
        "tenant_id": tenant_id,
        "period": {"from": from_date, "to": to_date},
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
        "entries_count": len(entries)
    }
