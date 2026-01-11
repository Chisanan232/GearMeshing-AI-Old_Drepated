"""
Approvals API Endpoints.

This module provides endpoints for managing human-in-the-loop approvals.
It allows listing pending approvals and submitting decisions (approve/reject).
"""

from typing import List

from fastapi import APIRouter, BackgroundTasks, HTTPException

from gearmeshing_ai.agent_core.schemas.domain import Approval, ApprovalDecision
from gearmeshing_ai.server.schemas import ApprovalSubmit
from gearmeshing_ai.server.services.deps import OrchestratorDep

router = APIRouter()


@router.get(
    "/{run_id}/approvals",
    response_model=List[Approval],
    summary="List Pending Approvals",
    description="Retrieve a list of pending approval requests for a specific run.",
    response_description="A list of pending approval objects.",
)
async def list_approvals(run_id: str, orchestrator: OrchestratorDep):
    """
    List pending approvals.

    Fetches all approval requests for the given run ID that have not yet been decided.
    """
    return await orchestrator.get_pending_approvals(run_id)


@router.post(
    "/{run_id}/approvals/{approval_id}",
    response_model=Approval,
    summary="Submit Approval Decision",
    description="Submit a decision (approve/reject) for a specific approval request.",
    response_description="The updated approval object.",
    responses={
        404: {"description": "Approval not found or run ID mismatch"},
        400: {"description": "Approval already decided"},
    },
)
async def submit_approval(
    run_id: str,
    approval_id: str,
    submission: ApprovalSubmit,
    orchestrator: OrchestratorDep,
    background_tasks: BackgroundTasks,
):
    """
    Submit approval decision.

    Resolves a pending approval request. Once resolved, the agent run may resume
    if it was blocked by this approval and the decision was 'approved'.
    """
    approval = await orchestrator.submit_approval(
        run_id=run_id,
        approval_id=approval_id,
        decision=submission.decision.value,
        note=submission.note,
        decided_by="user-placeholder",  # In real app, from auth
    )
    if not approval:
        raise HTTPException(status_code=404, detail="Approval not found")

    if submission.decision == ApprovalDecision.approved:
        # Schedule background resumption
        background_tasks.add_task(orchestrator.execute_resume, run_id, approval_id)

    return approval
