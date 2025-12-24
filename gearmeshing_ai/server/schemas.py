"""
API Schemas.

This module contains Pydantic models used for API request bodies and response validation.
These schemas define the interface contract between the client and the server.
"""
from typing import Optional, Any, Dict, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

from gearmeshing_ai.agent_core.schemas.domain import AutonomyProfile, ApprovalDecision


class RunCreate(BaseModel):
    """
    Schema for creating a new agent run.

    Defines the initial parameters required to start an agent execution session.
    """
    objective: str = Field(
        ..., 
        description="The high-level goal or task for the agent to accomplish.",
        examples=["Analyze the latest quarterly report for AAPL."]
    )
    tenant_id: str = Field(
        ..., 
        description="The identifier of the tenant initiating the run.",
        examples=["tenant-123"]
    )
    role: Optional[str] = Field(
        default=None, 
        description="The specific role the agent should assume. Defaults to 'planner' if not specified.",
        examples=["planner", "dev", "researcher"]
    )
    input: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Initial context or data payload for the run.",
        examples=[{"url": "https://example.com/report.pdf"}]
    )
    autonomy_profile: Optional[AutonomyProfile] = Field(
        default=None, 
        description="The level of autonomy granted to the agent. Affects approval requirements.",
        examples=[AutonomyProfile.balanced]
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "objective": "Research competitive landscape for AI agents",
            "tenant_id": "acme-corp",
            "role": "market",
            "autonomy_profile": "balanced",
            "input": {"focus_area": "enterprise"}
        }
    })

class RunResume(BaseModel):
    """
    Schema for resuming a paused agent run.

    Typically used when a run is halted waiting for human approval.
    """
    approved_by: Optional[str] = Field(
        default=None, 
        description="The identifier of the user or system authorizing the resume.",
        examples=["user-456"]
    )
    note: Optional[str] = Field(
        default=None, 
        description="Optional note or reason for resuming the run.",
        examples=["Budget approved"]
    )

class ApprovalSubmit(BaseModel):
    """
    Schema for submitting an approval decision.

    Used by the approvals API to record a human's verdict on a pending request.
    """
    decision: ApprovalDecision = Field(
        ..., 
        description="The decision made on the approval request (approved, rejected, etc.)."
    )
    note: Optional[str] = Field(
        default=None, 
        description="Optional justification or context for the decision.",
        examples=["Risk is acceptable within current scope."]
    )

class PolicyUpdate(BaseModel):
    """
    Schema for updating tenant policies.

    Allows modifying configuration settings like allowed tools and budget limits.
    """
    config: Dict[str, Any] = Field(
        ..., 
        description="The configuration object to update or merge into the tenant's policy.",
        examples=[{"allowed_tools": ["web_search"], "max_budget_usd": 50.0}]
    )

class RolePromptOverride(BaseModel):
    """
    Schema for overriding system prompts for specific roles.

    Enables tenant-specific customization of agent behavior.
    """
    tenant_id: str = Field(
        ..., 
        description="The tenant identifier for which the override applies."
    )
    prompt: str = Field(
        ..., 
        description="The custom system prompt text to use for the role.",
        examples=["You are a senior Python developer with a focus on security."]
    )


# ============================================================================
# Server-Sent Events (SSE) Schemas
# ============================================================================
# These models ensure all event data streamed via SSE is JSON-serializable
# with proper datetime handling.

class ThinkingData(BaseModel):
    """Enriched thinking event data.
    
    Represents a single thought or reasoning step executed by the agent during
    the run. Includes the thought content, sequence index, and timing information.
    """
    thought: Optional[str] = Field(
        default=None,
        description="The content of the thought or reasoning step.",
        examples=["I need to analyze the quarterly revenue trends."]
    )
    index: Optional[int] = Field(
        default=None,
        alias="idx",
        description="The sequence index of this thought within the run (0-based).",
        examples=[0, 1, 2]
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="The ISO 8601 timestamp when this thought was executed.",
        examples=["2025-12-24T22:00:00Z"]
    )

    class Config:
        populate_by_name = True


class ThinkingOutputData(BaseModel):
    """Enriched thinking output (artifact) data.
    
    Represents an artifact or output generated from a thinking process.
    Includes the original thought, structured data, formatted output, and
    the system prompt configuration used.
    """
    thought: Optional[str] = Field(
        default=None,
        description="The original thought or reasoning that led to this output.",
        examples=["I should create a summary of findings."]
    )
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured data extracted or generated during thinking.",
        examples=[{"key_findings": ["trend1", "trend2"], "confidence": 0.95}]
    )
    output: Optional[str] = Field(
        default=None,
        description="The formatted output or result of the thinking process.",
        examples=["The quarterly revenue increased by 15% YoY."]
    )
    system_prompt_key: Optional[str] = Field(
        default=None,
        description="The key or identifier of the system prompt used for this output.",
        examples=["analyst_prompt_v1", "summarizer_prompt_v2"]
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="The ISO 8601 timestamp when this artifact was created.",
        examples=["2025-12-24T22:00:05Z"]
    )


class OperationData(BaseModel):
    """Enriched capability execution data.
    
    Represents the execution of a capability (e.g., web search, code execution).
    Includes the capability name, execution status, and the result produced.
    """
    capability: Optional[str] = Field(
        default=None,
        description="The name of the capability that was executed.",
        examples=["web_search", "code_execution", "docs_read"]
    )
    status: str = Field(
        default="unknown",
        description="The execution status: 'success' or 'failed'.",
        examples=["success", "failed"]
    )
    result: Optional[Any] = Field(
        default=None,
        description="The result or output produced by the capability execution.",
        examples=[{"search_results": [{"title": "...", "url": "..."}]}]
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="The ISO 8601 timestamp when the capability was executed.",
        examples=["2025-12-24T22:00:10Z"]
    )


class ToolExecutionData(BaseModel):
    """Enriched tool invocation data.
    
    Represents a specific tool call made through an MCP server.
    Includes the tool name, arguments, result, success status, and risk level.
    """
    server_id: Optional[str] = Field(
        default=None,
        description="The identifier of the MCP server providing this tool.",
        examples=["github-mcp", "slack-mcp", "aws-mcp"]
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="The name of the tool being invoked.",
        examples=["create_issue", "send_message", "list_buckets"]
    )
    args: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The arguments passed to the tool.",
        examples=[{"title": "Bug fix", "body": "Fix memory leak"}]
    )
    result: Optional[Any] = Field(
        default=None,
        description="The result returned by the tool.",
        examples=[{"issue_id": "123", "url": "https://github.com/..."}, "Success"]
    )
    ok: Optional[bool] = Field(
        default=None,
        description="Whether the tool execution succeeded (true) or failed (false).",
        examples=[True, False]
    )
    risk: Optional[str] = Field(
        default=None,
        description="The risk level of this tool execution: 'low', 'medium', or 'high'.",
        examples=["low", "medium", "high"]
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="The ISO 8601 timestamp when the tool was invoked.",
        examples=["2025-12-24T22:00:15Z"]
    )


class ApprovalRequestData(BaseModel):
    """Enriched approval request data.
    
    Represents a request for human approval on a high-risk operation.
    Includes the capability being requested, its risk level, and the reason
    why approval is required.
    """
    capability: Optional[str] = Field(
        default=None,
        description="The capability that requires approval.",
        examples=["shell_exec", "code_execution", "delete_resource"]
    )
    risk: Optional[str] = Field(
        default=None,
        description="The risk level of the operation: 'low', 'medium', or 'high'.",
        examples=["low", "medium", "high"]
    )
    reason: Optional[str] = Field(
        default=None,
        description="The reason why this operation requires approval.",
        examples=["High-risk operation that could affect production data.", "Requires elevated permissions."]
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="The ISO 8601 timestamp when the approval was requested.",
        examples=["2025-12-24T22:00:20Z"]
    )


class ApprovalResolutionData(BaseModel):
    """Enriched approval resolution data.
    
    Represents the resolution of an approval request.
    Includes the decision made, who made it, and when.
    """
    decision: Optional[str] = Field(
        default=None,
        description="The decision on the approval request: 'approved', 'rejected', or 'expired'.",
        examples=["approved", "rejected", "expired"]
    )
    decided_by: Optional[str] = Field(
        default=None,
        description="The identifier of the user or system that made the decision.",
        examples=["user-123", "admin-456", "system"]
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="The ISO 8601 timestamp when the decision was made.",
        examples=["2025-12-24T22:00:25Z"]
    )


class RunStartData(BaseModel):
    """Enriched run start event data.
    
    Represents the start of an agent run.
    Includes the run identifier and the start timestamp.
    """
    run_id: Optional[str] = Field(
        default=None,
        description="The unique identifier of the run that started.",
        examples=["run-123", "run-abc-def-456"]
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="The ISO 8601 timestamp when the run started.",
        examples=["2025-12-24T22:00:00Z"]
    )


class RunCompletionData(BaseModel):
    """Enriched run completion event data.
    
    Represents the successful completion of an agent run.
    Includes the completion status and timestamp.
    """
    status: str = Field(
        default="succeeded",
        description="The completion status of the run: 'succeeded' or 'cancelled'.",
        examples=["succeeded", "cancelled"]
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="The ISO 8601 timestamp when the run completed.",
        examples=["2025-12-24T22:05:00Z"]
    )


class RunFailureData(BaseModel):
    """Enriched run failure event data.
    
    Represents the failure of an agent run.
    Includes the error message and the failure timestamp.
    """
    error: Optional[str] = Field(
        default=None,
        description="The error message or description of why the run failed.",
        examples=["Connection timeout", "Out of memory", "Invalid input provided"]
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="The ISO 8601 timestamp when the run failed.",
        examples=["2025-12-24T22:03:45Z"]
    )


class SSEEventData(BaseModel):
    """
    Base SSE event data structure.
    
    All events include:
    - id: Unique event identifier
    - type: Event type (thought_executed, artifact_created, etc.)
    - category: Event category (thinking, operation, approval, run_lifecycle, etc.)
    - created_at: Event creation timestamp
    - run_id: Associated run ID
    - payload: Original event payload
    - Enriched fields based on event type (thinking, operation, etc.)
    """
    id: Optional[str] = Field(
        default=None,
        description="The unique identifier for this event.",
        examples=["evt-123", "evt-abc-def-456"]
    )
    type: str = Field(
        ...,
        description="The event type (e.g., 'run.started', 'thought.executed', 'approval.requested').",
        examples=["run.started", "thought.executed", "artifact.created", "approval.requested"]
    )
    category: str = Field(
        ...,
        description="The event category for grouping: 'thinking', 'operation', 'approval', 'run_lifecycle', or 'other'.",
        examples=["thinking", "operation", "approval", "run_lifecycle", "other"]
    )
    created_at: datetime = Field(
        ...,
        description="The ISO 8601 timestamp when the event was created.",
        examples=["2025-12-24T22:00:00Z"]
    )
    run_id: Optional[str] = Field(
        default=None,
        description="The identifier of the run associated with this event.",
        examples=["run-123", "run-abc-def-456"]
    )
    payload: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The original event payload containing raw event data.",
        examples=[{"thought": "...", "idx": 0}]
    )
    
    # Enriched fields (optional, populated based on event type)
    thinking: Optional[ThinkingData] = Field(
        default=None,
        description="Enriched thinking data (populated for thought_executed events)."
    )
    thinking_output: Optional[ThinkingOutputData] = Field(
        default=None,
        description="Enriched thinking output data (populated for artifact_created events with kind='thought')."
    )
    operation: Optional[OperationData] = Field(
        default=None,
        description="Enriched capability execution data (populated for capability_executed events)."
    )
    tool_execution: Optional[ToolExecutionData] = Field(
        default=None,
        description="Enriched tool invocation data (populated for tool_invoked events)."
    )
    approval_request: Optional[ApprovalRequestData] = Field(
        default=None,
        description="Enriched approval request data (populated for approval_requested events)."
    )
    approval_resolution: Optional[ApprovalResolutionData] = Field(
        default=None,
        description="Enriched approval resolution data (populated for approval_resolved events)."
    )
    run_start: Optional[RunStartData] = Field(
        default=None,
        description="Enriched run start data (populated for run.started events)."
    )
    run_completion: Optional[RunCompletionData] = Field(
        default=None,
        description="Enriched run completion data (populated for run.completed events)."
    )
    run_failure: Optional[RunFailureData] = Field(
        default=None,
        description="Enriched run failure data (populated for run.failed events)."
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class SSEResponse(BaseModel):
    """
    SSE response wrapper.
    
    Wraps the event data for streaming responses.
    """
    data: SSEEventData = Field(
        ...,
        description="The enriched event data to be streamed to the client."
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class KeepAliveEvent(BaseModel):
    """Keep-alive event for idle streams.
    
    Sent periodically when no events are available to prevent client timeout.
    """
    comment: str = Field(
        default="keep-alive",
        description="A fixed comment indicating this is a keep-alive message.",
        examples=["keep-alive"]
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class ErrorEvent(BaseModel):
    """Error event for stream failures.
    
    Sent when an error occurs during event streaming.
    """
    error: str = Field(
        ...,
        description="The error message or error type.",
        examples=["Database connection failed", "Timeout", "Invalid run ID"]
    )
    details: Optional[str] = Field(
        default=None,
        description="Additional details or context about the error.",
        examples=["Connection refused at 192.168.1.1:5432", "Request timed out after 30 seconds"]
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
