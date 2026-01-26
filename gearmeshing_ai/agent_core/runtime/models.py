from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver

"""Runtime dependency bundle and LangGraph state types.

The runtime engine is designed to be dependency-injected.

- ``EngineDeps`` collects the repositories and registries the engine needs.
- ``_GraphState`` is the mutable state passed between LangGraph nodes.

These structures are intentionally small and serializable so they can be
checkpointed for pause/resume.
"""

from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    NotRequired,
    Optional,
    Required,
    TypedDict,
)

from ...info_provider.mcp.base import BaseAsyncMCPInfoProvider
from ...info_provider.prompt.base import PromptProvider
from ..capabilities import CapabilityRegistry
from ..repos import (
    ApprovalRepository,
    CheckpointRepository,
    EventRepository,
    RunRepository,
    ToolInvocationRepository,
    UsageRepository,
)
from gearmeshing_ai.info_provider import load_role_provider, RoleProvider


@dataclass(frozen=True)
class EngineDeps:
    """
    Dependency bundle for ``AgentEngine``.

    This object acts as the interface between the core runtime engine and the
    external world (persistence, LLMs, tools). It allows the engine to be
    instantiated with different backends (e.g., in-memory vs SQL repositories).

    Attributes:
        runs: Repository for managing AgentRun lifecycle.
        events: Append-only log for all runtime events.
        approvals: Repository for creating and querying approval requests.
        checkpoints: Repository for saving/loading LangGraph state snapshots.
        tool_invocations: Audit log for tool calls.
        capabilities: Registry of executable capabilities.
        usage: Optional repository for tracking token usage and costs.
        prompt_provider: Optional service to resolve system prompts.
        role_provider: Optional service to resolve agent role definitions.
        thought_model: Optional LLM model instance for generating 'thoughts'.
        mcp_info_provider: Optional provider for MCP tool metadata.
        mcp_call: Optional callable for executing MCP tools.
    """

    runs: RunRepository
    events: EventRepository
    approvals: ApprovalRepository
    checkpoints: CheckpointRepository
    tool_invocations: ToolInvocationRepository
    capabilities: CapabilityRegistry

    # Native LangGraph checkpointer
    checkpointer: BaseCheckpointSaver

    usage: Optional[UsageRepository] = None

    prompt_provider: Optional[PromptProvider] = None
    role_provider: Optional[RoleProvider] = None
    thought_model: Any | None = None
    mcp_info_provider: Optional[BaseAsyncMCPInfoProvider] = None
    mcp_call: Optional[Callable[[str, str, Dict[str, Any]], Awaitable[Any]]] = None


class _GraphState(TypedDict):
    """
    Mutable LangGraph state for a single engine run.

    This dictionary is passed between nodes in the StateGraph. It maintains the
    current execution cursor and the plan.

    Attributes:
        run_id (str): The unique identifier of the current AgentRun.
        plan (List[Dict[str, Any]]): The normalized sequence of steps to execute.
        idx (int): The index of the current step in the plan.
        awaiting_approval_id (Optional[str]): If set, indicates the run is paused waiting for this approval.
        _finished (bool): Internal flag to signal successful completion.
        _terminal_status (str): The final status to set on the run (e.g. 'succeeded', 'failed').
        _resume_skip_approval (bool): One-shot flag to bypass approval checks immediately after resumption.
    """

    run_id: Required[str]
    plan: Required[List[Dict[str, Any]]]
    idx: Required[int]
    awaiting_approval_id: Required[Optional[str]]
    _finished: NotRequired[bool]
    _terminal_status: NotRequired[str]
    _resume_skip_approval: NotRequired[bool]
