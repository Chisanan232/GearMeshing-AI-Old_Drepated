"""GearMeshing-AI.

This package contains an agent framework used by GearMeshing-AI to turn a user
objective into a persisted, auditable execution.

High-level architecture
-----------------------

The codebase is organized around two major concepts:

- **Thought (cognitive) operations**: LLM-only steps that produce structured
  artifacts (plans, summaries, intermediate reasoning outputs) and never perform
  side effects.
- **Action (side-effecting) operations**: steps that execute capabilities/tools
  (e.g. web fetch, shell execution, MCP calls) and may be gated by safety checks
  and human approval.

Core subpackages
----------------

- ``gearmeshing_ai.agent_core``:

  - Planning and step schemas.
  - A LangGraph-based execution engine with pause/resume.
  - Policy primitives (tool allow/deny, risk classification, approval, safety).
  - Repository interfaces and SQL implementations for persistence.

- ``gearmeshing_ai.info_provider``:

  - Provider integrations used by capabilities (e.g. MCP client strategies and
    prompt providers).

Typical workflow
----------------

Most integrations should use ``gearmeshing_ai.agent_core.service.AgentService``:

1. Create an ``AgentRun``.
2. Generate a plan (mixed Thought/Action steps).
3. Execute the plan.
4. If an action requires approval, the run pauses and stores a checkpoint.
5. After approval is resolved, resume from the checkpoint.

This separation is enforced structurally: Thought steps cannot contain tool
fields and cannot trigger approvals or tool invocations.
"""
