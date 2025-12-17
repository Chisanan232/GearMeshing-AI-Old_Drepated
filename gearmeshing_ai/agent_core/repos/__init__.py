"""Repository interfaces and SQL implementations for agent persistence.

The repository layer is the persistence boundary for the agent runtime.

Responsibilities
----------------

- Provide a small set of async repository interfaces (Protocols) that the
  runtime can depend on.
- Persist durable, auditable records of an agent run:

  - run metadata and status,
  - event stream (append-only),
  - approvals and their resolutions,
  - checkpoints (serialized graph state) for pause/resume,
  - tool invocation logs,
  - usage ledger entries (tokens/cost).

Design notes
------------

The runtime is intentionally written against interfaces so it can be used with:

- a SQL database (async SQLAlchemy implementation provided in ``repos.sql``),
- in-memory fakes for unit tests,
- future persistence backends.

The SQL implementation commits at repository-method boundaries. This keeps the
engine logic simple and ensures that each persisted artifact is written
atomically.
"""

from .interfaces import (
    ApprovalRepository,
    CheckpointRepository,
    EventRepository,
    RunRepository,
    ToolInvocationRepository,
    UsageRepository,
)

__all__ = [
    "RunRepository",
    "EventRepository",
    "ApprovalRepository",
    "CheckpointRepository",
    "ToolInvocationRepository",
    "UsageRepository",
]
