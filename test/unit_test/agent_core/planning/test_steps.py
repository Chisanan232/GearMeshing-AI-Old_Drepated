from __future__ import annotations

import pytest

from gearmeshing_ai.agent_core.planning.steps import normalize_plan
from gearmeshing_ai.agent_core.schemas.domain import CapabilityName


def test_normalize_plan_legacy_action_step_is_converted_to_kind_action() -> None:
    out = normalize_plan(
        [
            {"capability": CapabilityName.summarize.value, "args": {"text": "x"}},
        ]
    )
    assert out == [
        {
            "kind": "action",
            "capability": CapabilityName.summarize,
            "args": {"text": "x"},
            "server_id": None,
            "tool_name": None,
        }
    ]


def test_normalize_plan_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown step kind"):
        normalize_plan([{"kind": "wat"}])


def test_normalize_plan_rejects_thought_step_with_action_fields() -> None:
    with pytest.raises(ValueError, match="thought step cannot contain action fields"):
        normalize_plan(
            [
                {"kind": "thought", "thought": "x", "capability": CapabilityName.mcp_call.value},
            ]
        )
