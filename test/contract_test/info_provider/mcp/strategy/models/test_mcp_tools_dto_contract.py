from __future__ import annotations

from typing import Any, Dict, List

import pytest

from gearmeshing_ai.info_provider.mcp.schemas.core import (
    McpTool,
    ToolCallResult,
    ToolsPage,
)
from gearmeshing_ai.info_provider.mcp.strategy.models.dto import (
    ToolDescriptorDTO,
    ToolInvokePayloadDTO,
    ToolsListPayloadDTO,
    ToolsListQuery,
)


def test_tool_descriptor_dto_contract_alias_and_mapping() -> None:
    raw = {
        "name": "get_issue",
        "description": "Fetch an issue",
        "inputSchema": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
        "x-mutating": False,
    }
    dto = ToolDescriptorDTO.model_validate(raw)
    assert dto.name == "get_issue"
    assert dto.input_schema["type"] == "object"

    # Mapping to domain
    def _infer(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        props = (schema or {}).get("properties") or {}
        req = set((schema or {}).get("required") or [])
        return [
            {
                "name": k,
                "type": v.get("type", "string"),
                "required": k in req,
                "description": v.get("description"),
            }
            for k, v in props.items()
            if isinstance(v, dict)
        ]

    t = dto.to_mcp_tool(_infer, lambda n: n.startswith("create"))
    assert isinstance(t, McpTool)
    assert t.name == "get_issue" and t.mutating is False
    assert t.raw_parameters_schema.get("type") == "object"


def test_tools_list_payload_dto_coercions_contract() -> None:
    # list -> tools
    lst = [
        {"name": "echo", "inputSchema": {"type": "object"}},
        {"name": "get", "inputSchema": {"type": "object"}},
    ]
    p1 = ToolsListPayloadDTO.model_validate(lst)
    assert [d.name for d in p1.tools] == ["echo", "get"]
    # dict with items -> tools
    p2 = ToolsListPayloadDTO.model_validate({"items": lst})
    assert [d.name for d in p2.tools] == ["echo", "get"]
    # dict with tools stays
    p3 = ToolsListPayloadDTO.model_validate({"tools": lst, "nextCursor": "c2"})
    assert [d.name for d in p3.tools] == ["echo", "get"] and p3.next_cursor == "c2"


def test_tools_list_query_to_params_contract() -> None:
    q = ToolsListQuery(cursor="abc", limit=50)
    assert q.to_params() == {"cursor": "abc", "limit": "50"}


essential_cases = [
    ({"ok": True, "result": 1}, True, {"result": 1}),
    ({"value": 1}, True, {"value": 1}),
]


@pytest.mark.parametrize("body, expected_ok, expected_data", essential_cases)
def test_tool_invoke_payload_dto_coercion_and_result(
    body: Any, expected_ok: bool, expected_data: Dict[str, Any]
) -> None:
    inv = ToolInvokePayloadDTO.model_validate(body)
    res: ToolCallResult = inv.to_tool_call_result()
    assert res.ok == expected_ok
    assert res.data == expected_data


def test_tools_page_alias_dump_contract() -> None:
    page = ToolsPage(
        items=[McpTool(name="a", description=None, mutating=False, arguments=[], raw_parameters_schema={})],
        next_cursor="next",
    )
    dumped = page.model_dump(by_alias=True, mode="json")
    assert dumped.get("nextCursor") == "next"
