from __future__ import annotations

from typing import Any, Dict

import pytest
from pydantic import ValidationError

from gearmeshing_ai.mcp_client.schemas.dto import ToolMetadata, ToolResult


def test_toolmetadata_contract_shape() -> None:
    good: Dict[str, Any] = {
        "name": "echo",
        "description": "Echo tool",
        "parameters": {"text": {"type": "string"}},
    }
    m: ToolMetadata = ToolMetadata.model_validate(good)
    assert m.name == "echo"
    assert "text" in (m.parameters or {})

    bad: Dict[str, Any] = {"description": "missing name"}
    with pytest.raises(ValidationError):
        ToolMetadata.model_validate(bad)


def test_toolresult_contract_shape() -> None:
    r: ToolResult = ToolResult(ok=True, data={"k": "v"})
    assert r.ok is True
    assert isinstance(r.data, dict)
