from typing import Any, Dict

from gearmeshing_ai.mcp_client.schemas.core import ToolMetadata, ToolResult


def test_toolmetadata_fields() -> None:
    params: Dict[str, Any] = {"text": {"type": "string"}}
    m: ToolMetadata = ToolMetadata(name="echo", description="Echo tool", parameters=params)
    assert m.name == "echo"
    assert "text" in (m.parameters or {})


def test_toolresult_ok() -> None:
    r: ToolResult = ToolResult(ok=True, data={"x": 1}, error=None)
    assert r.ok is True
    assert r.error is None
