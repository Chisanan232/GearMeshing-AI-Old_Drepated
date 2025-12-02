from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from pydantic_core._pydantic_core import ValidationError

from gearmeshing_ai.mcp_client.gateway_api import GatewayTransport
from gearmeshing_ai.mcp_client.gateway_api.models.dto import (
    GatewayServerCreate,
    ListServersQuery,
    ServerReadDTO,
    ServersListPayloadDTO,
)


def test_list_servers_query_to_params_contract() -> None:
    q = ListServersQuery(include_inactive=True, tags="prod,search", team_id="t1", visibility="team")
    params = q.to_params()
    assert params == {
        "include_inactive": "true",
        "tags": "prod,search",
        "team_id": "t1",
        "visibility": "team",
    }


def test_server_read_dto_aliases_contract() -> None:
    raw = {
        "id": "s1",
        "name": "github-mcp",
        "url": "http://underlying/mcp/",
        "transport": "STREAMABLEHTTP",
        "teamId": "team-123",
        "isActive": True,
        "tags": ["prod"],
        "visibility": "team",
        "description": "desc",
        "metrics": {"uptime": 99},
    }
    dto = ServerReadDTO.model_validate(raw)
    assert dto.team_id == "team-123" and dto.is_active is True
    dumped = dto.model_dump(by_alias=True, mode="json")
    assert dumped["teamId"] == "team-123"
    assert dumped["isActive"] is True


def test_servers_list_payload_dto_coercions_contract() -> None:
    items = [
        {
            "id": "s1",
            "name": "n1",
            "url": "http://u/mcp/",
            "transport": "STREAMABLEHTTP",
        }
    ]
    p1 = ServersListPayloadDTO.model_validate(items)
    assert len(p1.items) == 1 and isinstance(p1.items[0], ServerReadDTO)
    p2 = ServersListPayloadDTO.model_validate({"items": items})
    assert len(p2.items) == 1 and p2.items[0].id == "s1"
    p3 = ServersListPayloadDTO.model_validate({"servers": items})
    assert len(p3.items) == 1 and p3.items[0].name == "n1"


def test_gateway_server_create_validation_and_dump_contract() -> None:
    dto = GatewayServerCreate(
        name="clickup-mcp",
        url="http://clickup/mcp/",
        transport=GatewayTransport.STREAMABLE_HTTP,
        auth_token="Bearer token",
        tags=["prod"],
        visibility="team",
        team_id="team-1",
    )
    dumped = dto.model_dump(by_alias=True, mode="json")
    # Ensure core fields present and enum serialized
    assert dumped["name"] == "clickup-mcp"
    assert dumped["url"] == "http://clickup/mcp/"
    assert dumped["transport"] == GatewayTransport.STREAMABLE_HTTP
    # Validation: bad transport string and bad URL should fail
    with pytest.raises(ValidationError):
        GatewayServerCreate(name="x", url="not-a-url", transport=GatewayTransport.SSE)


@pytest.mark.parametrize("schema_key, expected_fields", [
    ("ServerRead", {"id", "name", "teamId", "isActive"}),
])
def test_openapi_contains_expected_gateway_schemas(schema_key: str, expected_fields: set[str]) -> None:
    spec_path = Path(__file__).parents[5] / "docs" / "openapi_spec" / "mcp_gateway.json"
    with spec_path.open("r", encoding="utf-8") as f:
        spec: Dict[str, Any] = json.load(f)
    comps = (spec.get("components") or {}).get("schemas") or {}
    candidates = {k: v for k, v in comps.items() if schema_key.lower() in k.lower()}
    assert candidates, f"Schema containing {schema_key} not found in spec. Available: {list(comps.keys())[:20]}"
    for name, schema in candidates.items():
        props = (schema or {}).get("properties") or {}
        keys = set(props.keys())
        if expected_fields.issubset(keys):
            return
    detail = {name: set(((schema or {}).get("properties") or {}).keys()) for name, schema in candidates.items()}
    raise AssertionError(
        f"None of the candidate schemas for {schema_key} contain fields {expected_fields}. Candidates: {detail}"
    )


def test_list_servers_query_to_params_contract() -> None:
    q = ListServersQuery(include_inactive=True, tags="prod,search", team_id="t1", visibility="team")
    params = q.to_params()
    assert params == {
        "include_inactive": "true",
        "tags": "prod,search",
        "team_id": "t1",
        "visibility": "team",
    }


def test_server_read_dto_aliases_contract() -> None:
    raw = {
        "id": "s1",
        "name": "github-mcp",
        "url": "http://underlying/mcp/",
        "transport": "STREAMABLEHTTP",
        "teamId": "team-123",
        "isActive": True,
        "tags": ["prod"],
        "visibility": "team",
        "description": "desc",
        "metrics": {"uptime": 99},
    }
    dto = ServerReadDTO.model_validate(raw)
    assert dto.team_id == "team-123" and dto.is_active is True
    # Ensure round-trip preserves aliases when dumping
    dumped = dto.model_dump(by_alias=True, mode="json")
    assert dumped["teamId"] == "team-123"
    assert dumped["isActive"] is True


def test_servers_list_payload_dto_coercions_contract() -> None:
    # Accept list
    items = [
        {
            "id": "s1",
            "name": "n1",
            "url": "http://u/mcp/",
            "transport": "STREAMABLEHTTP",
        }
    ]
    p1 = ServersListPayloadDTO.model_validate(items)
    assert len(p1.items) == 1 and isinstance(p1.items[0], ServerReadDTO)

    # Accept {items: [...]}
    p2 = ServersListPayloadDTO.model_validate({"items": items})
    assert len(p2.items) == 1 and p2.items[0].id == "s1"

    # Accept {servers: [...]}
    p3 = ServersListPayloadDTO.model_validate({"servers": items})
    assert len(p3.items) == 1 and p3.items[0].name == "n1"


@pytest.mark.parametrize("schema_key, expected_fields", [
    ("ServerRead", {"id", "name", "teamId", "isActive"}),
])
def test_openapi_contains_expected_gateway_schemas(schema_key: str, expected_fields: set[str]) -> None:
    # Parse OpenAPI spec to ensure presence of known schemas/fields.
    spec_path = Path(__file__).parents[5] / "docs" / "openapi_spec" / "mcp_gateway.json"
    with spec_path.open("r", encoding="utf-8") as f:
        spec: Dict[str, Any] = json.load(f)
    # Walk components/schemas and search by key substring
    comps = (spec.get("components") or {}).get("schemas") or {}
    # Find a schema whose key includes the expected name (e.g., "ServerRead")
    candidates = {k: v for k, v in comps.items() if schema_key.lower() in k.lower()}
    assert candidates, f"Schema containing {schema_key} not found in spec. Available: {list(comps.keys())[:20]}"
    # Check that expected fields are present in at least one candidate
    for name, schema in candidates.items():
        props = (schema or {}).get("properties") or {}
        keys = set(props.keys())
        if expected_fields.issubset(keys):
            return
    # If not found, show candidates for debugging
    detail = {name: set(((schema or {}).get("properties") or {}).keys()) for name, schema in candidates.items()}
    raise AssertionError(f"None of the candidate schemas for {schema_key} contain fields {expected_fields}. Candidates: {detail}")
