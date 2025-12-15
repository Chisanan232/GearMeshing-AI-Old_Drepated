from __future__ import annotations

from datetime import datetime, timezone

from gearmeshing_ai.info_provider.mcp.gateway_api.models.dto import (
    AdminToolsListResponseDTO,
    CatalogBulkRegisterResponseDTO,
    CatalogListResponseDTO,
    CatalogRegisterFailureDTO,
    CatalogServerDTO,
    GatewayReadDTO,
    ToolReadDTO,
)


def _sample_tools_payload() -> dict:
    now = datetime.now(timezone.utc).isoformat()
    base = {
        "requestType": "SSE",
        "integrationType": "MCP",
        "inputSchema": {"type": "object", "properties": {}},
        "createdAt": now,
        "updatedAt": now,
        "enabled": True,
        "reachable": True,
        "executionCount": 0,
        "metrics": {
            "totalExecutions": 0,
            "successfulExecutions": 0,
            "failedExecutions": 0,
            "failureRate": 0.0,
        },
        "gatewaySlug": "gw",
    }
    t1 = {
        **base,
        "id": "t1",
        "originalName": "workspace.list",
        "name": "tool-one",
        "customName": "workspace.list",
        "customNameSlug": "workspace-list",
    }
    t2 = {
        **base,
        "id": "t2",
        "originalName": "get_authorized_teams",
        "name": "tool-two",
        "customName": "get_authorized_teams",
        "customNameSlug": "get-authorized-teams",
    }
    return {
        "data": [t1, t2],
        "pagination": {"page": 1, "per_page": 50, "total_items": 2, "total_pages": 1},
        "links": {"self": "/admin/tools?page=1&per_page=50"},
    }


def test_admin_tools_list_dto_contract() -> None:
    data = _sample_tools_payload()
    dto = AdminToolsListResponseDTO.model_validate(data)

    # Validate presence and typing of nested items
    assert dto.data is not None and len(dto.data) >= 2
    first: ToolReadDTO = dto.data[0]
    assert isinstance(first, ToolReadDTO)
    assert first.id and first.originalName and first.requestType == "SSE"
    assert first.metrics.totalExecutions == 0
    assert first.enabled is True and first.reachable is True
    assert first.integrationType == "MCP"

    # Pagination and links are accepted with extra fields
    assert dto.pagination is not None and dto.pagination.page == 1
    assert dto.links is not None and dto.links.self


def test_catalog_list_response_dto_contract_minimal() -> None:
    data = {
        "servers": [
            {
                "id": "clickup",
                "name": "clickup",
                "category": "Utilities",
                "url": "http://clickup-mcp:8082/sse/sse",
                "auth_type": "Open",
                "provider": "E2E",
                "description": "Project management tool",
                "transport": "SSE",
            }
        ],
        "total": 1,
        "categories": ["Utilities"],
        "auth_types": ["Open"],
        "providers": ["E2E"],
        "all_tags": ["clickup", "project-management", "python"],
    }
    dto = CatalogListResponseDTO.model_validate(data)
    assert dto.total == 1
    assert isinstance(dto.servers[0], CatalogServerDTO)
    s = dto.servers[0]
    assert s.id == "clickup" and s.transport == "SSE" and s.provider == "E2E"


def test_gateway_read_dto_contract_minimal() -> None:
    data = {
        "id": "g1",
        "name": "gw",
        "url": "http://mock",
        "transport": "SSE",
        "enabled": True,
        "reachable": True,
    }
    dto = GatewayReadDTO.model_validate(data)
    assert dto.id == "g1" and dto.name == "gw" and dto.transport == "SSE"


def test_catalog_bulk_register_response_dto_normalizes_failed_objects() -> None:
    raw = {
        "successful": ["s3"],
        "failed": [{"s1": "error-one"}, {"s2": "error-two"}],
        "total_attempted": 3,
        "total_successful": 1,
    }

    dto = CatalogBulkRegisterResponseDTO.model_validate(raw)

    assert dto.successful == ["s3"]
    assert dto.total_attempted == 3
    assert dto.total_successful == 1

    assert len(dto.failed) == 2
    first: CatalogRegisterFailureDTO = dto.failed[0]
    second: CatalogRegisterFailureDTO = dto.failed[1]

    assert first.server_id == "s1" and first.error == "error-one"
    assert second.server_id == "s2" and second.error == "error-two"


def test_catalog_bulk_register_response_dto_ignores_non_dict_failed_entries() -> None:
    raw = {
        "successful": [],
        "failed": [
            {"s1": "error-one"},
            "not-a-dict",
            123,
            {"s2": "error-two"},
        ],
        "total_attempted": 2,
        "total_successful": 0,
    }

    dto = CatalogBulkRegisterResponseDTO.model_validate(raw)

    # Only dict entries should be normalized
    assert len(dto.failed) == 2
    ids = {f.server_id for f in dto.failed}
    errors = {f.error for f in dto.failed}
    assert ids == {"s1", "s2"}
    assert errors == {"error-one", "error-two"}


def test_catalog_bulk_register_response_dto_handles_additional_properties_map_shape() -> None:
    raw = {
        "successful": ["s1"],
        # Gateway API may return failed as objects with additionalProperties: string
        # e.g. [{"s2": "e2", "s3": "e3"}]
        "failed": [
            {"s2": "e2", "s3": "e3"},
        ],
        "total_attempted": 3,
        "total_successful": 1,
    }

    dto = CatalogBulkRegisterResponseDTO.model_validate(raw)

    assert dto.successful == ["s1"]
    assert dto.total_attempted == 3
    assert dto.total_successful == 1

    ids = {f.server_id for f in dto.failed}
    errors = {f.error for f in dto.failed}
    assert ids == {"s2", "s3"}
    assert errors == {"e2", "e3"}


def test_catalog_bulk_register_response_dto_validator_noop_for_non_dict_root() -> None:
    # Directly exercise the pre-validation hook: non-dict input should be passed through unchanged
    payload = [
        {"s1": "e1"},
    ]

    result = CatalogBulkRegisterResponseDTO._coerce_failed(payload)  # type: ignore[arg-type]

    assert result is payload
