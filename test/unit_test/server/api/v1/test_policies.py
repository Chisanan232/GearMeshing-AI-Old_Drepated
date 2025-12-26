import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_get_policy_not_found(client: AsyncClient):
    """Test getting a non-existent policy returns 404."""
    response = await client.get("/api/v1/policies/non-existent-tenant-xyz-123")
    assert response.status_code == 404


async def test_update_and_get_policy(client: AsyncClient):
    """Test creating and retrieving a policy."""
    import uuid

    tenant_id = f"test-tenant-policy-{uuid.uuid4().hex[:8]}"
    payload = {"config": {"autonomy_profile": "strict", "tool_policy": {"allowed_tools": ["search"]}}}

    # Update (Create)
    response = await client.put(f"/api/v1/policies/{tenant_id}", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["tenant_id"] == tenant_id
    assert data["config"]["autonomy_profile"] == "strict"
    assert data["config"]["tool_policy"]["allowed_tools"] == ["search"]

    # Get
    response = await client.get(f"/api/v1/policies/{tenant_id}")
    assert response.status_code == 200
    config = response.json()["config"]
    assert config["autonomy_profile"] == "strict"
    assert config["tool_policy"]["allowed_tools"] == ["search"]

    # Update (Merge)
    payload2 = {"config": {"budget_policy": {"max_total_tokens": 100000}}}
    response = await client.put(f"/api/v1/policies/{tenant_id}", json=payload2)
    data = response.json()
    assert data["config"]["autonomy_profile"] == "strict"  # Preserved
    assert data["config"]["budget_policy"]["max_total_tokens"] == 100000  # Added
