import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio

async def test_list_roles(client: AsyncClient):
    response = await client.get("/api/v1/roles/")
    assert response.status_code == 200
    roles = response.json()
    assert isinstance(roles, list)
    assert "planner" in roles
    assert "dev" in roles
