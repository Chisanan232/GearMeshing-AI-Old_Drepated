import pytest
from httpx import AsyncClient

# Mark all tests in this module as async
pytestmark = pytest.mark.asyncio


async def test_health_check(client: AsyncClient):
    response = await client.get("http://localhost/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_version(client: AsyncClient):
    response = await client.get("http://localhost/version")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "schema_version" in data
