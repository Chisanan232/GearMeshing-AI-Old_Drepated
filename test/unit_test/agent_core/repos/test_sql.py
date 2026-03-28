from __future__ import annotations

import pytest

from gearmeshing_ai.core.database import create_engine


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "db_url",
    [
        "postgres://user:pass@localhost:5432/db",
        "postgresql://user:pass@localhost:5432/db",
        "postgresql+psycopg2://user:pass@localhost:5432/db",
        "postgresql+psycopg://user:pass@localhost:5432/db",
        "postgresql+asyncpg://user:pass@localhost:5432/db",
    ],
)
async def test_create_engine_normalizes_postgres_url_schemes(db_url: str) -> None:
    engine = create_engine(db_url)
    try:
        assert engine.url.drivername == "postgresql+asyncpg"
    finally:
        await engine.dispose()
