"""Shared fixtures for live postgres-cyrex Track A registry tests."""
from __future__ import annotations

import asyncio
import os

import pytest
import pytest_asyncio

from app.database.postgres import PostgreSQLManager
from app.pipeline.registry.postgres_store import PostgresArtifactStore

os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "5434")
os.environ.setdefault("POSTGRES_USER", "deepiri_cyrex")
os.environ.setdefault("POSTGRES_PASSWORD", "deepiripassword")
os.environ.setdefault("POSTGRES_DB", "cyrex_db")
os.environ.setdefault(
    "JWT_SECRET", "test-jwt-secret-minimum-32-characters-long-for-testing"
)


def _pg_manager_kwargs() -> dict:
    return {
        "host": os.environ["POSTGRES_HOST"],
        "port": int(os.environ["POSTGRES_PORT"]),
        "database": os.environ["POSTGRES_DB"],
        "user": os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
        "min_size": 1,
        "max_size": 2,
    }


def _postgres_reachable() -> bool:
    async def _probe() -> bool:
        mgr = PostgreSQLManager(**_pg_manager_kwargs())
        try:
            return await mgr.initialize(max_retries=1, retry_delay=0.1)
        finally:
            await mgr.close()

    return asyncio.run(_probe())


_POSTGRES_REACHABLE = _postgres_reachable()

pytestmark = pytest.mark.skipif(
    not _POSTGRES_REACHABLE,
    reason="postgres-cyrex not reachable at 127.0.0.1:5434",
)


@pytest_asyncio.fixture()
async def pg_manager():
    mgr = PostgreSQLManager(**_pg_manager_kwargs())
    ok = await mgr.initialize(max_retries=3, retry_delay=0.5)
    if not ok:
        pytest.skip("postgres-cyrex connection failed")
    yield mgr
    await mgr.close()


@pytest_asyncio.fixture()
async def store(pg_manager):
    artifact_store = PostgresArtifactStore(postgres=pg_manager)
    await artifact_store.ensure_schema()
    db = await artifact_store._db()
    await db.execute(
        "TRUNCATE cyrex.citations, cyrex.artifact_refs, "
        "cyrex.artifacts, cyrex.learning_artifacts CASCADE"
    )
    yield artifact_store
