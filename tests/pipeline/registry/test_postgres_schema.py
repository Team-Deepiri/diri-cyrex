"""Tests for Track A — PostgreSQL schema in ``cyrex`` on postgres-cyrex.

Verifies that ``PostgresArtifactStore.ensure_schema()`` creates the expected
tables with key columns and indexes via ``information_schema`` / ``pg_indexes``.
"""

from __future__ import annotations

import pytest

from app.pipeline.registry.postgres_store import PostgresArtifactStore


async def _column_names(pg_manager, table: str) -> set[str]:
    rows = await pg_manager.fetch(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'cyrex' AND table_name = $1
        """,
        table,
    )
    return {row["column_name"] for row in rows}


async def _index_names(pg_manager) -> set[str]:
    rows = await pg_manager.fetch(
        """
        SELECT indexname
        FROM pg_indexes
        WHERE schemaname = 'cyrex'
        """
    )
    return {row["indexname"] for row in rows}


@pytest.mark.asyncio()
async def test_schema_tables_exist(pg_manager):
    """All four cyrex tables exist after ensure_schema()."""
    store = PostgresArtifactStore(postgres=pg_manager)
    await store.ensure_schema()

    rows = await pg_manager.fetch(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'cyrex'
          AND table_type = 'BASE TABLE'
        """
    )
    tables = {row["table_name"] for row in rows}
    assert tables >= {
        "artifacts",
        "artifact_refs",
        "citations",
        "learning_artifacts",
    }


@pytest.mark.asyncio()
async def test_columns_artifacts(pg_manager):
    """``cyrex.artifacts`` has expected key columns."""
    store = PostgresArtifactStore(postgres=pg_manager)
    await store.ensure_schema()

    columns = await _column_names(pg_manager, "artifacts")
    expected = {
        "artifact_id",
        "document_id",
        "version",
        "artifact_type",
        "source_doc_hash",
        "confidence",
        "payload_json",
        "provenance_json",
        "is_deleted",
        "created_at",
    }
    assert expected <= columns, f"Missing columns: {expected - columns}"


@pytest.mark.asyncio()
async def test_columns_artifact_refs(pg_manager):
    """``cyrex.artifact_refs`` has expected key columns."""
    store = PostgresArtifactStore(postgres=pg_manager)
    await store.ensure_schema()

    columns = await _column_names(pg_manager, "artifact_refs")
    expected = {
        "from_artifact",
        "to_artifact",
        "ref_type",
        "created_at",
    }
    assert expected <= columns, f"Missing columns: {expected - columns}"


@pytest.mark.asyncio()
async def test_columns_citations(pg_manager):
    """``cyrex.citations`` has all expected locator and metadata columns."""
    store = PostgresArtifactStore(postgres=pg_manager)
    await store.ensure_schema()

    columns = await _column_names(pg_manager, "citations")
    expected = {
        "citation_id",
        "artifact_id",
        "document_id",
        "source_doc_hash",
        "locator_type",
        "char_start",
        "char_end",
        "page_start",
        "page_end",
        "element_id",
        "quote",
        "confidence",
        "extraction_pass",
    }
    assert expected <= columns, f"Missing columns: {expected - columns}"


@pytest.mark.asyncio()
async def test_columns_learning_artifacts(pg_manager):
    """``cyrex.learning_artifacts`` has correction/export columns."""
    store = PostgresArtifactStore(postgres=pg_manager)
    await store.ensure_schema()

    columns = await _column_names(pg_manager, "learning_artifacts")
    expected = {
        "artifact_id",
        "document_id",
        "field_name",
        "original_value",
        "corrected_value",
        "citation_json",
        "actor_id",
        "timestamp",
        "exported",
    }
    assert expected <= columns, f"Missing columns: {expected - columns}"


@pytest.mark.asyncio()
async def test_indexes_exist(pg_manager):
    """Indexes on refs and citations composite span exist."""
    store = PostgresArtifactStore(postgres=pg_manager)
    await store.ensure_schema()

    indexes = await _index_names(pg_manager)
    assert "idx_cyrex_refs_from" in indexes
    assert "idx_cyrex_refs_to" in indexes
    assert "idx_cyrex_citations_doc_span" in indexes


@pytest.mark.asyncio()
async def test_idempotent_ensure_schema(pg_manager):
    """Calling ``ensure_schema()`` twice is a safe no-op."""
    store = PostgresArtifactStore(postgres=pg_manager)
    await store.ensure_schema()

    tables_first = await pg_manager.fetch(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'cyrex' AND table_type = 'BASE TABLE'
        """
    )
    indexes_first = await _index_names(pg_manager)

    await store.ensure_schema()

    tables_second = await pg_manager.fetch(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'cyrex' AND table_type = 'BASE TABLE'
        """
    )
    indexes_second = await _index_names(pg_manager)

    assert {r["table_name"] for r in tables_first} == {r["table_name"] for r in tables_second}
    assert indexes_first == indexes_second
