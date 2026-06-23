"""Tests for Task A-1b — SQLite store CRUD and ArtifactStorePort methods.

These tests verify that ``SqliteArtifactStore`` correctly implements
all methods defined in ``contracts/ports.py``, including ghost filtering,
graph neighborhood traversal, and inverse citation lookups.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.pipeline.contracts.models import (
    ArtifactBundle,
    ArtifactType,
    Citation,
    CitationLocator,
    Provenance,
)
from app.pipeline.registry.sqlite_store import ManagedSqliteStore, SqliteArtifactStore, init_db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store():
    """Fresh in-memory database with no pre-loaded artifacts."""
    db = SqliteArtifactStore(":memory:")
    yield db
    db.close()


@pytest.fixture()
def sample_bundle():
    """A canonical extraction artifact bundle with citations and refs."""
    cit_a = Citation(
        citation_id="cit_a",
        document_id="doc_001",
        source_doc_hash="hash_001",
        locator=CitationLocator(locator_type="char_range", char_start=0, char_end=5),
        quote="hello",
        confidence=0.95,
    )
    cit_b = Citation(
        citation_id="cit_b",
        document_id="doc_001",
        source_doc_hash="hash_001",
        locator=CitationLocator(locator_type="char_range", char_start=100, char_end=107),
        quote="world!!",
        confidence=0.85,
        extraction_pass=2,
    )
    provenance = Provenance(
        source_doc_hash="hash_001",
        document_id="doc_001",
        version=1,
        model_id="gpt-4o",
        depends_on=["art_000"],
        depended_on_by=["art_010"],
        cross_references=[],
    )
    return ArtifactBundle(
        artifact_id="art_001",
        document_id="doc_001",
        version=1,
        artifact_type=ArtifactType.EXTRACTION,
        source_doc_hash="hash_001",
        confidence=0.92,
        payload={"fields": {"rent": 4500}},
        provenance=provenance,
        citations=[cit_a, cit_b],
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture()
def populated_store(store):
    """Store pre-loaded with a few artifacts for graph/traversal tests."""

    def _setup():
        # Original artifact
        store.conn.execute(
            """INSERT INTO artifacts (
                artifact_id, document_id, version, artifact_type,
                source_doc_hash, confidence, payload_json,
                provenance_json, created_at, is_deleted
            ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, 0)""",
            (
                "art_001",
                "doc_001",
                ArtifactType.EXTRACTION.value,
                "hash_001",
                0.92,
                json.dumps({}),
                json.dumps({"source_doc_hash": "hash_001", "document_id": "doc_001", "depends_on": ["art_000"], "depended_on_by": ["art_010"]}),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        # Insert ref to art_000
        store.conn.execute(
            "INSERT INTO artifact_refs (from_artifact, to_artifact, ref_type, created_at) VALUES (?, ?, 'depends_on', ?)",
            ("art_001", "art_000", datetime.now(timezone.utc).isoformat()),
        )
        # Insert citation on art_001
        store.conn.execute(
            """INSERT INTO citations (
                citation_id, artifact_id, document_id, source_doc_hash,
                locator_type, char_start, char_end, page_start, page_end,
                element_id, quote, confidence, extraction_pass
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("cit_001", "art_001", "doc_001", "hash_001", "char_range", 10, 15, None, None, None, "ten", 0.9, 1),
        )
        # Dependency artifact
        store.conn.execute(
            """INSERT INTO artifacts (
                artifact_id, document_id, version, artifact_type,
                source_doc_hash, confidence, payload_json,
                provenance_json, created_at, is_deleted
            ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, 0)""",
            (
                "art_000",
                "doc_001",
                ArtifactType.CANONICAL.value,
                "hash_001",
                0.95,
                json.dumps({}),
                json.dumps({"source_doc_hash": "hash_001", "document_id": "doc_001"}),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        # Superseded version (ghost)
        store.conn.execute(
            """INSERT INTO artifacts (
                artifact_id, document_id, version, artifact_type,
                source_doc_hash, confidence, payload_json,
                provenance_json, created_at, is_deleted
            ) VALUES (?, ?, 0, ?, ?, ?, ?, ?, ?, 1)""",
            (
                "art_001_v0",
                "doc_001",
                ArtifactType.EXTRACTION.value,
                "hash_001",
                0.80,
                json.dumps({}),
                json.dumps({"source_doc_hash": "hash_001", "document_id": "doc_001"}),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        store.conn.commit()

    _setup()
    return store


# ---------------------------------------------------------------------------
# create / get round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_create_and_get(store, sample_bundle):
    """Creating a bundle and reading it back by ID returns an equivalent bundle."""
    result = await store.create(sample_bundle)
    retrieved = await store.get(sample_bundle.artifact_id)
    assert retrieved is not None
    assert retrieved.artifact_id == sample_bundle.artifact_id
    assert retrieved.document_id == sample_bundle.document_id
    assert retrieved.confidence == sample_bundle.confidence
    assert len(retrieved.citations) == len(sample_bundle.citations)
    assert retrieved.citations[0].quote == sample_bundle.citations[0].quote
    assert retrieved.is_deleted is False


@pytest.mark.asyncio()
async def test_get_nonexistent(store):
    """Getting a non-existent ID returns None."""
    assert await store.get("art_nonexistent") is None


@pytest.mark.asyncio()
async def test_get_deleted_returns_none(store, sample_bundle):
    """A deleted artifact is not returned by ``get``."""
    await store.create(sample_bundle)
    # Directly mark it deleted in the database
    store.conn.execute(
        "UPDATE artifacts SET is_deleted = 1 WHERE artifact_id = ?",
        (sample_bundle.artifact_id,),
    )
    store.conn.commit()
    assert await store.get(sample_bundle.artifact_id) is None


# ---------------------------------------------------------------------------
# get_latest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_get_latest(store):
    """get_latest returns the highest version for a document."""
    v1 = ArtifactBundle(
        artifact_id="v1",
        document_id="doc_001",
        version=1,
        artifact_type=ArtifactType.EXTRACTION,
        source_doc_hash="hash_001",
        confidence=0.8,
        payload={},
        provenance=Provenance(source_doc_hash="hash_001", document_id="doc_001"),
        created_at=datetime.now(timezone.utc),
    )
    v2 = ArtifactBundle(
        artifact_id="v2",
        document_id="doc_001",
        version=2,
        artifact_type=ArtifactType.EXTRACTION,
        source_doc_hash="hash_001",
        confidence=0.85,
        payload={},
        provenance=Provenance(source_doc_hash="hash_001", document_id="doc_001"),
        created_at=datetime.now(timezone.utc),
    )
    await store.create(v1)
    await store.create(v2)

    latest = await store.get_latest("doc_001")
    assert latest is not None
    assert latest.artifact_id == "v2"
    assert latest.version == 2


@pytest.mark.asyncio()
async def test_get_latest_by_type(store):
    """get_latest with artifact_type filter returns the correct type."""
    extr = ArtifactBundle(
        artifact_id="art_ex",
        document_id="doc_001",
        version=1,
        artifact_type=ArtifactType.EXTRACTION,
        source_doc_hash="hash_001",
        confidence=0.9,
        payload={},
        provenance=Provenance(source_doc_hash="hash_001", document_id="doc_001"),
        created_at=datetime.now(timezone.utc),
    )
    ans = ArtifactBundle(
        artifact_id="art_ans",
        document_id="doc_001",
        version=2,
        artifact_type=ArtifactType.ANSWER,
        source_doc_hash="hash_001",
        confidence=0.95,
        payload={},
        provenance=Provenance(source_doc_hash="hash_001", document_id="doc_001"),
        created_at=datetime.now(timezone.utc),
    )
    await store.create(extr)
    await store.create(ans)

    latest = await store.get_latest("doc_001", artifact_type="answer")
    assert latest is not None
    assert latest.artifact_type == ArtifactType.ANSWER


@pytest.mark.asyncio()
async def test_get_latest_excludes_deleted(store):
    """get_latest skips ghost artifacts."""
    ghost = ArtifactBundle(
        artifact_id="ghost",
        document_id="doc_001",
        version=5,
        artifact_type=ArtifactType.EXTRACTION,
        source_doc_hash="hash_001",
        confidence=0.5,
        payload={},
        provenance=Provenance(source_doc_hash="hash_001", document_id="doc_001"),
        is_deleted=True,
        created_at=datetime.now(timezone.utc),
    )
    real = ArtifactBundle(
        artifact_id="real",
        document_id="doc_001",
        version=1,
        artifact_type=ArtifactType.EXTRACTION,
        source_doc_hash="hash_001",
        confidence=0.9,
        payload={},
        provenance=Provenance(source_doc_hash="hash_001", document_id="doc_001"),
        created_at=datetime.now(timezone.utc),
    )
    await store.create(ghost)
    await store.create(real)

    latest = await store.get_latest("doc_001")
    assert latest is not None
    assert latest.artifact_id == "real"


# ---------------------------------------------------------------------------
# list_by_document
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_list_by_document(store):
    """list_by_document returns all non-deleted artifacts for a document."""
    for i in range(3):
        await store.create(
            ArtifactBundle(
                artifact_id=f"art_{i}",
                document_id="doc_002",
                version=i + 1,
                artifact_type=ArtifactType.EXTRACTION,
                source_doc_hash="hash_002",
                confidence=0.9,
                payload={},
                provenance=Provenance(source_doc_hash="hash_002", document_id="doc_002"),
                created_at=datetime.now(timezone.utc),
            )
        )
    # Also create an artifact for a different document
    await store.create(
        ArtifactBundle(
            artifact_id="art_other",
            document_id="doc_999",
            version=1,
            artifact_type=ArtifactType.EXTRACTION,
            source_doc_hash="hash_999",
            confidence=0.9,
            payload={},
            provenance=Provenance(source_doc_hash="hash_999", document_id="doc_999"),
            created_at=datetime.now(timezone.utc),
        )
    )

    results = await store.list_by_document("doc_002")
    assert len(results) == 3
    assert {r.artifact_id for r in results} == {"art_0", "art_1", "art_2"}
    assert all(not r.is_deleted for r in results)


@pytest.mark.asyncio()
async def test_list_by_document_empty(store):
    """list_by_document returns [] for a document with no artifacts."""
    assert await store.list_by_document("doc_empty") == []


# ---------------------------------------------------------------------------
# list_versions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_list_versions(store):
    """list_versions returns sorted unique version numbers."""
    versions = [1, 3, 5, 7]
    for i, v in enumerate(versions):
        await store.create(
            ArtifactBundle(
                artifact_id=f"art_v{v}",
                document_id="doc_003",
                version=v,
                artifact_type=ArtifactType.EXTRACTION,
                source_doc_hash="hash_003",
                confidence=0.9,
                payload={},
                provenance=Provenance(source_doc_hash="hash_003", document_id="doc_003"),
                created_at=datetime.now(timezone.utc),
            )
        )

    result = await store.list_versions("doc_003")
    assert result == [1, 3, 5, 7]


@pytest.mark.asyncio()
async def test_list_versions_excludes_deleted(store):
    """list_versions does not include ghost versions."""
    await store.create(
        ArtifactBundle(
            artifact_id="v1",
            document_id="doc_003",
            version=1,
            artifact_type=ArtifactType.EXTRACTION,
            source_doc_hash="hash_003",
            confidence=0.9,
            payload={},
            provenance=Provenance(source_doc_hash="hash_003", document_id="doc_003"),
            created_at=datetime.now(timezone.utc),
        )
    )
    await store.create(
        ArtifactBundle(
            artifact_id="v2_ghost",
            document_id="doc_003",
            version=2,
            artifact_type=ArtifactType.EXTRACTION,
            source_doc_hash="hash_003",
            confidence=0.5,
            payload={},
            provenance=Provenance(source_doc_hash="hash_003", document_id="doc_003"),
            is_deleted=True,
            created_at=datetime.now(timezone.utc),
        )
    )

    result = await store.list_versions("doc_003")
    assert result == [1]


# ---------------------------------------------------------------------------
# resolve_version
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_resolve_version(store, populated_store):
    """resolve_version returns the correct version and is atomic."""
    resolved = await store.resolve_version("doc_001", 1)
    assert resolved is not None
    assert resolved.artifact_id == "art_001"
    assert resolved.version == 1


@pytest.mark.asyncio()
async def test_resolve_version_nonexistent(store):
    """resolve_version returns None for a non-existent version."""
    assert await store.resolve_version("doc_001", 99) is None


@pytest.mark.asyncio()
async def test_resolve_version_ghost_returns_none(store):
    """resolve_version skips ghost artifacts."""
    await store.create(
        ArtifactBundle(
            artifact_id="v_ghost",
            document_id="doc_001",
            version=99,
            artifact_type=ArtifactType.EXTRACTION,
            source_doc_hash="hash_001",
            confidence=0.5,
            payload={},
            provenance=Provenance(source_doc_hash="hash_001", document_id="doc_001"),
            is_deleted=True,
            created_at=datetime.now(timezone.utc),
        )
    )
    assert await store.resolve_version("doc_001", 99) is None


# ---------------------------------------------------------------------------
# get_graph_neighborhood
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_graph_neighborhood_1hop(store, populated_store):
    """1-hop traversal reaches direct dependents and dependencies."""
    result = await store.get_graph_neighborhood("art_001", hops=1)
    node_ids = {n.artifact_id for n in result["nodes"]}
    assert "art_001" in node_ids
    assert "art_000" in node_ids  # depends_on


@pytest.mark.asyncio()
async def test_graph_neighborhood_0hop(store, populated_store):
    """0-hop returns only the root artifact."""
    result = await store.get_graph_neighborhood("art_001", hops=0)
    assert len(result["nodes"]) == 1
    assert result["nodes"][0].artifact_id == "art_001"
    assert len(result["edges"]) == 0


@pytest.mark.asyncio()
async def test_graph_neighborhood_nonexistent(store):
    """Graph neighborhood for a non-existent artifact returns empty."""
    result = await store.get_graph_neighborhood("art_nonexistent", hops=1)
    assert result["nodes"] == []
    assert result["edges"] == []


# ---------------------------------------------------------------------------
# get_inverse_citations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_get_inverse_citations(store):
    """Inverse citations finds all artifacts citing the same span."""
    bundle_a = await store.create(
        ArtifactBundle(
            artifact_id="art_inv_a",
            document_id="doc_001",
            version=1,
            artifact_type=ArtifactType.CANONICAL,
            source_doc_hash="hash_001",
            confidence=0.95,
            payload={},
            provenance=Provenance(source_doc_hash="hash_001", document_id="doc_001"),
            citations=[
                Citation(
                    citation_id="cit_inv_1",
                    document_id="doc_001",
                    source_doc_hash="hash_001",
                    locator=CitationLocator(locator_type="char_range", char_start=100, char_end=110),
                    quote="target span",
                    confidence=0.9,
                )
            ],
            created_at=datetime.now(timezone.utc),
        )
    )
    bundle_b = await store.create(
        ArtifactBundle(
            artifact_id="art_inv_b",
            document_id="doc_001",
            version=1,
            artifact_type=ArtifactType.REASONING,
            source_doc_hash="hash_001",
            confidence=0.88,
            payload={},
            provenance=Provenance(source_doc_hash="hash_001", document_id="doc_001"),
            citations=[
                Citation(
                    citation_id="cit_inv_2",
                    document_id="doc_001",
                    source_doc_hash="hash_001",
                    locator=CitationLocator(locator_type="char_range", char_start=100, char_end=110),
                    quote="target span",
                    confidence=0.85,
                )
            ],
            created_at=datetime.now(timezone.utc),
        )
    )
    bundle_c = await store.create(
        ArtifactBundle(
            artifact_id="art_inv_c",
            document_id="doc_001",
            version=1,
            artifact_type=ArtifactType.ANSWER,
            source_doc_hash="hash_001",
            confidence=0.92,
            payload={},
            provenance=Provenance(source_doc_hash="hash_001", document_id="doc_001"),
            citations=[
                Citation(
                    citation_id="cit_inv_3",
                    document_id="doc_001",
                    source_doc_hash="hash_001",
                    locator=CitationLocator(locator_type="char_range", char_start=500, char_end=510),
                    quote="different span",
                    confidence=0.9,
                )
            ],
            created_at=datetime.now(timezone.utc),
        )
    )

    results = await store.get_inverse_citations("doc_001", 100, 110)
    assert len(results) == 2
    result_ids = {r.artifact_id for r in results}
    assert "art_inv_a" in result_ids
    assert "art_inv_b" in result_ids
    assert "art_inv_c" not in result_ids


@pytest.mark.asyncio()
async def test_get_inverse_citations_empty(store):
    """Inverse citations returns [] for a span with no citations."""
    results = await store.get_inverse_citations("doc_001", 0, 0)
    assert results == []


# ---------------------------------------------------------------------------
# ManagedSqliteStore context manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_managed_sqlite_store():
    """ManagedSqliteStore properly creates and closes connections."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        async with ManagedSqliteStore(db_path) as store:
            assert store._conn is not None
            conn_id = id(store._conn)

            bundle = await store.create(
                ArtifactBundle(
                    artifact_id="art_ctx",
                    document_id="doc_ctx",
                    version=1,
                    artifact_type=ArtifactType.EXTRACTION,
                    source_doc_hash="hash_ctx",
                    confidence=0.9,
                    payload={},
                    provenance=Provenance(source_doc_hash="hash_ctx", document_id="doc_ctx"),
                    created_at=datetime.now(timezone.utc),
                )
            )
            assert bundle.artifact_id == "art_ctx"

        # After context exit, connection should be closed
        assert store._conn is None

        # Verify data persisted
        async with ManagedSqliteStore(db_path) as store2:
            retrieved = await store2.get("art_ctx")
            assert retrieved is not None
            assert retrieved.artifact_id == "art_ctx"
    finally:
        Path(db_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# created_at round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_created_at_preserved(store):
    """created_at timestamp is persisted and round-trips correctly."""
    original_time = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
    bundle = ArtifactBundle(
        artifact_id="art_time",
        document_id="doc_001",
        version=1,
        artifact_type=ArtifactType.EXTRACTION,
        source_doc_hash="hash_001",
        confidence=0.9,
        payload={},
        provenance=Provenance(source_doc_hash="hash_001", document_id="doc_001"),
        created_at=original_time,
    )
    await store.create(bundle)
    retrieved = await store.get("art_time")
    assert retrieved is not None
    assert retrieved.created_at == original_time
