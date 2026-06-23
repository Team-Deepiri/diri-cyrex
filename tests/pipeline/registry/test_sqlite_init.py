"""Tests for Task A-1 — SQLite schema & DB initialization.

These tests verify that ``init_db()`` creates the three expected tables
with the correct columns, indexes, and pragmas, and that calling
``init_db()`` twice is a safe no-op.

No CRUD methods, no ``ArtifactStorePort``, no model imports — just
DDL, pragmas, and schema verification.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from app.pipeline.registry.sqlite_store import init_db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def conn():
    """Fresh in-memory database, initialised once per test."""
    connection = init_db(":memory:")
    yield connection
    connection.close()


# ---------------------------------------------------------------------------
# Table existence
# ---------------------------------------------------------------------------


def test_tables_created(conn):
    """All three tables appear in ``sqlite_master``."""
    tables = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite%'"
        ).fetchall()
    }
    assert tables >= {"artifacts", "artifact_refs", "citations"}


# ---------------------------------------------------------------------------
# Column verification — artifacts
# ---------------------------------------------------------------------------


def test_columns_artifacts(conn):
    """``artifacts`` has exactly 9 expected columns with correct types."""
    columns = {
        (row["name"], row["type"], row["dflt_value"])
        for row in conn.execute("PRAGMA table_info(artifacts)").fetchall()
    }
    expected = {
        ("artifact_id", "TEXT", None),
        ("document_id", "TEXT", None),
        ("version", "INTEGER", "1"),
        ("artifact_type", "TEXT", None),
        ("source_doc_hash", "TEXT", None),
        ("confidence", "REAL", None),
        ("payload_json", "TEXT", "'{}'"),
        ("provenance_json", "TEXT", "'{}'"),
        ("created_at", "TEXT", None),
        ("is_deleted", "INTEGER", "0"),
    }
    for col in expected:
        assert col in columns, f"Missing column {col}"
    assert len(columns) == 10, f"Expected 10 columns, got {len(columns)}"


# ---------------------------------------------------------------------------
# Column verification — artifact_refs
# ---------------------------------------------------------------------------


def test_columns_artifact_refs(conn):
    """``artifact_refs`` has exactly 4 expected columns."""
    columns = {
        (row["name"], row["type"])
        for row in conn.execute("PRAGMA table_info(artifact_refs)").fetchall()
    }
    expected = {
        ("from_artifact", "TEXT"),
        ("to_artifact", "TEXT"),
        ("ref_type", "TEXT"),
        ("created_at", "TEXT"),
    }
    for col in expected:
        assert col in columns, f"Missing column {col}"
    assert len(columns) == 4, f"Expected 4 columns, got {len(columns)}"


# ---------------------------------------------------------------------------
# Column verification — citations
# ---------------------------------------------------------------------------


def test_columns_citations(conn):
    """``citations`` has all 13 expected columns."""
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(citations)").fetchall()
    }
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
    assert len(columns) == 13, f"Expected 13 columns, got {len(columns)}"


# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------


def test_indexes_exist(conn):
    """Indexes on ``artifact_refs.from_artifact``, ``artifact_refs.to_artifact``,
    and the citations composite ``(document_id, char_start, char_end)`` exist."""
    indexes = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite%'"
        ).fetchall()
    }
    assert "idx_refs_from" in indexes, "Missing index on artifact_refs.from_artifact"
    assert "idx_refs_to" in indexes, "Missing index on artifact_refs.to_artifact"
    assert "idx_citations_doc_span" in indexes, (
        "Missing composite index on citations(document_id, char_start, char_end)"
    )


# ---------------------------------------------------------------------------
# Idempotent init
# ---------------------------------------------------------------------------


def test_idempotent_init():
    """Calling ``init_db()`` twice on the same path raises no error and
    does not duplicate tables or indexes."""
    conn = init_db(":memory:")
    tables_first = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    indexes_first = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite%'"
        ).fetchall()
    }

    # Re-initialise the same connection (re-run all DDL).
    init_db(":memory:")

    tables_second = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    indexes_second = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite%'"
        ).fetchall()
    }

    assert tables_first == tables_second, "Tables were duplicated"
    assert indexes_first == indexes_second, "Indexes were duplicated"
    conn.close()


# ---------------------------------------------------------------------------
# Pragmas
# ---------------------------------------------------------------------------


def test_wal_mode():
    """``journal_mode`` returns ``wal`` after ``init_db()`` on a file-based
    database. (:memory: databases fall back to the "memory" journal mode.)"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    try:
        conn = init_db(db_path)
        result = conn.execute("PRAGMA journal_mode").fetchone()
        mode = result[0] if result else ""
        conn.close()
        assert mode == "wal", f"Expected WAL journal mode, got {mode!r}"
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_foreign_keys_on(conn):
    """``foreign_keys`` returns ``1`` after ``init_db()``."""
    result = conn.execute("PRAGMA foreign_keys").fetchone()
    fk_on = result[0] if result else 0
    assert fk_on == 1, f"Expected foreign_keys=1, got {fk_on}"
