"""SQLite-backed implementation of the Cyrex Artifact Store.

This module provides:
- ``init_db(db_path)`` — idempotent schema initialization with pragmas
- ``SqliteArtifactStore`` — ``ArtifactStorePort`` implementation backed by
  a three-table SQLite schema (artifacts, artifact_refs, citations)

The schema is **idempotent**: calling ``init_db()`` on an already-initialised
database is a safe no-op.  WAL mode and foreign-key enforcement are set on
every new connection.

Tracks A–D import ``contracts/models.py`` and ``contracts/ports.py`` — this
module imports from contracts to re-use model definitions but never
duplicates them.

Usage::

    from app.pipeline.registry.sqlite_store import SqliteArtifactStore

    store = SqliteArtifactStore("cyrex.db")
    await store.init_db()  # safe to call at startup every time
    bundle = await store.create(my_bundle)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_ARTIFACTS = """\
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id      TEXT PRIMARY KEY,
    document_id      TEXT    NOT NULL,
    version          INTEGER NOT NULL DEFAULT 1,
    artifact_type    TEXT    NOT NULL,
    source_doc_hash  TEXT    NOT NULL,
    confidence       REAL    NOT NULL,
    payload_json     TEXT    NOT NULL DEFAULT '{}',
    provenance_json  TEXT    NOT NULL DEFAULT '{}',
    is_deleted       INTEGER NOT NULL DEFAULT 0
);
"""

_CREATE_ARTIFACT_REFS = """\
CREATE TABLE IF NOT EXISTS artifact_refs (
    from_artifact TEXT NOT NULL,
    to_artifact   TEXT NOT NULL,
    ref_type      TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (from_artifact, to_artifact, ref_type)
);
"""

_CREATE_CITATIONS = """\
CREATE TABLE IF NOT EXISTS citations (
    citation_id      TEXT PRIMARY KEY,
    artifact_id      TEXT NOT NULL,
    document_id      TEXT NOT NULL,
    source_doc_hash  TEXT NOT NULL,
    locator_type     TEXT NOT NULL,
    char_start       INTEGER,
    char_end         INTEGER,
    page_start       INTEGER,
    page_end         INTEGER,
    element_id       TEXT,
    quote            TEXT NOT NULL,
    confidence       REAL NOT NULL,
    extraction_pass  INTEGER,
    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id)
        ON DELETE CASCADE
);
"""

# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------

_INDEX_REF_FROM = "CREATE INDEX IF NOT EXISTS idx_refs_from ON artifact_refs(from_artifact);"
_INDEX_REF_TO   = "CREATE INDEX IF NOT EXISTS idx_refs_to   ON artifact_refs(to_artifact);"

_INDEX_CITATIONS_DOC_SPAN = (
    "CREATE INDEX IF NOT EXISTS idx_citations_doc_span "
    "ON citations(document_id, char_start, char_end);"
)

_ALL_DDL = [
    _CREATE_ARTIFACTS,
    _CREATE_ARTIFACT_REFS,
    _CREATE_CITATIONS,
    _INDEX_REF_FROM,
    _INDEX_REF_TO,
    _INDEX_CITATIONS_DOC_SPAN,
]


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------


def init_db(db_path: str) -> sqlite3.Connection:
    """Open (or create) a SQLite database and run DDL idempotently.

    Sets **WAL** journal mode and enables **foreign keys** on every
    new connection.  Calling this function twice on the same path is
    a safe no-op — ``CREATE TABLE IF NOT EXISTS`` ensures the schema
    is never wiped or duplicated.

    Args:
        db_path: Path to the SQLite database file.  Use ``":memory:"``
                 for ephemeral test databases.

    Returns:
        A ``sqlite3.Connection`` with pragmas applied and DDL executed.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # enable dict-like column access
    conn.execute("PRAGMA foreign_keys = ON;")

    # WAL mode must be set *before* the first write transaction.
    # ``PRAGMA journal_mode=WAL`` returns the *previous* mode, so we
    # check the return value to confirm the change took effect.
    result = conn.execute("PRAGMA journal_mode=WAL;").fetchone()
    mode = result[0] if result else "unknown"
    if mode != "wal":
        logger.warning("Failed to set WAL journal mode — got %r", mode)

    for stmt in _ALL_DDL:
        conn.execute(stmt)

    conn.commit()

    logger.info("Database initialised at %s (journal_mode=%s)", db_path, mode)
    return conn


# ---------------------------------------------------------------------------
# SqliteArtifactStore
# ---------------------------------------------------------------------------


class SqliteArtifactStore:
    """SQLite-backed ``ArtifactStorePort`` implementation.

    Three-table schema:
    * **artifacts** — primary artifact table (bundle data)
    * **artifact_refs** — bidirectional typed edge table
    * **citations** — flat citation index for fast location queries

    The constructor opens a connection (lazy on first use).  Call
    ``init_db()`` at application startup — it is safe to call
    repeatedly.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        """Return a connection, initializing the schema on first access."""
        if self._conn is None:
            self._conn = init_db(self.db_path)
        return self._conn

    def close(self) -> None:
        """Close the underlying connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # CRUD helpers (used by the Protocol methods below)
    # ------------------------------------------------------------------

    def _bundle_to_row(self, bundle) -> Dict[str, Any]:
        """Serialize a ``ArtifactBundle`` into a dict suitable for INSERT."""
        return {
            "artifact_id": bundle.artifact_id,
            "document_id": bundle.document_id,
            "version": bundle.version,
            "artifact_type": bundle.artifact_type.value,
            "source_doc_hash": bundle.source_doc_hash,
            "confidence": bundle.confidence,
            "payload_json": json.dumps(bundle.payload),
            "provenance_json": json.dumps(bundle.provenance.model_dump()),
            "is_deleted": int(bundle.is_deleted),
        }

    def _row_to_bundle(self, row: Dict[str, Any]) -> Any:
        """Deserialize a database row back into an ``ArtifactBundle``."""
        from app.pipeline.contracts.models import ArtifactBundle, Provenance
        from app.pipeline.contracts.models import ArtifactType

        payload = json.loads(row["payload_json"])
        provenance_dict = json.loads(row["provenance_json"])
        provenance = Provenance.model_validate(provenance_dict)

        return ArtifactBundle(
            artifact_id=row["artifact_id"],
            document_id=row["document_id"],
            version=row["version"],
            artifact_type=ArtifactType(row["artifact_type"]),
            source_doc_hash=row["source_doc_hash"],
            confidence=row["confidence"],
            payload=payload,
            provenance=provenance,
            is_deleted=bool(row["is_deleted"]),
            created_at=datetime.fromisoformat(row["created_at"])
            if row.get("created_at")
            else datetime.now(timezone.utc),
        )

    def _insert_refs(self, bundle) -> None:
        """Persist provenance edges into ``artifact_refs``."""
        conn = self.conn
        now = datetime.now(timezone.utc).isoformat()

        for ref_id in bundle.provenance.depends_on:
            conn.execute(
                "INSERT OR IGNORE INTO artifact_refs (from_artifact, to_artifact, ref_type, created_at) "
                "VALUES (?, ?, 'depends_on', ?)",
                (bundle.artifact_id, ref_id, now),
            )
        for ref_id in bundle.provenance.depended_on_by:
            conn.execute(
                "INSERT OR IGNORE INTO artifact_refs (from_artifact, to_artifact, ref_type, created_at) "
                "VALUES (?, ?, 'depended_on_by', ?)",
                (bundle.artifact_id, ref_id, now),
            )
        for ref_id in bundle.provenance.cross_references:
            conn.execute(
                "INSERT OR IGNORE INTO artifact_refs (from_artifact, to_artifact, ref_type, created_at) "
                "VALUES (?, ?, 'cross_reference', ?)",
                (bundle.artifact_id, ref_id, now),
            )

    def _insert_citations(self, bundle) -> None:
        """Persist citations into the flat ``citations`` table."""
        conn = self.conn
        for cit in bundle.citations:
            loc = cit.locator
            conn.execute(
                """INSERT OR IGNORE INTO citations (
                       citation_id, artifact_id, document_id, source_doc_hash,
                       locator_type, char_start, char_end,
                       page_start, page_end, element_id,
                       quote, confidence, extraction_pass
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    cit.citation_id,
                    bundle.artifact_id,
                    cit.document_id,
                    cit.source_doc_hash,
                    loc.locator_type,
                    loc.char_start,
                    loc.char_end,
                    loc.page_start,
                    loc.page_end,
                    loc.element_id,
                    cit.quote,
                    cit.confidence,
                    cit.extraction_pass,
                ),
            )

    # ------------------------------------------------------------------
    # ArtifactStorePort implementation
    # ------------------------------------------------------------------

    async def create(self, bundle) -> Any:
        """Persist a new artifact bundle."""
        conn = self.conn
        row = self._bundle_to_row(bundle)

        conn.execute(
            """INSERT INTO artifacts (
                   artifact_id, document_id, version, artifact_type,
                   source_doc_hash, confidence, payload_json,
                   provenance_json, is_deleted
               ) VALUES (
                   :artifact_id, :document_id, :version, :artifact_type,
                   :source_doc_hash, :confidence, :payload_json,
                   :provenance_json, :is_deleted
               )""",
            row,
        )

        self._insert_refs(bundle)
        self._insert_citations(bundle)

        conn.commit()
        return bundle

    async def get(self, artifact_id: str) -> Optional[Any]:
        """Retrieve an artifact by ID. Returns ``None`` if not found or deleted."""
        conn = self.conn
        row = conn.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        if row is None:
            return None
        if row["is_deleted"]:
            return None
        return self._row_to_bundle(dict(row))

    async def get_latest(
        self,
        document_id: str,
        artifact_type: Optional[str] = None,
    ) -> Optional[Any]:
        """Get the latest version of an artifact for a document."""
        conn = self.conn
        query = "SELECT * FROM artifacts WHERE document_id = ? AND is_deleted = 0"
        params: list = [document_id]

        if artifact_type is not None:
            query += " AND artifact_type = ?"
            params.append(artifact_type)

        query += " ORDER BY version DESC LIMIT 1"

        row = conn.execute(query, params).fetchone()
        if row is None:
            return None
        return self._row_to_bundle(dict(row))

    async def list_by_document(self, document_id: str):
        """List all non-deleted artifacts for a document."""
        conn = self.conn
        rows = conn.execute(
            "SELECT * FROM artifacts WHERE document_id = ? AND is_deleted = 0",
            (document_id,),
        ).fetchall()
        return [self._row_to_bundle(dict(r)) for r in rows]

    async def list_versions(self, document_id: str):
        """Return all version numbers for a document."""
        conn = self.conn
        rows = conn.execute(
            "SELECT DISTINCT version FROM artifacts "
            "WHERE document_id = ? AND is_deleted = 0 "
            "ORDER BY version",
            (document_id,),
        ).fetchall()
        return [r["version"] for r in rows]

    async def resolve_version(
        self,
        document_id: str,
        version: int,
    ) -> Optional[Any]:
        """Atomic version resolution with lock."""
        conn = self.conn
        # ``SERIALIZABLE`` + explicit SELECT FOR UPDATE equivalent
        # SQLite doesn't support SELECT FOR UPDATE, but BEGIN IMMEDIATE
        # gives us a write lock that serialises access.
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT * FROM artifacts "
                "WHERE document_id = ? AND version = ? AND is_deleted = 0",
                (document_id, version),
            ).fetchone()
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        if row is None:
            return None
        return self._row_to_bundle(dict(row))

    async def get_graph_neighborhood(
        self,
        artifact_id: str,
        hops: int = 1,
    ) -> Dict[str, Any]:
        """N-hop dependency traversal from an artifact."""
        conn = self.conn
        nodes: List[Any] = []
        edges: List[Dict[str, Any]] = []
        visited: set = {artifact_id}

        async def _traverse(current_id: str, depth: int) -> None:
            if depth > hops:
                return
            bundle = await self.get(current_id)
            if bundle is None:
                return
            nodes.append(bundle)

            for ref_row in conn.execute(
                "SELECT to_artifact, ref_type FROM artifact_refs WHERE from_artifact = ?",
                (current_id,),
            ).fetchall():
                ref_id = ref_row["to_artifact"]
                if ref_id not in visited:
                    edges.append({
                        "from": current_id,
                        "to": ref_id,
                        "ref_type": ref_row["ref_type"],
                    })
                    visited.add(ref_id)
                    await _traverse(ref_id, depth + 1)

            for ref_row in conn.execute(
                "SELECT from_artifact, ref_type FROM artifact_refs WHERE to_artifact = ?",
                (current_id,),
            ).fetchall():
                ref_id = ref_row["from_artifact"]
                if ref_id not in visited:
                    edges.append({
                        "from": ref_id,
                        "to": current_id,
                        "ref_type": ref_row["ref_type"],
                    })
                    visited.add(ref_id)
                    await _traverse(ref_id, depth + 1)

        await _traverse(artifact_id, 0)
        return {"nodes": nodes, "edges": edges}

    async def get_inverse_citations(
        self,
        document_id: str,
        char_start: int,
        char_end: int,
    ):
        """Find all artifacts that cite this exact span in the source document."""
        conn = self.conn
        rows = conn.execute(
            """SELECT DISTINCT a.* FROM artifacts a
               JOIN citations c ON a.artifact_id = c.artifact_id
               WHERE c.document_id = ?
                 AND c.char_start = ?
                 AND c.char_end = ?
                 AND a.is_deleted = 0""",
            (document_id, char_start, char_end),
        ).fetchall()
        return [self._row_to_bundle(dict(r)) for r in rows]


# ---------------------------------------------------------------------------
# Context manager support
# ---------------------------------------------------------------------------


class ManagedSqliteStore:
    """Context-manager wrapper around ``SqliteArtifactStore``.

    Ensures the connection is always closed when the caller is done.

    Usage::

        async with ManagedSqliteStore("cyrex.db") as store:
            bundle = await store.create(bundle)
        # connection closed here
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._store = SqliteArtifactStore(db_path)

    async def __aenter__(self):
        return self._store

    async def __aexit__(self, *exc):
        self._store.close()
