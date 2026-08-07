"""PostgreSQL-backed ArtifactStorePort for Cyrex AGI (postgres-cyrex / cyrex_db).

Replaces SqliteArtifactStore for runtime. Schema lives in ``cyrex.*`` and is
created idempotently on first use (also mirrored in platform
``scripts/database/postgres-init-cyrex.sql`` for fresh volumes).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.pipeline.contracts.models import ArtifactBundle, ArtifactType, Provenance
from app.pipeline.contracts.ports import PressureSignalSink

logger = logging.getLogger("cyrex.pipeline.registry.postgres_store")

_DDL = [
    "CREATE SCHEMA IF NOT EXISTS cyrex",
    """
    CREATE TABLE IF NOT EXISTS cyrex.artifacts (
        artifact_id      TEXT PRIMARY KEY,
        document_id      TEXT    NOT NULL,
        version          INTEGER NOT NULL DEFAULT 1,
        artifact_type    TEXT    NOT NULL,
        source_doc_hash  TEXT    NOT NULL,
        confidence       DOUBLE PRECISION NOT NULL,
        payload_json     JSONB   NOT NULL DEFAULT '{}'::jsonb,
        provenance_json  JSONB   NOT NULL DEFAULT '{}'::jsonb,
        is_deleted       BOOLEAN NOT NULL DEFAULT FALSE,
        created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cyrex.artifact_refs (
        from_artifact TEXT NOT NULL,
        to_artifact   TEXT NOT NULL,
        ref_type      TEXT NOT NULL,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (from_artifact, to_artifact, ref_type)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cyrex.citations (
        citation_id      TEXT PRIMARY KEY,
        artifact_id      TEXT NOT NULL REFERENCES cyrex.artifacts(artifact_id)
                         ON DELETE CASCADE,
        document_id      TEXT NOT NULL,
        source_doc_hash  TEXT NOT NULL,
        locator_type     TEXT NOT NULL,
        char_start       INTEGER,
        char_end         INTEGER,
        page_start       INTEGER,
        page_end         INTEGER,
        element_id       TEXT,
        quote            TEXT NOT NULL,
        confidence       DOUBLE PRECISION NOT NULL,
        extraction_pass  INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cyrex.learning_artifacts (
        artifact_id      TEXT PRIMARY KEY,
        document_id      TEXT NOT NULL,
        field_name       TEXT NOT NULL,
        original_value   JSONB,
        corrected_value  JSONB NOT NULL,
        citation_json    JSONB NOT NULL,
        actor_id         TEXT NOT NULL,
        timestamp        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        exported         BOOLEAN NOT NULL DEFAULT FALSE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cyrex_artifacts_doc ON cyrex.artifacts(document_id)",
    "CREATE INDEX IF NOT EXISTS idx_cyrex_artifacts_doc_type ON cyrex.artifacts"
    "(document_id, artifact_type)",
    "CREATE INDEX IF NOT EXISTS idx_cyrex_refs_from ON cyrex.artifact_refs(from_artifact)",
    "CREATE INDEX IF NOT EXISTS idx_cyrex_refs_to ON cyrex.artifact_refs(to_artifact)",
    "CREATE INDEX IF NOT EXISTS idx_cyrex_citations_doc_span ON cyrex.citations"
    "(document_id, char_start, char_end)",
    "CREATE INDEX IF NOT EXISTS idx_cyrex_learning_exported ON cyrex.learning_artifacts(exported)",
]


class PostgresArtifactStore:
    """asyncpg-backed ``ArtifactStorePort`` using ``postgres-cyrex``.

    Optional ``pressure_sink`` mirrors Track A SqliteArtifactStore (#128): after
    a successful ``create()``, project and emit pressure events when the sink
    is wired.
    """

    def __init__(
        self,
        postgres: Any = None,
        *,
        pressure_sink: Optional[PressureSignalSink] = None,
    ) -> None:
        self._postgres = postgres
        self._pressure_sink = pressure_sink
        self._schema_ready = False

    async def _db(self) -> Any:
        if self._postgres is not None:
            return self._postgres
        from app.database.postgres import get_postgres_manager

        return await get_postgres_manager()

    async def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        db = await self._db()
        for stmt in _DDL:
            await db.execute(stmt)
        self._schema_ready = True
        logger.info("cyrex artifact schema ensured on postgres")

    def _row_to_bundle(self, row: Any) -> ArtifactBundle:
        payload = row["payload_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        provenance_raw = row["provenance_json"]
        if isinstance(provenance_raw, str):
            provenance_raw = json.loads(provenance_raw)
        provenance = Provenance.model_validate(provenance_raw or {})
        created = row.get("created_at") if hasattr(row, "get") else row["created_at"]
        if created is None:
            created = datetime.now(timezone.utc)
        elif isinstance(created, str):
            created = datetime.fromisoformat(created)

        return ArtifactBundle(
            artifact_id=row["artifact_id"],
            document_id=row["document_id"],
            version=int(row["version"]),
            artifact_type=ArtifactType(row["artifact_type"]),
            source_doc_hash=row["source_doc_hash"],
            confidence=float(row["confidence"]),
            payload=dict(payload or {}),
            provenance=provenance,
            is_deleted=bool(row["is_deleted"]),
            created_at=created,
        )

    async def _insert_refs(self, db: Any, bundle: ArtifactBundle) -> None:
        now = datetime.now(timezone.utc)
        edges: list[tuple[str, str, str]] = []
        for ref_id in bundle.provenance.depends_on:
            edges.append((bundle.artifact_id, ref_id, "depends_on"))
        for ref_id in bundle.provenance.depended_on_by:
            edges.append((bundle.artifact_id, ref_id, "depended_on_by"))
        for ref_id in bundle.provenance.cross_references:
            edges.append((bundle.artifact_id, ref_id, "cross_reference"))
        for from_id, to_id, ref_type in edges:
            await db.execute(
                """
                INSERT INTO cyrex.artifact_refs (from_artifact, to_artifact, ref_type, created_at)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT DO NOTHING
                """,
                from_id,
                to_id,
                ref_type,
                now,
            )

    async def _insert_citations(self, db: Any, bundle: ArtifactBundle) -> None:
        for cit in bundle.citations:
            loc = cit.locator
            await db.execute(
                """
                INSERT INTO cyrex.citations (
                    citation_id, artifact_id, document_id, source_doc_hash,
                    locator_type, char_start, char_end, page_start, page_end,
                    element_id, quote, confidence, extraction_pass
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                ON CONFLICT (citation_id) DO NOTHING
                """,
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
            )

    async def create(self, bundle: ArtifactBundle) -> ArtifactBundle:
        await self.ensure_schema()
        db = await self._db()
        await db.execute(
            """
            INSERT INTO cyrex.artifacts (
                artifact_id, document_id, version, artifact_type,
                source_doc_hash, confidence, payload_json, provenance_json,
                is_deleted, created_at
            ) VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8::jsonb,$9,$10)
            """,
            bundle.artifact_id,
            bundle.document_id,
            bundle.version,
            bundle.artifact_type.value,
            bundle.source_doc_hash,
            bundle.confidence,
            json.dumps(bundle.payload),
            json.dumps(bundle.provenance.model_dump(mode="json")),
            bundle.is_deleted,
            bundle.created_at or datetime.now(timezone.utc),
        )
        await self._insert_refs(db, bundle)
        await self._insert_citations(db, bundle)

        # Same seam as Track A SqliteArtifactStore (#128) — optional pressure emit.
        if self._pressure_sink is not None:
            try:
                from app.pipeline.projectors.pressure_signals import project_pressure_events

                events = project_pressure_events(bundle)
                if events:
                    await self._pressure_sink.emit_many(events)
            except ImportError:
                logger.debug(
                    "pressure_signals projector not available yet (Track A #128); skip emit"
                )
            except Exception as exc:
                logger.warning("pressure emit after artifact create failed: %s", exc)

        return bundle

    async def get(self, artifact_id: str) -> Optional[ArtifactBundle]:
        await self.ensure_schema()
        db = await self._db()
        row = await db.fetchrow(
            """
            SELECT * FROM cyrex.artifacts
            WHERE artifact_id = $1 AND is_deleted = FALSE
            """,
            artifact_id,
        )
        return self._row_to_bundle(row) if row else None

    async def get_latest(
        self,
        document_id: str,
        artifact_type: Optional[str] = None,
    ) -> Optional[ArtifactBundle]:
        await self.ensure_schema()
        db = await self._db()
        if artifact_type is not None:
            row = await db.fetchrow(
                """
                SELECT * FROM cyrex.artifacts
                WHERE document_id = $1 AND artifact_type = $2 AND is_deleted = FALSE
                ORDER BY version DESC
                LIMIT 1
                """,
                document_id,
                artifact_type,
            )
        else:
            row = await db.fetchrow(
                """
                SELECT * FROM cyrex.artifacts
                WHERE document_id = $1 AND is_deleted = FALSE
                ORDER BY version DESC
                LIMIT 1
                """,
                document_id,
            )
        return self._row_to_bundle(row) if row else None

    async def list_by_document(self, document_id: str) -> List[ArtifactBundle]:
        await self.ensure_schema()
        db = await self._db()
        rows = await db.fetch(
            """
            SELECT * FROM cyrex.artifacts
            WHERE document_id = $1 AND is_deleted = FALSE
            ORDER BY version
            """,
            document_id,
        )
        return [self._row_to_bundle(r) for r in rows]

    async def list_versions(self, document_id: str) -> List[int]:
        await self.ensure_schema()
        db = await self._db()
        rows = await db.fetch(
            """
            SELECT DISTINCT version FROM cyrex.artifacts
            WHERE document_id = $1 AND is_deleted = FALSE
            ORDER BY version
            """,
            document_id,
        )
        return [int(r["version"]) for r in rows]

    async def resolve_version(
        self,
        document_id: str,
        version: int,
    ) -> Optional[ArtifactBundle]:
        await self.ensure_schema()
        db = await self._db()
        row = await db.fetchrow(
            """
            SELECT * FROM cyrex.artifacts
            WHERE document_id = $1 AND version = $2 AND is_deleted = FALSE
            FOR SHARE
            """,
            document_id,
            version,
        )
        return self._row_to_bundle(row) if row else None

    async def get_graph_neighborhood(
        self,
        artifact_id: str,
        hops: int = 1,
    ) -> Dict[str, Any]:
        await self.ensure_schema()
        db = await self._db()
        nodes: List[ArtifactBundle] = []
        edges: List[Dict[str, Any]] = []
        visited: set[str] = {artifact_id}

        async def _traverse(current_id: str, depth: int) -> None:
            if depth > hops:
                return
            bundle = await self.get(current_id)
            if bundle is None:
                return
            nodes.append(bundle)

            out_rows = await db.fetch(
                "SELECT to_artifact, ref_type FROM cyrex.artifact_refs WHERE from_artifact = $1",
                current_id,
            )
            for ref_row in out_rows:
                ref_id = ref_row["to_artifact"]
                if ref_id not in visited:
                    edges.append(
                        {
                            "from": current_id,
                            "to": ref_id,
                            "ref_type": ref_row["ref_type"],
                        }
                    )
                    visited.add(ref_id)
                    await _traverse(ref_id, depth + 1)

            in_rows = await db.fetch(
                "SELECT from_artifact, ref_type FROM cyrex.artifact_refs WHERE to_artifact = $1",
                current_id,
            )
            for ref_row in in_rows:
                ref_id = ref_row["from_artifact"]
                if ref_id not in visited:
                    edges.append(
                        {
                            "from": ref_id,
                            "to": current_id,
                            "ref_type": ref_row["ref_type"],
                        }
                    )
                    visited.add(ref_id)
                    await _traverse(ref_id, depth + 1)

        await _traverse(artifact_id, 0)
        return {"nodes": nodes, "edges": edges}

    async def get_inverse_citations(
        self,
        document_id: str,
        char_start: int,
        char_end: int,
    ) -> List[ArtifactBundle]:
        await self.ensure_schema()
        db = await self._db()
        rows = await db.fetch(
            """
            SELECT DISTINCT a.*
            FROM cyrex.artifacts a
            JOIN cyrex.citations c ON a.artifact_id = c.artifact_id
            WHERE c.document_id = $1
              AND c.char_start = $2
              AND c.char_end = $3
              AND a.is_deleted = FALSE
            """,
            document_id,
            char_start,
            char_end,
        )
        return [self._row_to_bundle(r) for r in rows]
