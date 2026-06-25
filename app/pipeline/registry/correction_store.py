"""SQLite-backed correction store implementing CorrectionWriterPort."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from app.pipeline.contracts.models import (
    ArtifactBundle,
    ArtifactType,
    Citation,
    LearningArtifact,
    Provenance,
)
from app.pipeline.registry.sqlite_store import init_db


_CREATE_LEARNING = """\
CREATE TABLE IF NOT EXISTS learning_artifacts (
    artifact_id   TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL,
    field_name    TEXT NOT NULL,
    original_value TEXT,
    corrected_value TEXT NOT NULL,
    citation_json TEXT NOT NULL,
    actor_id      TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    exported      INTEGER NOT NULL DEFAULT 0
);
"""


class SqliteCorrectionStore:
    """Persistent store for LearningArtifacts used in live fine-tuning."""

    def __init__(self, db_path: str = "cyrex_corrections.db") -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = init_db(self.db_path)
            self._conn.execute(_CREATE_LEARNING)
            self._conn.commit()
        return self._conn

    async def submit_correction(
        self,
        artifact_id: str,
        field_name: str,
        corrected_value: Any,
        corrected_citation: Citation,
        actor_id: str,
        *,
        document_id: str = "",
        original_value: Any = None,
    ) -> ArtifactBundle:
        artifact = LearningArtifact(
            artifact_id=artifact_id,
            document_id=document_id or corrected_citation.document_id,
            field_name=field_name,
            original_value=original_value,
            corrected_value=corrected_value,
            corrected_citation=corrected_citation,
            actor_id=actor_id,
        )
        self.conn.execute(
            """INSERT OR REPLACE INTO learning_artifacts
               (artifact_id, document_id, field_name, original_value,
                corrected_value, citation_json, actor_id, timestamp)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                artifact.artifact_id,
                artifact.document_id,
                artifact.field_name,
                json.dumps(artifact.original_value),
                json.dumps(artifact.corrected_value),
                artifact.corrected_citation.model_dump_json(),
                artifact.actor_id,
                artifact.timestamp.isoformat(),
            ),
        )
        self.conn.commit()
        bundle = ArtifactBundle(
            document_id=artifact.document_id,
            artifact_type=ArtifactType.LEARNING,
            source_doc_hash=corrected_citation.source_doc_hash,
            confidence=corrected_citation.confidence,
            payload=artifact.model_dump(mode="json"),
            provenance=Provenance(
                source_doc_hash=corrected_citation.source_doc_hash,
                document_id=artifact.document_id,
            ),
            citations=[corrected_citation],
        )
        return bundle

    def drain_for_training(self, batch_size: int = 100) -> List[LearningArtifact]:
        rows = self.conn.execute(
            "SELECT * FROM learning_artifacts WHERE exported = 0 LIMIT ?",
            (batch_size,),
        ).fetchall()
        artifacts: List[LearningArtifact] = []
        ids: List[str] = []
        for row in rows:
            citation = Citation.model_validate_json(row["citation_json"])
            artifacts.append(
                LearningArtifact(
                    artifact_id=row["artifact_id"],
                    document_id=row["document_id"],
                    field_name=row["field_name"],
                    original_value=json.loads(row["original_value"]) if row["original_value"] else None,
                    corrected_value=json.loads(row["corrected_value"]),
                    corrected_citation=citation,
                    actor_id=row["actor_id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                )
            )
            ids.append(row["artifact_id"])
        if ids:
            placeholders = ",".join("?" for _ in ids)
            self.conn.execute(
                f"UPDATE learning_artifacts SET exported = 1 WHERE artifact_id IN ({placeholders})",
                ids,
            )
            self.conn.commit()
        return artifacts

    def get_by_id(self, artifact_id: str) -> Optional[LearningArtifact]:
        row = self.conn.execute(
            "SELECT * FROM learning_artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        if row is None:
            return None
        citation = Citation.model_validate_json(row["citation_json"])
        return LearningArtifact(
            artifact_id=row["artifact_id"],
            document_id=row["document_id"],
            field_name=row["field_name"],
            original_value=json.loads(row["original_value"]) if row["original_value"] else None,
            corrected_value=json.loads(row["corrected_value"]),
            corrected_citation=citation,
            actor_id=row["actor_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )
