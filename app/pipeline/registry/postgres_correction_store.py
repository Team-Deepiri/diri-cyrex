"""PostgreSQL correction / learning-artifact store (postgres-cyrex)."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, List, Optional

from app.pipeline.contracts.models import (
    ArtifactBundle,
    ArtifactType,
    Citation,
    LearningArtifact,
    Provenance,
)
from app.pipeline.registry.postgres_store import PostgresArtifactStore

logger = logging.getLogger("cyrex.pipeline.registry.postgres_correction_store")


class PostgresCorrectionStore:
    """CorrectionWriterPort backed by ``cyrex.learning_artifacts`` + artifact graph."""

    def __init__(self, postgres: Any = None) -> None:
        self._postgres = postgres
        self._artifacts = PostgresArtifactStore(postgres=postgres)

    async def _db(self) -> Any:
        if self._postgres is not None:
            return self._postgres
        from app.database.postgres import get_postgres_manager

        return await get_postgres_manager()

    async def ensure_schema(self) -> None:
        await self._artifacts.ensure_schema()

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
        await self._artifacts.ensure_schema()
        db = await self._db()

        learning = LearningArtifact(
            artifact_id=artifact_id if artifact_id.startswith("learn_") else f"learn_{artifact_id}",
            document_id=document_id or corrected_citation.document_id,
            field_name=field_name,
            original_value=original_value,
            corrected_value=corrected_value,
            corrected_citation=corrected_citation,
            actor_id=actor_id,
        )

        await db.execute(
            """
            INSERT INTO cyrex.learning_artifacts (
                artifact_id, document_id, field_name, original_value,
                corrected_value, citation_json, actor_id, timestamp, exported
            ) VALUES ($1,$2,$3,$4::jsonb,$5::jsonb,$6::jsonb,$7,$8,FALSE)
            ON CONFLICT (artifact_id) DO UPDATE SET
                document_id = EXCLUDED.document_id,
                field_name = EXCLUDED.field_name,
                original_value = EXCLUDED.original_value,
                corrected_value = EXCLUDED.corrected_value,
                citation_json = EXCLUDED.citation_json,
                actor_id = EXCLUDED.actor_id,
                timestamp = EXCLUDED.timestamp,
                exported = FALSE
            """,
            learning.artifact_id,
            learning.document_id,
            learning.field_name,
            json.dumps(learning.original_value),
            json.dumps(learning.corrected_value),
            learning.corrected_citation.model_dump_json(),
            learning.actor_id,
            learning.timestamp or datetime.now(timezone.utc),
        )

        bundle = ArtifactBundle(
            artifact_id=learning.artifact_id,
            document_id=learning.document_id,
            artifact_type=ArtifactType.LEARNING,
            source_doc_hash=corrected_citation.source_doc_hash,
            confidence=float(corrected_citation.confidence),
            payload={"learning_artifact": learning.model_dump(mode="json")},
            provenance=Provenance(
                source_doc_hash=corrected_citation.source_doc_hash,
                document_id=learning.document_id,
            ),
            citations=[corrected_citation],
        )
        # Persist into the main artifact graph as well
        existing = await self._artifacts.get(bundle.artifact_id)
        if existing is None:
            await self._artifacts.create(bundle)
        return bundle

    async def drain_for_training(self, batch_size: int = 100) -> List[LearningArtifact]:
        await self._artifacts.ensure_schema()
        db = await self._db()
        rows = await db.fetch(
            """
            SELECT * FROM cyrex.learning_artifacts
            WHERE exported = FALSE
            ORDER BY timestamp
            LIMIT $1
            """,
            batch_size,
        )
        artifacts: List[LearningArtifact] = []
        ids: List[str] = []
        for row in rows:
            citation_raw = row["citation_json"]
            if isinstance(citation_raw, str):
                citation = Citation.model_validate_json(citation_raw)
            else:
                citation = Citation.model_validate(citation_raw)
            orig = row["original_value"]
            corr = row["corrected_value"]
            if isinstance(orig, str):
                orig = json.loads(orig)
            if isinstance(corr, str):
                corr = json.loads(corr)
            ts = row["timestamp"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            artifacts.append(
                LearningArtifact(
                    artifact_id=row["artifact_id"],
                    document_id=row["document_id"],
                    field_name=row["field_name"],
                    original_value=orig,
                    corrected_value=corr,
                    corrected_citation=citation,
                    actor_id=row["actor_id"],
                    timestamp=ts,
                )
            )
            ids.append(row["artifact_id"])
        if ids:
            await db.execute(
                """
                UPDATE cyrex.learning_artifacts
                SET exported = TRUE
                WHERE artifact_id = ANY($1::text[])
                """,
                ids,
            )
        return artifacts

    async def get_by_id(self, artifact_id: str) -> Optional[LearningArtifact]:
        await self._artifacts.ensure_schema()
        db = await self._db()
        row = await db.fetchrow(
            "SELECT * FROM cyrex.learning_artifacts WHERE artifact_id = $1",
            artifact_id,
        )
        if row is None:
            return None
        citation_raw = row["citation_json"]
        if isinstance(citation_raw, str):
            citation = Citation.model_validate_json(citation_raw)
        else:
            citation = Citation.model_validate(citation_raw)
        orig = row["original_value"]
        corr = row["corrected_value"]
        if isinstance(orig, str):
            orig = json.loads(orig)
        if isinstance(corr, str):
            corr = json.loads(corr)
        ts = row["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return LearningArtifact(
            artifact_id=row["artifact_id"],
            document_id=row["document_id"],
            field_name=row["field_name"],
            original_value=orig,
            corrected_value=corr,
            corrected_citation=citation,
            actor_id=row["actor_id"],
            timestamp=ts,
        )
