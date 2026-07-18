"""
AGI training_emitter — dual-write Helox training samples.

Writes:
  - Redis pipeline.helox-training.{raw,structured} via Sugar Glider bus
  - Postgres cyrex.helox_training_samples (durable mirror)
  - Postgres cyrex.helox_sample_lineage (provenance)

Used by the artifact pipeline (corrections, uploads, MCP) in addition to
RealtimeDataPipeline's runtime capture path.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.integrations.streaming.bus_publisher import (
    HELOX_TRAINING_RAW,
    HELOX_TRAINING_STRUCTURED,
    get_bus_publisher,
)
from app.logging_config import get_logger

logger = get_logger("cyrex.pipeline.training_emitter")

MIN_QUALITY = 0.4

_ENSURE_SAMPLES_SQL = """
CREATE SCHEMA IF NOT EXISTS cyrex;

CREATE TABLE IF NOT EXISTS cyrex.helox_training_samples (
    id BIGSERIAL,
    record_id TEXT PRIMARY KEY,
    stream_type TEXT NOT NULL CHECK (stream_type IN ('raw', 'structured')),
    producer TEXT NOT NULL DEFAULT 'training_emitter',
    text TEXT,
    instruction TEXT,
    input_text TEXT,
    output_text TEXT,
    category TEXT,
    quality_score DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cyrex.helox_sample_lineage (
    lineage_id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_id TEXT,
    producer TEXT NOT NULL,
    document_id TEXT,
    artifact_id TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_UPSERT_SAMPLE_SQL = """
INSERT INTO cyrex.helox_training_samples (
    record_id, stream_type, producer, text, instruction, input_text,
    output_text, category, quality_score, metadata_json, created_at
) VALUES (
    %(record_id)s, %(stream_type)s, %(producer)s, %(text)s, %(instruction)s,
    %(input_text)s, %(output_text)s, %(category)s, %(quality_score)s,
    %(metadata_json)s::jsonb, %(created_at)s
)
ON CONFLICT (record_id) DO UPDATE SET
    stream_type = EXCLUDED.stream_type,
    producer = EXCLUDED.producer,
    text = EXCLUDED.text,
    instruction = EXCLUDED.instruction,
    input_text = EXCLUDED.input_text,
    output_text = EXCLUDED.output_text,
    category = EXCLUDED.category,
    quality_score = EXCLUDED.quality_score,
    metadata_json = EXCLUDED.metadata_json
"""

_INSERT_LINEAGE_SQL = """
INSERT INTO cyrex.helox_sample_lineage (
    lineage_id, record_id, source_type, source_id, producer,
    document_id, artifact_id, metadata_json, created_at
) VALUES (
    %(lineage_id)s, %(record_id)s, %(source_type)s, %(source_id)s, %(producer)s,
    %(document_id)s, %(artifact_id)s, %(metadata_json)s::jsonb, %(created_at)s
)
ON CONFLICT (lineage_id) DO NOTHING
"""


class TrainingEmitter:
    """Emit artifact-derived / correction training samples to Helox."""

    def __init__(
        self,
        *,
        postgres: Any = None,
        redis_client: Any = None,
        producer: str = "training_emitter",
    ) -> None:
        self._postgres = postgres
        self._producer = producer
        self._bus = get_bus_publisher(redis_client=redis_client)
        self._schema_ready = False

    async def _ensure_schema(self) -> None:
        if self._schema_ready or self._postgres is None:
            return
        if hasattr(self._postgres, "execute"):
            await self._postgres.execute(_ENSURE_SAMPLES_SQL)
        elif hasattr(self._postgres, "run"):
            await self._postgres.run(_ENSURE_SAMPLES_SQL)
        self._schema_ready = True

    async def emit_structured(
        self,
        *,
        instruction: str,
        output: str,
        input_text: str = "",
        category: str = "artifact",
        quality_score: float = 1.0,
        record_id: Optional[str] = None,
        document_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
        source_type: str = "artifact",
        source_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        producer: Optional[str] = None,
    ) -> Optional[str]:
        if quality_score < MIN_QUALITY:
            logger.debug("training_emitter_quality_drop", score=quality_score)
            return None

        rid = record_id or str(uuid.uuid4())
        prod = producer or self._producer
        meta = dict(metadata or {})
        if document_id:
            meta.setdefault("document_id", document_id)
        if artifact_id:
            meta.setdefault("artifact_id", artifact_id)

        payload = {
            "id": rid,
            "record_id": rid,
            "instruction": instruction,
            "input": input_text,
            "input_text": input_text,
            "output": output,
            "output_text": output,
            "category": category,
            "quality_score": quality_score,
            "producer": prod,
            "source": prod,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": meta,
        }

        await self._persist_postgres(
            record_id=rid,
            stream_type="structured",
            producer=prod,
            text=None,
            instruction=instruction,
            input_text=input_text,
            output_text=output,
            category=category,
            quality_score=quality_score,
            metadata=meta,
            document_id=document_id,
            artifact_id=artifact_id,
            source_type=source_type,
            source_id=source_id or artifact_id,
        )

        await self._bus.publish(
            HELOX_TRAINING_STRUCTURED,
            "helox.training.structured",
            payload,
        )
        return rid

    async def emit_raw(
        self,
        *,
        text: str,
        quality_score: float = 1.0,
        record_id: Optional[str] = None,
        document_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
        source_type: str = "artifact",
        source_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        producer: Optional[str] = None,
        category: str = "artifact_raw",
    ) -> Optional[str]:
        if quality_score < MIN_QUALITY:
            return None

        rid = record_id or str(uuid.uuid4())
        prod = producer or self._producer
        meta = dict(metadata or {})

        payload = {
            "id": rid,
            "record_id": rid,
            "text": text,
            "quality_score": quality_score,
            "producer": prod,
            "source": prod,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": meta,
        }

        await self._persist_postgres(
            record_id=rid,
            stream_type="raw",
            producer=prod,
            text=text,
            instruction=None,
            input_text=None,
            output_text=None,
            category=category,
            quality_score=quality_score,
            metadata=meta,
            document_id=document_id,
            artifact_id=artifact_id,
            source_type=source_type,
            source_id=source_id or artifact_id,
        )

        await self._bus.publish(HELOX_TRAINING_RAW, "helox.training.raw", payload)
        return rid

    async def emit_correction(
        self,
        *,
        instruction: str,
        corrected_output: str,
        input_text: str = "",
        document_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
        quality_score: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Corrections use producer=correction_writer for Helox filter pipelines."""
        return await self.emit_structured(
            instruction=instruction,
            output=corrected_output,
            input_text=input_text,
            category="correction",
            quality_score=quality_score,
            document_id=document_id,
            artifact_id=artifact_id,
            source_type="correction",
            source_id=artifact_id,
            metadata=metadata,
            producer="correction_writer",
        )

    async def _persist_postgres(
        self,
        *,
        record_id: str,
        stream_type: str,
        producer: str,
        text: Optional[str],
        instruction: Optional[str],
        input_text: Optional[str],
        output_text: Optional[str],
        category: Optional[str],
        quality_score: float,
        metadata: Dict[str, Any],
        document_id: Optional[str],
        artifact_id: Optional[str],
        source_type: str,
        source_id: Optional[str],
    ) -> None:
        if self._postgres is None:
            return
        try:
            await self._ensure_schema()
            now = datetime.now(timezone.utc)
            sample_params = {
                "record_id": record_id,
                "stream_type": stream_type,
                "producer": producer,
                "text": text,
                "instruction": instruction,
                "input_text": input_text,
                "output_text": output_text,
                "category": category,
                "quality_score": quality_score,
                "metadata_json": json.dumps(metadata),
                "created_at": now,
            }
            lineage_params = {
                "lineage_id": str(uuid.uuid4()),
                "record_id": record_id,
                "source_type": source_type,
                "source_id": source_id,
                "producer": producer,
                "document_id": document_id,
                "artifact_id": artifact_id,
                "metadata_json": json.dumps(metadata),
                "created_at": now,
            }
            if hasattr(self._postgres, "execute"):
                await self._postgres.execute(_UPSERT_SAMPLE_SQL, sample_params)
                await self._postgres.execute(_INSERT_LINEAGE_SQL, lineage_params)
            elif hasattr(self._postgres, "run"):
                await self._postgres.run(_UPSERT_SAMPLE_SQL, sample_params)
                await self._postgres.run(_INSERT_LINEAGE_SQL, lineage_params)
        except Exception as exc:
            logger.warning(
                "training_emitter_postgres_failed",
                record_id=record_id,
                error=str(exc),
            )


def create_training_emitter(
    *,
    postgres: Any = None,
    redis_client: Any = None,
    producer: str = "training_emitter",
) -> TrainingEmitter:
    return TrainingEmitter(
        postgres=postgres,
        redis_client=redis_client,
        producer=producer,
    )
