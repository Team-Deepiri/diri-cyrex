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
from app.pipeline.helox_training_schema import (
    HELOX_SAMPLE_LINEAGE_INSERT_SQL,
    HELOX_TRAINING_SAMPLES_DDL,
    HELOX_TRAINING_SAMPLE_UPSERT_SQL,
)

logger = get_logger("cyrex.pipeline.training_emitter")

MIN_QUALITY = 0.4


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
            await self._postgres.execute(HELOX_TRAINING_SAMPLES_DDL)
        elif hasattr(self._postgres, "run"):
            await self._postgres.run(HELOX_TRAINING_SAMPLES_DDL)
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
            training_text = text or "\n\n".join(
                p for p in (instruction, input_text, output_text) if p
            ) or json.dumps(metadata)
            sample_params = {
                "record_id": record_id,
                "stream_type": stream_type,
                "category": category,
                "text": training_text,
                "instruction": instruction,
                "input_text": input_text,
                "output_text": output_text,
                "context": None,
                "quality_score": quality_score,
                "producer": producer,
                "agent_id": None,
                "session_id": None,
                "user_id": None,
                "tool_name": None,
                "model_name": None,
                "schema_version": "training_emitter.v1",
                "payload": json.dumps(
                    {
                        "instruction": instruction,
                        "input_text": input_text,
                        "output_text": output_text,
                        "text": text,
                        "metadata": metadata,
                    },
                    default=str,
                ),
                "metadata_json": json.dumps(metadata),
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
                await self._postgres.execute(
                    HELOX_TRAINING_SAMPLE_UPSERT_SQL, sample_params
                )
                await self._postgres.execute(
                    HELOX_SAMPLE_LINEAGE_INSERT_SQL, lineage_params
                )
            elif hasattr(self._postgres, "run"):
                await self._postgres.run(HELOX_TRAINING_SAMPLE_UPSERT_SQL, sample_params)
                await self._postgres.run(HELOX_SAMPLE_LINEAGE_INSERT_SQL, lineage_params)
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
