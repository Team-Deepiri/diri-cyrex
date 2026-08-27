"""Write reckoning records to Postgres after the reckoning stage."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.database.postgres import get_postgres_manager
from app.pipeline.contracts.models import PredictionRecord, PredictionStatus

logger = logging.getLogger("cyrex.pipeline.registry.reckoning_writer")


class PostgresReckoningWriter:
    """Persist reckoning stage output to cyrex.reckoning_* tables."""

    def __init__(self, postgres: Any = None) -> None:
        self._postgres = postgres

    async def _db(self) -> Any:
        return self._postgres or await get_postgres_manager()

    async def persist(
        self, document_id: str, records: list[PredictionRecord]
    ) -> int:
        if not records:
            return 0
        db = await self._db()
        n = 0
        for rec in records:
            payload = rec.model_dump(mode="json")
            status = (
                rec.status.value
                if isinstance(rec.status, PredictionStatus)
                else str(rec.status)
            )
            await db.execute(
                """
                INSERT INTO cyrex.reckoning_records
                    (document_id, field_name, record_json, status)
                VALUES ($1::uuid, $2, $3::jsonb, $4)
                ON CONFLICT (document_id, field_name) DO UPDATE SET
                    record_json = EXCLUDED.record_json,
                    status = EXCLUDED.status,
                    created_at = NOW()
                """,
                document_id,
                rec.field_name,
                json.dumps(payload),
                status,
            )
            if rec.actual_value is not None:
                await db.execute(
                    """
                    INSERT INTO cyrex.reckoning_actuals
                        (document_id, field_name, actual_value)
                    VALUES ($1::uuid, $2, $3::jsonb)
                    ON CONFLICT (document_id, field_name) DO UPDATE SET
                        actual_value = EXCLUDED.actual_value,
                        confirmed_at = NOW()
                    """,
                    document_id,
                    rec.field_name,
                    json.dumps(rec.actual_value, default=str),
                )
            if rec.status == PredictionStatus.ANOMALOUS and rec.sigma_delta is not None:
                await db.execute(
                    """
                    INSERT INTO cyrex.reckoning_anomalies
                        (document_id, field_name, sigma_delta)
                    VALUES ($1::uuid, $2, $3)
                    ON CONFLICT (document_id, field_name) DO UPDATE SET
                        sigma_delta = EXCLUDED.sigma_delta,
                        detected_at = NOW()
                    """,
                    document_id,
                    rec.field_name,
                    float(rec.sigma_delta),
                )
            n += 1
        logger.info(
            "reckoning persisted",
            extra={"document_id": document_id, "fields": n},
        )
        return n
