"""Update reckoning corpus stats after document reckoning completes."""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

from app.pipeline.contracts.models import PredictionRecord, PredictionStatus

logger = logging.getLogger("cyrex.pipeline.reckoning_updater")


class ReckoningCorpusUpdater:
    """Increment corpus priors from confirmed/anomalous/novel field actuals."""

    def __init__(self, postgres: Any = None) -> None:
        self._postgres = postgres

    async def _db(self) -> Any:
        if self._postgres is not None:
            return self._postgres
        from app.database.postgres import get_postgres_manager

        return await get_postgres_manager()

    async def update_from_records(
        self, records: List[PredictionRecord]
    ) -> int:
        """Upsert corpus stats + field priors for numeric fields with actuals."""
        updated = 0
        db = await self._db()
        for rec in records:
            if rec.actual_value is None:
                continue
            if rec.status not in (
                PredictionStatus.CONFIRMED,
                PredictionStatus.ANOMALOUS,
                PredictionStatus.NOVEL,
            ):
                continue
            try:
                actual_num = float(rec.actual_value)
            except (TypeError, ValueError):
                continue

            await db.execute(
                """
                INSERT INTO cyrex.reckoning_corpus_stats
                    (field_name, doc_count, mean, std, updated_at)
                VALUES ($1, 1, $2, 0, NOW())
                ON CONFLICT (field_name) DO UPDATE SET
                    doc_count = cyrex.reckoning_corpus_stats.doc_count + 1,
                    mean = (
                        cyrex.reckoning_corpus_stats.mean
                        * cyrex.reckoning_corpus_stats.doc_count
                        + EXCLUDED.mean
                    ) / (cyrex.reckoning_corpus_stats.doc_count + 1),
                    updated_at = NOW()
                """,
                rec.field_name,
                actual_num,
            )

            prior_range = rec.predicted_range or {}
            await db.execute(
                """
                INSERT INTO cyrex.reckoning_field_priors
                    (field_name, predicted_range_json, last_prior_update, updated_at)
                VALUES ($1, $2::jsonb, NOW(), NOW())
                ON CONFLICT (field_name) DO UPDATE SET
                    predicted_range_json = COALESCE(
                        EXCLUDED.predicted_range_json,
                        cyrex.reckoning_field_priors.predicted_range_json
                    ),
                    last_prior_update = NOW(),
                    updated_at = NOW()
                """,
                rec.field_name,
                json.dumps(
                    prior_range
                    if prior_range
                    else {"min": actual_num, "max": actual_num}
                ),
            )
            updated += 1

        if updated:
            logger.info("reckoning corpus updated", extra={"fields": updated})
        return updated
