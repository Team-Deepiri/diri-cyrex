"""PostgreSQL read model adapter for dead-reckoning records."""

from __future__ import annotations

from typing import Any

from app.database.postgres import get_postgres_manager
from app.pipeline.contracts.models import PredictionRecord
from app.pipeline.contracts.ports import ReckoningReadPort
from app.utils.json_utils import _json_value


class PostgresReckoningStore(ReckoningReadPort):
    """Read-only adapter over the reckoning PostgreSQL tables."""

    def __init__(self, postgres: Any = None) -> None:
        self._postgres = postgres

    async def _db(self) -> Any:
        return self._postgres or await get_postgres_manager()

    async def get_reckoning(self, document_id: str) -> list[PredictionRecord]:
        db = await self._db()
        rows = await db.fetch(
            """
            SELECT r.field_name, r.record_json, r.status,
                   a.actual_value, an.sigma_delta,
                   p.predicted_range_json, p.last_prior_update,
                   s.doc_count AS corpus_doc_count
            FROM cyrex.reckoning_records r
            LEFT JOIN cyrex.reckoning_actuals a
              ON a.document_id = r.document_id AND a.field_name = r.field_name
            LEFT JOIN cyrex.reckoning_anomalies an
              ON an.document_id = r.document_id AND an.field_name = r.field_name
            LEFT JOIN cyrex.reckoning_field_priors p
              ON p.field_name = r.field_name
            LEFT JOIN cyrex.reckoning_corpus_stats s
              ON s.field_name = r.field_name
            WHERE r.document_id = $1
            ORDER BY r.field_name
            """,
            document_id,
        )

        records: list[PredictionRecord] = []
        for row in rows:
            data = dict(_json_value(row["record_json"], {}))
            data["field_name"] = row["field_name"]
            data["status"] = row["status"] or data.get("status", "no_prior")
            if row["actual_value"] is not None:
                data["actual_value"] = _json_value(row["actual_value"], row["actual_value"])
            if row["sigma_delta"] is not None:
                data["sigma_delta"] = float(row["sigma_delta"])
            if row["predicted_range_json"] is not None:
                data["predicted_range"] = _json_value(row["predicted_range_json"], {})
            if row["last_prior_update"] is not None:
                data["last_prior_update"] = row["last_prior_update"]
            if row["corpus_doc_count"] is not None:
                data["corpus_doc_count"] = row["corpus_doc_count"]
            records.append(PredictionRecord.model_validate(data))
        return records


ReckoningStore = PostgresReckoningStore
