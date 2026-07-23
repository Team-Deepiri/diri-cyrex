"""PostgreSQL read model adapter for epistemic pressure cells."""

from __future__ import annotations

import json
from typing import Any, Optional

from app.database.postgres import get_postgres_manager
from app.pipeline.contracts.models import PressureCell
from app.pipeline.contracts.ports import PressureReadModelPort


def _json_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return value


class PostgresPressureStore(PressureReadModelPort):
    """Read-only adapter over pressure cells and their supporting tables."""

    def __init__(self, postgres: Any = None) -> None:
        self._postgres = postgres

    async def _db(self) -> Any:
        return self._postgres or await get_postgres_manager()

    async def get_pressure(self, document_id: Optional[str] = None) -> list[PressureCell]:
        db = await self._db()
        params: tuple[Any, ...] = () if document_id is None else (document_id,)
        where = "" if document_id is None else "WHERE c.document_id = $1"
        rows = await db.fetch(
            f"""
            SELECT c.document_id, c.section_id, c.page, c.score, c.is_fault_zone,
                   c.cell_json,
                   COALESCE(m.discrepancy_count, 0) AS discrepancy_count,
                   COALESCE(m.reflect_failures, 0) AS reflect_failures,
                   COALESCE(m.low_confidence_count, 0) AS low_confidence_count,
                   COALESCE(m.duel_disagreements, 0) AS duel_disagreements,
                   COALESCE(
                     ARRAY(
                       SELECT ca.artifact_id
                       FROM cyrex.pressure_cell_artifacts ca
                       WHERE ca.document_id = c.document_id
                         AND ca.section_id = c.section_id
                         AND ca.page = c.page
                       ORDER BY ca.artifact_id
                     ), '{{}}'::text[]
                   ) AS artifact_ids
            FROM cyrex.pressure_cells c
            LEFT JOIN cyrex.pressure_cell_metrics m
              ON m.document_id = c.document_id
             AND m.section_id = c.section_id
             AND m.page = c.page
            {where}
            ORDER BY c.section_id, c.page
            """,
            *params,
        )

        cells: list[PressureCell] = []
        for row in rows:
            cell_data = dict(_json_value(row["cell_json"], {}))
            cell_data.update(
                document_id=row["document_id"],
                section_id=row["section_id"],
                page=None if row["page"] == -1 else row["page"],
                score=float(row["score"]),
                is_fault_zone=bool(row["is_fault_zone"]),
                discrepancy_count=int(row["discrepancy_count"]),
                reflect_failures=int(row["reflect_failures"]),
                low_confidence_count=int(row["low_confidence_count"]),
                duel_disagreements=int(row["duel_disagreements"]),
                drill_down_artifact_ids=list(row["artifact_ids"] or []),
            )
            cells.append(PressureCell.model_validate(cell_data))
        return cells


PressureStore = PostgresPressureStore
