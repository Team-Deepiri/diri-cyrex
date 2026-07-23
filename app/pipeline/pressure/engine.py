"""Pressure event projector and persistence engine."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Iterable, Mapping

from app.database.postgres import get_postgres_manager
from app.pipeline.contracts.models import PressureCell
from app.pipeline.contracts.pressure_events import PressureEvent
from app.settings import settings


class PressureEngine:
    """Project pressure events into PostgreSQL read-model rows."""

    def __init__(
        self,
        postgres: Any = None,
        *,
        fault_zone_threshold: float = 0.5,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        self._postgres = postgres
        self.fault_zone_threshold = fault_zone_threshold
        self._weights = dict(
            weights
            or {
                "pass_discrepancy": settings.PRESSURE_PASS_DISCREPANCY_WEIGHT,
                "reflect_failure": settings.PRESSURE_REFLECT_FAILURE_WEIGHT,
                "low_confidence_field": settings.PRESSURE_LOW_CONFIDENCE_WEIGHT,
                "duel_disagreement": settings.PRESSURE_DUEL_DISAGREEMENT_WEIGHT,
            }
        )

    async def _db(self) -> Any:
        return self._postgres or await get_postgres_manager()

    @staticmethod
    def _page_for_storage(page: int | None) -> int:
        # The composite migration key cannot contain NULL; -1 represents no page.
        return -1 if page is None else page

    def _cell_from_counts(
        self,
        key: tuple[str, str, int | None],
        counts: dict[str, Any],
    ) -> PressureCell:
        document_id, section_id, page = key
        score = min(
            1.0,
            sum(counts[name] * weight for name, weight in self._weights.items()),
        )
        return PressureCell(
            document_id=document_id,
            section_id=section_id,
            page=page,
            discrepancy_count=counts["pass_discrepancy"],
            reflect_failures=counts["reflect_failure"],
            low_confidence_count=counts["low_confidence_field"],
            duel_disagreements=counts["duel_disagreement"],
            score=score,
            is_fault_zone=score >= self.fault_zone_threshold,
            drill_down_artifact_ids=sorted(counts["artifact_ids"]),
        )

    async def accept(self, event: PressureEvent) -> PressureCell:
        return (await self.accept_many([event]))[0]

    async def accept_many(self, events: Iterable[PressureEvent]) -> list[PressureCell]:
        event_list = list(events)
        if not event_list:
            return []
        db = await self._db()
        grouped: dict[tuple[str, str, int | None], dict[str, Any]] = defaultdict(
            lambda: {
                "pass_discrepancy": 0,
                "reflect_failure": 0,
                "low_confidence_field": 0,
                "duel_disagreement": 0,
                "artifact_ids": set(),
            }
        )

        for event in event_list:
            payload = event.model_dump(mode="json")
            await db.execute(
                """
                INSERT INTO cyrex.pressure_events
                    (event_type, document_id, section_id, page, artifact_id, payload_json)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                """,
                event.event_type,
                event.document_id,
                event.section_id,
                event.page,
                event.artifact_id,
                json.dumps(payload),
            )
            counts = grouped[(event.document_id, event.section_id, event.page)]
            counts[event.event_type] += 1
            if event.artifact_id:
                counts["artifact_ids"].add(event.artifact_id)

        cells: list[PressureCell] = []
        for key, counts in grouped.items():
            cell = self._cell_from_counts(key, counts)
            storage_page = self._page_for_storage(cell.page)
            await db.execute(
                """
                INSERT INTO cyrex.pressure_cells
                    (document_id, section_id, page, score, is_fault_zone, cell_json)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                ON CONFLICT (document_id, section_id, page) DO UPDATE SET
                    score = EXCLUDED.score,
                    is_fault_zone = EXCLUDED.is_fault_zone,
                    cell_json = EXCLUDED.cell_json,
                    updated_at = NOW()
                """,
                cell.document_id,
                cell.section_id,
                storage_page,
                cell.score,
                cell.is_fault_zone,
                json.dumps(cell.model_dump(mode="json")),
            )
            await db.execute(
                """
                INSERT INTO cyrex.pressure_cell_metrics
                    (document_id, section_id, page, discrepancy_count,
                     reflect_failures, low_confidence_count, duel_disagreements)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (document_id, section_id, page) DO UPDATE SET
                    discrepancy_count = EXCLUDED.discrepancy_count,
                    reflect_failures = EXCLUDED.reflect_failures,
                    low_confidence_count = EXCLUDED.low_confidence_count,
                    duel_disagreements = EXCLUDED.duel_disagreements,
                    updated_at = NOW()
                """,
                cell.document_id,
                cell.section_id,
                storage_page,
                cell.discrepancy_count,
                cell.reflect_failures,
                cell.low_confidence_count,
                cell.duel_disagreements,
            )
            for artifact_id in cell.drill_down_artifact_ids:
                await db.execute(
                    """
                    INSERT INTO cyrex.pressure_cell_artifacts
                        (document_id, section_id, page, artifact_id)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT DO NOTHING
                    """,
                    cell.document_id,
                    cell.section_id,
                    storage_page,
                    artifact_id,
                )
            cells.append(cell)
        return cells
