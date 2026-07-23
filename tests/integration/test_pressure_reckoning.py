"""Integration coverage for the PostgreSQL pressure and reckoning read models."""

from __future__ import annotations

import json

import pytest

from app.pipeline.contracts.pressure_events import (
    DuelDisagreement,
    LowConfidenceField,
    PassDiscrepancy,
)
from app.pipeline.pressure.engine import PressureEngine
from app.pipeline.registry.pressure_store import PostgresPressureStore
from app.pipeline.registry.reckoning_store import PostgresReckoningStore
from app.routes.pressure import get_document_pressure


class DatabaseShape:
    """Minimal asyncpg-compatible store for the integration workflow."""

    def __init__(self) -> None:
        self.reckoning_rows = [
            {
                "field_name": "base_rent",
                "record_json": {"predicted_mean": 4500},
                "status": "anomalous",
                "actual_value": "4600",
                "sigma_delta": 1.4,
                "predicted_range_json": {"min": 4000, "max": 5000},
                "last_prior_update": None,
                "corpus_doc_count": 12,
            }
        ]
        self.cells: dict[tuple[str, str, int], dict] = {}
        self.metrics: dict[tuple[str, str, int], dict] = {}
        self.artifacts: dict[tuple[str, str, int], set[str]] = {}

    async def execute(self, query: str, *args):
        if "INSERT INTO cyrex.pressure_cells" in query:
            key = (args[0], args[1], args[2])
            self.cells[key] = {
                "document_id": args[0],
                "section_id": args[1],
                "page": args[2],
                "score": args[3],
                "is_fault_zone": args[4],
                "cell_json": json.loads(args[5]),
            }
        elif "INSERT INTO cyrex.pressure_cell_metrics" in query:
            key = (args[0], args[1], args[2])
            self.metrics[key] = {
                "discrepancy_count": args[3],
                "reflect_failures": args[4],
                "low_confidence_count": args[5],
                "duel_disagreements": args[6],
            }
        elif "INSERT INTO cyrex.pressure_cell_artifacts" in query:
            key = (args[0], args[1], args[2])
            self.artifacts.setdefault(key, set()).add(args[3])

    async def fetch(self, query: str, *args):
        if "FROM cyrex.reckoning_records" in query:
            return self.reckoning_rows
        if "FROM cyrex.pressure_cells" in query:
            document_id = args[0] if args else None
            rows = []
            for key, cell in self.cells.items():
                if document_id is not None and key[0] != document_id:
                    continue
                rows.append(
                    {
                        **cell,
                        **self.metrics[key],
                        "artifact_ids": sorted(self.artifacts.get(key, set())),
                    }
                )
            return rows
        raise AssertionError(f"Unexpected query: {query}")


@pytest.mark.asyncio
async def test_reckoning_and_pressure_database_read_models():
    db = DatabaseShape()

    reckoning = await PostgresReckoningStore(db).get_reckoning("lease_001")
    assert reckoning[0].field_name == "base_rent"
    assert reckoning[0].actual_value == 4600
    assert reckoning[0].sigma_delta == 1.4

    events = [
        PassDiscrepancy(
            document_id="lease_001",
            section_id="financial_terms",
            page=1,
            artifact_id="art_001",
            field_name="base_rent",
        ),
        DuelDisagreement(
            document_id="lease_001",
            section_id="financial_terms",
            page=1,
            artifact_id="art_duel_001",
            field_name="notice_period",
        ),
        LowConfidenceField(
            document_id="lease_001",
            section_id="financial_terms",
            page=1,
            artifact_id="art_003",
            field_name="maintenance_obligation",
            confidence=0.52,
        ),
    ]
    await PressureEngine(db).accept_many(events)

    pressure = PostgresPressureStore(db)
    cells = await pressure.get_pressure("lease_001")
    assert len(cells) == 1
    assert cells[0].score == pytest.approx(0.75)
    assert cells[0].is_fault_zone is True
    assert cells[0].low_confidence_count == 1
    assert cells[0].drill_down_artifact_ids == ["art_001", "art_003", "art_duel_001"]

    response = await get_document_pressure("lease_001", store=pressure)
    assert response.fault_zone_count == 1
    assert response.max_score == pytest.approx(0.75)
