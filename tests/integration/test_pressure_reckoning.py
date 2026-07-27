"""Integration coverage for the PostgreSQL pressure and reckoning read models."""

from __future__ import annotations

from unittest.mock import AsyncMock

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

# UUID strings aligned with platform cyrex migrations (document_id / artifact_id).
DOC_ID = "11111111-1111-4111-8111-111111111111"
ART_001 = "22222222-2222-4222-8222-222222222221"
ART_003 = "22222222-2222-4222-8222-222222222223"
ART_DUEL = "22222222-2222-4222-8222-222222222224"


class ReckoningDatabase:
    """Async database double exposing only the read interface under test."""

    def __init__(self) -> None:
        self.fetch = AsyncMock(
            return_value=[
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
        )


class PressureDatabase:
    """Async database double for pressure persistence and read operations."""

    def __init__(self) -> None:
        self.execute = AsyncMock()
        self.fetch = AsyncMock(
            return_value=[
                {
                    "document_id": DOC_ID,
                    "section_id": "financial_terms",
                    "page": 1,
                    "score": 0.75,
                    "is_fault_zone": True,
                    "cell_json": {
                        "document_id": DOC_ID,
                        "section_id": "financial_terms",
                        "page": 1,
                        "score": 0.75,
                        "is_fault_zone": True,
                        "discrepancy_count": 1,
                        "reflect_failures": 0,
                        "low_confidence_count": 1,
                        "duel_disagreements": 1,
                        "drill_down_artifact_ids": [ART_001, ART_003, ART_DUEL],
                    },
                    "discrepancy_count": 1,
                    "reflect_failures": 0,
                    "low_confidence_count": 1,
                    "duel_disagreements": 1,
                    "artifact_ids": [ART_001, ART_003, ART_DUEL],
                }
            ]
        )


@pytest.mark.asyncio
async def test_reckoning_and_pressure_database_read_models():
    reckoning_db = ReckoningDatabase()
    pressure_db = PressureDatabase()

    reckoning = await PostgresReckoningStore(reckoning_db).get_reckoning(DOC_ID)
    assert reckoning[0].field_name == "base_rent"
    assert reckoning[0].actual_value == 4600
    assert reckoning[0].sigma_delta == 1.4

    events = [
        PassDiscrepancy(
            document_id=DOC_ID,
            section_id="financial_terms",
            page=1,
            artifact_id=ART_001,
            field_name="base_rent",
        ),
        DuelDisagreement(
            document_id=DOC_ID,
            section_id="financial_terms",
            page=1,
            artifact_id=ART_DUEL,
            field_name="notice_period",
        ),
        LowConfidenceField(
            document_id=DOC_ID,
            section_id="financial_terms",
            page=1,
            artifact_id=ART_003,
            field_name="maintenance_obligation",
            confidence=0.52,
        ),
    ]
    await PressureEngine(pressure_db).accept_many(events)
    assert pressure_db.execute.await_count == 8

    pressure = PostgresPressureStore(pressure_db)
    cells = await pressure.get_pressure(DOC_ID)
    assert len(cells) == 1
    assert cells[0].score == pytest.approx(0.75)
    assert cells[0].is_fault_zone is True
    assert cells[0].low_confidence_count == 1
    assert cells[0].drill_down_artifact_ids == [ART_001, ART_003, ART_DUEL]

    response = await get_document_pressure(DOC_ID, store=pressure)
    assert response.fault_zone_count == 1
    assert response.max_score == pytest.approx(0.75)
