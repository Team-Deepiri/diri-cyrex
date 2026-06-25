"""Round-trip serialize/deserialize tests for all golden contract fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.pipeline.contracts.models import (
    ArtifactBundle,
    DuelState,
    PredictionRecord,
    PressureCell,
    ReflectionResult,
    SynthesisResult,
)
from app.pipeline.contracts.pressure_events import (
    DuelDisagreement,
    LowConfidenceField,
    PassDiscrepancy,
    PressureEvent,
    ReflectFailure,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "cyrex_contracts"


def _load(name: str):
    with open(FIXTURE_DIR / name) as f:
        return json.load(f)


def _records(data) -> list:
    return data if isinstance(data, list) else data["records"]


class TestArtifactBundleRoundtrip:
    def test_full_bundle(self):
        data = _load("artifact_bundle_full.json")
        bundle = ArtifactBundle.model_validate(data)
        restored = ArtifactBundle.model_validate(bundle.model_dump(mode="json"))
        assert restored.artifact_id == bundle.artifact_id
        assert restored.artifact_type == bundle.artifact_type
        assert restored.provenance.passes[0].method.value == "llm"


class TestDuelStateRoundtrip:
    def test_duel_state(self):
        data = _load("duel_state.json")
        state = DuelState.model_validate(data)
        restored = DuelState.model_validate(state.model_dump(mode="json"))
        assert restored.agent_a_id == state.agent_a_id
        assert len(restored.disagreements) == len(state.disagreements)


class TestPredictionRecordRoundtrip:
    def test_prediction_records(self):
        data = _load("prediction_records.json")
        records = [PredictionRecord.model_validate(r) for r in _records(data)]
        for rec in records:
            restored = PredictionRecord.model_validate(rec.model_dump(mode="json"))
            assert restored.field_name == rec.field_name
            assert restored.status == rec.status


class TestPressureCellRoundtrip:
    def test_pressure_cells(self):
        data = _load("pressure_cells.json")
        cells = [PressureCell.model_validate(c) for c in _records(data)]
        for cell in cells:
            restored = PressureCell.model_validate(cell.model_dump(mode="json"))
            assert restored.score == cell.score
            assert restored.is_fault_zone == cell.is_fault_zone


class TestPressureEventRoundtrip:
    def test_pressure_events(self):
        data = _load("pressure_events.json")
        type_map = {
            "pass_discrepancy": PassDiscrepancy,
            "reflect_failure": ReflectFailure,
            "low_confidence_field": LowConfidenceField,
            "duel_disagreement": DuelDisagreement,
        }
        for raw in _records(data):
            cls = type_map[raw["event_type"]]
            event = cls.model_validate(raw)
            restored = cls.model_validate(event.model_dump(mode="json"))
            assert restored.event_type == event.event_type
            assert restored.document_id == event.document_id


class TestReflectionResultRoundtrip:
    def test_reflection_result(self):
        data = _load("reflection_result.json")
        result = ReflectionResult.model_validate(data)
        restored = ReflectionResult.model_validate(result.model_dump(mode="json"))
        assert restored.passed == result.passed
        assert len(restored.issues) == len(result.issues)


class TestSynthesisResultRoundtrip:
    def test_synthesis_result(self):
        data = _load("synthesis_result.json")
        result = SynthesisResult.model_validate(data)
        restored = SynthesisResult.model_validate(result.model_dump(mode="json"))
        assert restored.document_id == result.document_id
        assert len(restored.final_fields) == len(result.final_fields)
