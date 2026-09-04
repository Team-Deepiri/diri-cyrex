"""Tests for ReckoningStage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import AsyncMock

import pytest

from app.pipeline.contracts.models import (
    Citation,
    CitationLocator,
    CitedField,
    LearningArtifact,
    PredictionRecord,
    PredictionStatus,
)
from app.pipeline.stages.anticipate import AnticipateStage, default_lease_prior_lookup
from app.pipeline.stages.reckoning import (
    ReckoningStage,
    emit_learning_artifacts,
    emit_reckoning_training,
)
from tests.fakes.reckoning import FakeReckoningRead

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "cyrex_contracts"
GOLDEN_RECORDS = json.loads(
    (FIXTURES_DIR / "prediction_records.json").read_text(encoding="utf-8")
)


def _field(name: str, value: Any, confidence: float = 0.9) -> CitedField:
    return CitedField(field_name=name, value=value, confidence=confidence)


def _prior(
    name: str,
    *,
    mean: Optional[float] = None,
    range_min: Optional[float] = None,
    range_max: Optional[float] = None,
    corpus_doc_count: int = 147,
) -> PredictionRecord:
    predicted_range = None
    if range_min is not None and range_max is not None:
        predicted_range = {"min": range_min, "max": range_max}
    return PredictionRecord(
        field_name=name,
        predicted_mean=mean,
        predicted_range=predicted_range,
        status=PredictionStatus.NO_PRIOR,
        corpus_doc_count=corpus_doc_count,
    )


class TestReckoningConfirmed:
    @pytest.mark.asyncio
    async def test_confirmed_within_threshold(self):
        priors = [
            _prior("base_rent", mean=4500.0, range_min=3800.0, range_max=5200.0)
        ]
        extracted = [_field("base_rent", 4500)]
        stage = ReckoningStage()
        result = await stage.run(priors, extracted)
        assert len(result) == 1
        assert result[0].status == PredictionStatus.CONFIRMED
        assert result[0].sigma_delta == pytest.approx(0.0)
        assert result[0].actual_value == 4500


class TestReckoningAnomalous:
    @pytest.mark.asyncio
    async def test_anomalous_outside_threshold(self):
        priors = [
            _prior("notice_period", mean=90.0, range_min=60.0, range_max=120.0)
        ]
        extracted = [_field("notice_period", 60)]
        stage = ReckoningStage()
        result = await stage.run(priors, extracted)
        assert len(result) == 1
        assert result[0].status == PredictionStatus.ANOMALOUS
        assert result[0].sigma_delta == pytest.approx(-0.5)
        assert result[0].actual_value == 60


class TestReckoningNovelAndMissing:
    @pytest.mark.asyncio
    async def test_novel_field_no_prior(self):
        priors: List[PredictionRecord] = []
        extracted = [_field("lease_start", "2024-01-01")]
        stage = ReckoningStage()
        result = await stage.run(priors, extracted)
        assert len(result) == 1
        assert result[0].status == PredictionStatus.NOVEL
        assert result[0].sigma_delta is None
        assert result[0].actual_value == "2024-01-01"
        assert result[0].predicted_mean is None

    @pytest.mark.asyncio
    async def test_missing_extraction_keeps_prior_unchanged(self):
        priors = [
            PredictionRecord(
                field_name="custom_addendum",
                status=PredictionStatus.NO_PRIOR,
                corpus_doc_count=0,
            )
        ]
        stage = ReckoningStage()
        result = await stage.run(priors, extracted_fields=[])
        assert len(result) == 1
        assert result[0].field_name == "custom_addendum"
        assert result[0].status == PredictionStatus.NO_PRIOR
        assert result[0].actual_value is None
        assert result[0].sigma_delta is None


class TestGoldenFixture:
    @pytest.mark.asyncio
    async def test_reproduces_golden_fixture(self):
        anticipate = AnticipateStage(prior_lookup=default_lease_prior_lookup())
        priors = await anticipate.run(parsed_doc={}, document_class="lease")
        extracted = [
            _field("base_rent", 4500.0),
            _field("security_deposit", 5000.0),
            _field("notice_period", 60.0),
        ]
        stage = ReckoningStage()
        result = await stage.run(priors, extracted)
        by_name = {r.field_name: r for r in result}

        for expected in GOLDEN_RECORDS:
            actual = by_name[expected["field_name"]]
            assert actual.status.value == expected["status"]
            assert actual.actual_value == expected["actual_value"]
            if expected["sigma_delta"] is None:
                assert actual.sigma_delta is None
            else:
                assert actual.sigma_delta == pytest.approx(expected["sigma_delta"])
            assert actual.predicted_mean == expected["predicted_mean"]
            assert actual.predicted_range == expected["predicted_range"]
            assert actual.corpus_doc_count == expected["corpus_doc_count"]


class TestThresholdOverride:
    @pytest.mark.asyncio
    async def test_threshold_param_override(self):
        priors = [
            _prior("notice_period", mean=90.0, range_min=60.0, range_max=120.0)
        ]
        extracted = [_field("notice_period", 60)]
        # Default threshold 0.3 → | -0.5 | > 0.3 → anomalous
        default_result = await ReckoningStage().run(priors, extracted)
        assert default_result[0].status == PredictionStatus.ANOMALOUS

        # Raised threshold 0.6 → | -0.5 | <= 0.6 → confirmed
        loose_result = await ReckoningStage(
            anomaly_sigma_threshold=0.6
        ).run(priors, extracted)
        assert loose_result[0].status == PredictionStatus.CONFIRMED
        assert loose_result[0].sigma_delta == pytest.approx(-0.5)


class TestReckoningDiffersFromFake:
    @pytest.mark.asyncio
    async def test_differs_from_fake_reckoning(self):
        priors = [
            _prior("base_rent", mean=4500.0, range_min=3800.0, range_max=5200.0)
        ]
        extracted = [_field("base_rent", 4500)]
        stage = ReckoningStage()
        stage_result = await stage.run(priors, extracted)

        fake = FakeReckoningRead()
        fake_result = await fake.get_reckoning("lease_001")
        assert stage_result != fake_result
        assert len(stage_result) > len(fake_result)


class TestDegenerateInputs:
    """A prior that cannot be compared against must not report CONFIRMED.

    Regression coverage: these all previously collapsed into CONFIRMED because
    an undefined sigma_delta was treated as agreement.
    """

    @pytest.mark.asyncio
    async def test_non_numeric_actual_records_without_judging(self):
        priors = [
            _prior("base_rent", mean=4500.0, range_min=3800.0, range_max=5200.0)
        ]
        extracted = [_field("base_rent", "see addendum")]
        result = await ReckoningStage().run(priors, extracted)
        assert result[0].status == PredictionStatus.CONFIRMED
        assert result[0].sigma_delta is None
        assert result[0].actual_value == "see addendum"

    @pytest.mark.asyncio
    async def test_zero_width_range_falls_back_to_mean_comparison(self):
        priors = [
            _prior("base_rent", mean=4500.0, range_min=4500.0, range_max=4500.0)
        ]
        result = await ReckoningStage().run(priors, [_field("base_rent", 4500)])
        assert result[0].status == PredictionStatus.CONFIRMED
        assert result[0].sigma_delta is None

    @pytest.mark.asyncio
    async def test_zero_width_range_flags_a_value_off_the_mean(self):
        priors = [
            _prior("base_rent", mean=4500.0, range_min=4500.0, range_max=4500.0)
        ]
        result = await ReckoningStage().run(priors, [_field("base_rent", 99000)])
        assert result[0].status == PredictionStatus.ANOMALOUS
        assert result[0].sigma_delta is None

    @pytest.mark.asyncio
    async def test_mean_without_range_flags_a_value_off_the_mean(self):
        priors = [_prior("base_rent", mean=4500.0)]
        result = await ReckoningStage().run(priors, [_field("base_rent", 12000)])
        assert result[0].status == PredictionStatus.ANOMALOUS

    @pytest.mark.asyncio
    async def test_range_without_mean_judges_containment(self):
        priors = [_prior("base_rent", range_min=3800.0, range_max=5200.0)]

        inside = await ReckoningStage().run(priors, [_field("base_rent", 4000)])
        assert inside[0].status == PredictionStatus.CONFIRMED

        outside = await ReckoningStage().run(priors, [_field("base_rent", 9000)])
        assert outside[0].status == PredictionStatus.ANOMALOUS


class TestResultDoesNotAliasPriors:
    @pytest.mark.asyncio
    async def test_passthrough_prior_is_copied(self):
        prior = PredictionRecord(
            field_name="custom_addendum",
            predicted_range={"min": 1.0, "max": 2.0},
            status=PredictionStatus.NO_PRIOR,
            corpus_doc_count=0,
        )
        result = await ReckoningStage().run([prior], extracted_fields=[])

        assert result[0] is not prior
        result[0].status = PredictionStatus.ANOMALOUS
        result[0].predicted_range["min"] = 999.0
        assert prior.status == PredictionStatus.NO_PRIOR
        assert prior.predicted_range == {"min": 1.0, "max": 2.0}


class TestReckoningPortCompliance:
    def test_stage_implements_run(self):
        stage = ReckoningStage()
        assert hasattr(stage, "run")
        assert callable(stage.run)


class TestTrainingBridge:
    @pytest.mark.asyncio
    async def test_training_bridge_calls_emit_correction(self):
        citation = Citation(
            document_id="lease_001",
            source_doc_hash="sha256:abc",
            locator=CitationLocator(locator_type="char_range", char_start=0, char_end=5),
            quote="4500",
            confidence=1.0,
        )
        artifacts = [
            LearningArtifact(
                document_id="lease_001",
                field_name="base_rent",
                original_value=4400,
                corrected_value=4500,
                corrected_citation=citation,
                actor_id="user_1",
            ),
            LearningArtifact(
                document_id="lease_001",
                field_name="notice_period",
                original_value=90,
                corrected_value=60,
                corrected_citation=citation,
                actor_id="user_1",
            ),
        ]
        emitter = AsyncMock()
        emitter.emit_correction = AsyncMock(side_effect=["rid_1", "rid_2"])

        record_ids = await emit_learning_artifacts(artifacts, emitter)

        assert record_ids == ["rid_1", "rid_2"]
        assert emitter.emit_correction.await_count == 2
        first_call = emitter.emit_correction.await_args_list[0].kwargs
        assert "base_rent" in first_call["instruction"]
        assert first_call["corrected_output"] == "4500"
        assert first_call["document_id"] == "lease_001"
        assert first_call["metadata"]["field_name"] == "base_rent"
        assert first_call["metadata"]["actor_id"] == "user_1"

    @pytest.mark.asyncio
    async def test_one_failed_artifact_does_not_drop_the_rest(self):
        citation = Citation(
            document_id="lease_001",
            source_doc_hash="sha256:abc",
            locator=CitationLocator(locator_type="char_range", char_start=0, char_end=5),
            quote="4500",
            confidence=1.0,
        )
        artifacts = [
            LearningArtifact(
                document_id="lease_001",
                field_name=name,
                original_value=1,
                corrected_value=2,
                corrected_citation=citation,
                actor_id="user_1",
            )
            for name in ("base_rent", "notice_period", "security_deposit")
        ]
        emitter = AsyncMock()
        emitter.emit_correction = AsyncMock(
            side_effect=["rid_1", RuntimeError("postgres down"), "rid_3"]
        )

        record_ids = await emit_learning_artifacts(artifacts, emitter)

        assert record_ids == ["rid_1", "rid_3"]
        assert emitter.emit_correction.await_count == 3


class TestReckoningTrainingEmit:
    @pytest.mark.asyncio
    async def test_emit_reckoning_training_anomalous_and_novel(self):
        records = [
            PredictionRecord(
                field_name="base_rent",
                predicted_mean=4500.0,
                predicted_range={"min": 3800.0, "max": 5200.0},
                actual_value=9000,
                sigma_delta=3.0,
                status=PredictionStatus.ANOMALOUS,
            ),
            PredictionRecord(
                field_name="lease_start",
                actual_value="2024-01-01",
                status=PredictionStatus.NOVEL,
            ),
            PredictionRecord(
                field_name="notice_period",
                predicted_mean=90.0,
                predicted_range={"min": 60.0, "max": 120.0},
                actual_value=90,
                sigma_delta=0.0,
                status=PredictionStatus.CONFIRMED,
            ),
        ]
        emitter = AsyncMock()
        emitter.emit_structured = AsyncMock(side_effect=["rid_a", "rid_n"])

        emitted = await emit_reckoning_training(
            records,
            document_id="doc_1",
            artifact_id="art_1",
            emitter=emitter,
        )

        assert emitted == ["rid_a", "rid_n"]
        assert emitter.emit_structured.await_count == 2
        first = emitter.emit_structured.await_args_list[0].kwargs
        assert first["category"] == "reckoning"
        assert first["document_id"] == "doc_1"
        assert first["metadata"]["status"] == "anomalous"
