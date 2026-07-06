"""Tests for AnticipateStage."""

from __future__ import annotations

from datetime import datetime

import pytest

from app.pipeline.contracts.models import PredictionRecord, PredictionStatus
from app.pipeline.stages.anticipate import (
    AnticipateStage,
    InMemoryPriorLookup,
    default_lease_prior_lookup,
    prior_only,
)
from tests.fakes.anticipate import FakeAnticipate


class TestPriorOnly:
    def test_strips_post_extraction_fields(self):
        full = PredictionRecord(
            field_name="base_rent",
            predicted_range={"min": 3800.0, "max": 5200.0},
            predicted_mean=4500.0,
            actual_value=4500.0,
            sigma_delta=0.0,
            status=PredictionStatus.CONFIRMED,
            corpus_doc_count=147,
            last_prior_update=datetime.fromisoformat("2024-06-10T08:00:00+00:00"),
        )
        stripped = prior_only(full)
        assert stripped.actual_value is None
        assert stripped.sigma_delta is None
        assert stripped.status == PredictionStatus.NO_PRIOR
        assert stripped.predicted_mean == 4500.0


class TestAnticipateStage:
    @pytest.mark.asyncio
    async def test_unknown_document_class_returns_empty(self):
        stage = AnticipateStage()
        result = await stage.run(parsed_doc={}, document_class="unknown")
        assert result == []

    @pytest.mark.asyncio
    async def test_lease_returns_all_template_fields(self):
        stage = AnticipateStage()
        result = await stage.run(parsed_doc={}, document_class="lease")
        assert len(result) == 4
        assert [r.field_name for r in result] == [
            "base_rent",
            "security_deposit",
            "notice_period",
            "custom_addendum",
        ]

    @pytest.mark.asyncio
    async def test_lease_priors_match_fixture_shape(self):
        stage = AnticipateStage(prior_lookup=default_lease_prior_lookup())
        result = await stage.run(parsed_doc={}, document_class="lease")

        base_rent = next(r for r in result if r.field_name == "base_rent")
        assert base_rent.predicted_mean == 4500.0
        assert base_rent.predicted_range == {"min": 3800.0, "max": 5200.0}
        assert base_rent.corpus_doc_count == 147
        assert base_rent.actual_value is None
        assert base_rent.status == PredictionStatus.NO_PRIOR

        no_prior = next(r for r in result if r.field_name == "custom_addendum")
        assert no_prior.predicted_mean is None
        assert no_prior.predicted_range is None
        assert no_prior.corpus_doc_count == 0
        assert no_prior.status == PredictionStatus.NO_PRIOR

    @pytest.mark.asyncio
    async def test_custom_prior_lookup(self):
        lookup = InMemoryPriorLookup(
            {
                ("invoice", "total"): PredictionRecord(
                    field_name="total",
                    predicted_mean=100.0,
                    corpus_doc_count=10,
                ),
            }
        )
        stage = AnticipateStage(
            prior_lookup=lookup,
            field_templates={"invoice": ["total"]},
        )
        result = await stage.run(parsed_doc={}, document_class="invoice")
        assert len(result) == 1
        assert result[0].field_name == "total"
        assert result[0].predicted_mean == 100.0
        assert result[0].status == PredictionStatus.NO_PRIOR

    @pytest.mark.asyncio
    async def test_parsed_doc_ignored_in_v1(self):
        stage = AnticipateStage()
        with_doc = await stage.run(parsed_doc={"sections": ["ignored"]}, document_class="lease")
        without_doc = await stage.run(parsed_doc=None, document_class="lease")
        assert len(with_doc) == len(without_doc)


class TestAnticipatePortCompliance:
    def test_stage_implements_port_methods(self):
        stage = AnticipateStage()
        assert hasattr(stage, "run")
        assert stage.run.__name__ == "run"

    @pytest.mark.asyncio
    async def test_stage_differs_from_noop_fake(self):
        stage = AnticipateStage()
        fake = FakeAnticipate()
        stage_result = await stage.run(parsed_doc={}, document_class="lease")
        fake_result = await fake.run(parsed_doc={}, document_class="lease")
        assert stage_result != fake_result
        assert len(stage_result) > 0

    def test_anticipate_port_is_protocol(self):
        stage = AnticipateStage()
        assert hasattr(stage, "run")
        assert callable(stage.run)
