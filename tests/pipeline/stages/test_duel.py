"""Tests for DuelStage."""

from __future__ import annotations

import pytest

from app.pipeline.contracts.models import (
    CitedField,
    DuelResolutionStatus,
    DuelState,
    Provenance,
    SynthesisResult,
)
from app.pipeline.contracts.ports import ExtractPort
from app.pipeline.stages.duel import DuelStage, to_arena_rows
from tests.fakes.duel import NoOpDuelRunner
from tests.fakes.extract import FixedExtract


def _synthesis(
    document_id: str, source_doc_hash: str, fields: list[CitedField]
) -> SynthesisResult:
    return SynthesisResult(
        document_id=document_id,
        source_doc_hash=source_doc_hash,
        final_fields=fields,
        confidence=sum(f.confidence for f in fields) / len(fields) if fields else 0.0,
        provenance=Provenance(source_doc_hash=source_doc_hash, document_id=document_id),
    )


def _extract_stub(fields: list[CitedField]) -> FixedExtract:
    return FixedExtract(
        _synthesis("lease_001", "sha256:a1b2c3d4e5f6", fields)
    )


class _RaisingExtract(ExtractPort):
    """Simulates an agent that throws instead of returning a SynthesisResult."""

    async def run(self, parsed_doc: object, document_id: str, source_doc_hash: str):
        del parsed_doc, document_id, source_doc_hash
        raise RuntimeError("agent backend unavailable")


class TestDuelStageAgreement:
    @pytest.mark.asyncio
    async def test_identical_agents_no_disagreements(self):
        fields = [
            CitedField(field_name="base_rent", value=4500, confidence=0.95),
            CitedField(field_name="notice_period", value=90, confidence=0.88),
        ]
        stage = DuelStage(
            agent_a=_extract_stub(fields),
            agent_b=_extract_stub(fields),
        )
        result = await stage.run(
            parsed_doc={"raw_text": "irrelevant"},
            document_id="lease_001",
            source_doc_hash="sha256:a1b2c3d4e5f6",
        )
        assert isinstance(result, DuelState)
        assert result.disagreements == []


class TestDuelStageConflict:
    @pytest.mark.asyncio
    async def test_conflicting_fields_emit_discrepancy(self):
        fields_a = [
            CitedField(field_name="base_rent", value=4500, confidence=0.95),
            CitedField(field_name="notice_period", value=90, confidence=0.88),
        ]
        fields_b = [
            CitedField(field_name="base_rent", value=4500, confidence=0.93),
            CitedField(field_name="notice_period", value=60, confidence=0.80),
        ]
        stage = DuelStage(
            agent_a=_extract_stub(fields_a),
            agent_b=_extract_stub(fields_b),
            agent_a_id="agent_a_llama",
            agent_b_id="agent_b_gpt",
        )
        result = await stage.run(
            parsed_doc={"raw_text": "irrelevant"},
            document_id="lease_001",
            source_doc_hash="sha256:a1b2c3d4e5f6",
        )
        assert len(result.disagreements) == 1
        disagreement = result.disagreements[0]
        assert disagreement.field_name == "notice_period"
        assert disagreement.confidence_delta == pytest.approx(0.08)
        assert disagreement.agent_a_value == 90
        assert disagreement.agent_b_value == 60

    @pytest.mark.asyncio
    async def test_agent_field_values_populated(self):
        fields_a = [CitedField(field_name="notice_period", value=90, confidence=0.88)]
        fields_b = [CitedField(field_name="notice_period", value=60, confidence=0.80)]
        stage = DuelStage(agent_a=_extract_stub(fields_a), agent_b=_extract_stub(fields_b))
        result = await stage.run(
            parsed_doc={"raw_text": "irrelevant"},
            document_id="lease_001",
            source_doc_hash="sha256:a1b2c3d4e5f6",
        )
        disagreement = result.disagreements[0]
        assert disagreement.agent_a_value == 90
        assert disagreement.agent_b_value == 60
        assert disagreement.pass_a_value is None
        assert disagreement.pass_b_value is None

    @pytest.mark.asyncio
    async def test_one_sided_field_emits_null_other_side(self):
        fields_a = [CitedField(field_name="security_deposit", value=1000, confidence=0.9)]
        fields_b: list[CitedField] = []
        stage = DuelStage(agent_a=_extract_stub(fields_a), agent_b=_extract_stub(fields_b))
        result = await stage.run(
            parsed_doc={"raw_text": "irrelevant"},
            document_id="lease_001",
            source_doc_hash="sha256:a1b2c3d4e5f6",
        )
        assert len(result.disagreements) == 1
        disagreement = result.disagreements[0]
        assert disagreement.agent_a_value == 1000
        assert disagreement.agent_b_value is None
        assert disagreement.agent_b_confidence is None
        assert disagreement.confidence_delta is None


class TestDuelStateAssembly:
    @pytest.mark.asyncio
    async def test_resolution_unresolved(self):
        stage = DuelStage(agent_a=_extract_stub([]), agent_b=_extract_stub([]))
        result = await stage.run(
            parsed_doc={"raw_text": "irrelevant"},
            document_id="lease_001",
            source_doc_hash="sha256:a1b2c3d4e5f6",
        )
        assert result.resolution_status == DuelResolutionStatus.UNRESOLVED

    @pytest.mark.asyncio
    async def test_agent_ids_and_fields_carried_through(self):
        fields_a = [CitedField(field_name="base_rent", value=4500, confidence=0.95)]
        fields_b = [CitedField(field_name="base_rent", value=4500, confidence=0.93)]
        stage = DuelStage(
            agent_a=_extract_stub(fields_a),
            agent_b=_extract_stub(fields_b),
            agent_a_id="agent_a_llama",
            agent_b_id="agent_b_gpt",
        )
        result = await stage.run(
            parsed_doc={"raw_text": "irrelevant"},
            document_id="lease_001",
            source_doc_hash="sha256:a1b2c3d4e5f6",
        )
        assert result.agent_a_id == "agent_a_llama"
        assert result.agent_b_id == "agent_b_gpt"
        assert result.agent_a_fields == fields_a
        assert result.agent_b_fields == fields_b

    @pytest.mark.asyncio
    async def test_differs_from_fake_duel(self):
        fields_a = [CitedField(field_name="notice_period", value=90, confidence=0.88)]
        fields_b = [CitedField(field_name="notice_period", value=60, confidence=0.80)]
        stage = DuelStage(agent_a=_extract_stub(fields_a), agent_b=_extract_stub(fields_b))
        fake = NoOpDuelRunner()

        stage_result = await stage.run(
            parsed_doc={"raw_text": "irrelevant"},
            document_id="lease_001",
            source_doc_hash="sha256:a1b2c3d4e5f6",
        )
        fake_result = await fake.run(
            parsed_doc={"raw_text": "irrelevant"},
            document_id="lease_001",
            source_doc_hash="sha256:a1b2c3d4e5f6",
        )
        assert stage_result != fake_result
        assert len(stage_result.disagreements) > len(fake_result.disagreements)


class TestDuelStageAgentFailure:
    @pytest.mark.asyncio
    async def test_agent_a_exception_degrades_instead_of_raising(self):
        fields_b = [CitedField(field_name="notice_period", value=60, confidence=0.80)]
        stage = DuelStage(agent_a=_RaisingExtract(), agent_b=_extract_stub(fields_b))
        result = await stage.run(
            parsed_doc={"raw_text": "irrelevant"},
            document_id="lease_001",
            source_doc_hash="sha256:a1b2c3d4e5f6",
        )
        assert isinstance(result, DuelState)
        assert result.agent_a_fields == []
        assert result.agent_b_fields == fields_b
        assert len(result.disagreements) == 1
        assert result.disagreements[0].agent_a_value is None
        assert result.disagreements[0].agent_b_value == 60

    @pytest.mark.asyncio
    async def test_agent_b_exception_degrades_instead_of_raising(self):
        fields_a = [CitedField(field_name="notice_period", value=90, confidence=0.88)]
        stage = DuelStage(agent_a=_extract_stub(fields_a), agent_b=_RaisingExtract())
        result = await stage.run(
            parsed_doc={"raw_text": "irrelevant"},
            document_id="lease_001",
            source_doc_hash="sha256:a1b2c3d4e5f6",
        )
        assert isinstance(result, DuelState)
        assert result.agent_a_fields == fields_a
        assert result.agent_b_fields == []

    @pytest.mark.asyncio
    async def test_both_agents_raise_returns_empty_duel_state(self):
        stage = DuelStage(agent_a=_RaisingExtract(), agent_b=_RaisingExtract())
        result = await stage.run(
            parsed_doc={"raw_text": "irrelevant"},
            document_id="lease_001",
            source_doc_hash="sha256:a1b2c3d4e5f6",
        )
        assert result.agent_a_fields == []
        assert result.agent_b_fields == []
        assert result.disagreements == []
        assert result.resolution_status == DuelResolutionStatus.UNRESOLVED


class TestArenaRowsHelper:
    def test_arena_rows_match_viz_shape(self):
        state = DuelState(
            document_id="lease_001",
            agent_a_id="agent_a_llama",
            agent_b_id="agent_b_gpt",
            agent_a_fields=[
                CitedField(field_name="base_rent", value=4500, confidence=0.95),
                CitedField(field_name="notice_period", value=90, confidence=0.88),
            ],
            agent_b_fields=[
                CitedField(field_name="base_rent", value=4500, confidence=0.93),
                CitedField(field_name="notice_period", value=60, confidence=0.80),
            ],
            disagreements=[],
        )
        rows = to_arena_rows(state)
        row_by_name = {row["field_name"]: row for row in rows}
        assert row_by_name["base_rent"]["agent_a_value"] == 4500
        assert row_by_name["base_rent"]["agent_b_value"] == 4500
        assert row_by_name["notice_period"]["agent_a_confidence"] == 0.88
        assert row_by_name["notice_period"]["agent_b_confidence"] == 0.80


class TestDuelPortCompliance:
    def test_stage_implements_port_methods(self):
        stage = DuelStage(agent_a=_extract_stub([]), agent_b=_extract_stub([]))
        assert hasattr(stage, "run")
        assert callable(stage.run)
