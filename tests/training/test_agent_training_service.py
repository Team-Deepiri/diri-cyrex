"""Tests for agent training service."""

from app.pipeline.contracts.models import Citation, CitationLocator
from app.training.agent_training_service import AgentTrainingService


def test_submit_correction_buffers_artifact():
    service = AgentTrainingService()
    citation = Citation(
        document_id="doc_1",
        source_doc_hash="hash",
        locator=CitationLocator(locator_type="char_range", char_start=0, char_end=3),
        quote="foo",
        confidence=0.8,
    )
    artifact = service.submit_correction(
        document_id="doc_1",
        field_name="name",
        original_value="old",
        corrected_value="new",
        corrected_citation=citation,
        actor_id="tester",
    )
    assert artifact.corrected_value == "new"
    assert service.correction_trainer.size == 1


def test_feedback_loop_enqueues_or_stub():
    service = AgentTrainingService()
    citation = Citation(
        document_id="doc_1",
        source_doc_hash="hash",
        locator=CitationLocator(locator_type="char_range", char_start=0, char_end=3),
        quote="foo",
        confidence=0.8,
    )
    artifact = service.submit_correction(
        document_id="doc_1",
        field_name="name",
        original_value="old",
        corrected_value="new value text",
        corrected_citation=citation,
        actor_id="tester",
    )
    result = service.start_feedback_loop([artifact.artifact_id])
    assert "correlation_id" in result
    assert result["status"] in ("enqueued", "stub", "buffered")
