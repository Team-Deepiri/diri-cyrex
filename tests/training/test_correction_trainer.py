"""Tests for correction buffer."""

from app.pipeline.contracts.models import Citation, CitationLocator, LearningArtifact
from app.training.correction_trainer import CorrectionTrainer


def _sample_artifact(document_id: str = "doc_1") -> LearningArtifact:
    citation = Citation(
        document_id=document_id,
        source_doc_hash="hash",
        locator=CitationLocator(locator_type="char_range", char_start=0, char_end=5),
        quote="hello",
        confidence=0.9,
    )
    return LearningArtifact(
        document_id=document_id,
        field_name="amount",
        original_value=100,
        corrected_value=120,
        corrected_citation=citation,
        actor_id="user_1",
    )


def test_buffer_and_peek():
    trainer = CorrectionTrainer(max_buffer=10)
    artifact = _sample_artifact()
    trainer.buffer(artifact)
    assert trainer.size == 1
    peeked = trainer.peek()
    assert len(peeked) == 1
    assert peeked[0].corrected_value == 120


def test_drain_clears_buffer():
    trainer = CorrectionTrainer()
    trainer.buffer(_sample_artifact("a"))
    trainer.buffer(_sample_artifact("b"))
    drained = trainer.drain()
    assert len(drained) == 2
    assert trainer.size == 0


def test_export_jsonl_lines():
    trainer = CorrectionTrainer()
    trainer.buffer(_sample_artifact())
    lines = list(trainer.export_jsonl_lines())
    assert len(lines) == 1
    assert '"field_name":"amount"' in lines[0]
