"""Fake CorrectionWriterPort for track-local tests."""

from __future__ import annotations

from typing import Any

from app.pipeline.contracts.models import (
    ArtifactBundle,
    ArtifactType,
    Citation,
    LearningArtifact,
    Provenance,
)
from app.pipeline.contracts.ports import CorrectionWriterPort


class FakeCorrectionWriter(CorrectionWriterPort):
    """Stores corrections in-memory for test assertions."""

    def __init__(self) -> None:
        self._corrections: list[LearningArtifact] = []

    async def submit_correction(
        self,
        artifact_id: str,
        field_name: str,
        corrected_value: Any,
        corrected_citation: Citation,
        actor_id: str,
    ) -> ArtifactBundle:
        learning = LearningArtifact(
            document_id=corrected_citation.document_id,
            field_name=field_name,
            original_value="old",
            corrected_value=corrected_value,
            corrected_citation=corrected_citation,
            actor_id=actor_id,
        )
        self._corrections.append(learning)
        return ArtifactBundle(
            artifact_id=f"learn_{artifact_id}",
            document_id=corrected_citation.document_id,
            artifact_type=ArtifactType.LEARNING,
            source_doc_hash=corrected_citation.source_doc_hash,
            confidence=1.0,
            payload={"learning_artifact": learning.model_dump(mode="json")},
            provenance=Provenance(
                source_doc_hash=corrected_citation.source_doc_hash,
                document_id=corrected_citation.document_id,
            ),
        )
