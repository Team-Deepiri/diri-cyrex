"""Voice of the Document — corrections stage.

Accepts a human correction for an extracted field and persists it as a
LearningArtifact via CorrectionWriterPort.
"""

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


class CorrectionStage(CorrectionWriterPort):
    """Submits a human correction and returns a LearningArtifact bundle."""

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
            original_value=None,  # filled when PostgresArtifactStore look-up
            # is wired into corrections UI
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

    @property
    def corrections(self) -> list[LearningArtifact]:
        """Read-only access to stored corrections for tests and inspection."""
        return list(self._corrections)
