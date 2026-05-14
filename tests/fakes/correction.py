"""Fake CorrectionWriterPort for track-local tests."""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.pipeline.contracts.models import ArtifactBundle, LearningArtifact
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
        corrected_citation: Dict[str, Any],
        actor_id: str,
    ) -> ArtifactBundle:
        # Build a LearningArtifact from the correction payload
        from datetime import datetime, timezone
        from app.pipeline.contracts.models import (
            ArtifactBundle,
            ArtifactType,
            Citation,
            CitationLocator,
            Provenance,
        )
        from app.pipeline.contracts.pressure_events import LowConfidenceField

        citation = Citation(
            document_id="test_doc",
            source_doc_hash="test_hash",
            locator=CitationLocator(locator_type="char_range", char_start=0, char_end=10),
            quote=str(corrected_value)[:500],
            confidence=1.0,
        )
        learning = LearningArtifact(
            document_id="test_doc",
            field_name=field_name,
            original_value="old",
            corrected_value=corrected_value,
            corrected_citation=citation,
            actor_id=actor_id,
        )
        self._corrections.append(learning)
        return ArtifactBundle(
            artifact_id=f"learn_{artifact_id}",
            document_id="test_doc",
            artifact_type=ArtifactType.LEARNING,
            source_doc_hash="test_hash",
            confidence=1.0,
            payload={"learning_artifact": learning.model_dump()},
            provenance=Provenance(
                source_doc_hash="test_hash",
                document_id="test_doc",
            ),
        )
