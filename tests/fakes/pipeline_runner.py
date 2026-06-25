"""Fake PipelineRunnerPort for track-local API route tests."""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.pipeline.contracts.models import ArtifactBundle, ArtifactType, Provenance
from app.pipeline.contracts.ports import PipelineRunnerPort


def _default_golden_bundle() -> ArtifactBundle:
    return ArtifactBundle(
        document_id="doc-test",
        artifact_type=ArtifactType.EXTRACTION,
        source_doc_hash="hash-test",
        confidence=0.9,
        provenance=Provenance(source_doc_hash="hash-test", document_id="doc-test"),
    )


class FakePipelineRunner(PipelineRunnerPort):
    """Returns a golden ArtifactBundle without running the real pipeline."""

    def __init__(self, golden_bundle: ArtifactBundle | None = None) -> None:
        self._golden = golden_bundle or _default_golden_bundle()
        self.call_count: int = 0

    async def run_document(
        self,
        file_content: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactBundle:
        self.call_count += 1
        return self._golden
