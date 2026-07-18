"""Fake ExtractPort for track-local tests."""

from __future__ import annotations

from typing import Any

from app.pipeline.contracts.models import CitedField, Provenance, SynthesisResult
from app.pipeline.contracts.ports import ExtractPort


def _default_synthesis(document_id: str, source_doc_hash: str) -> SynthesisResult:
    return SynthesisResult(
        document_id=document_id,
        source_doc_hash=source_doc_hash,
        final_fields=[
            CitedField(field_name="sample_field", value="sample", confidence=0.9),
        ],
        confidence=0.9,
        provenance=Provenance(source_doc_hash=source_doc_hash, document_id=document_id),
    )


class NoOpExtract(ExtractPort):
    """Returns a minimal valid SynthesisResult without running extraction."""

    async def run(
        self,
        parsed_doc: Any,
        document_id: str,
        source_doc_hash: str,
    ) -> SynthesisResult:
        return _default_synthesis(document_id, source_doc_hash)


class FixedExtract(ExtractPort):
    """Returns a fixed synthesis result for tests."""

    def __init__(self, result: SynthesisResult) -> None:
        self._result = result

    async def run(
        self,
        parsed_doc: Any,
        document_id: str,
        source_doc_hash: str,
    ) -> SynthesisResult:
        return self._result


# Alias used by contract compliance tests
FakeExtract = NoOpExtract
