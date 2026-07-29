"""Answers a question about a document strictly by matching it to
extracted CitedFields and returning their verbatim Citation.quote text.
If nothing grounds the question, the field is confessed instead of fabricated."""

from __future__ import annotations

import re
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.pipeline.contracts.models import ArtifactType, Citation, CitedField, PersonaScope
from app.pipeline.contracts.ports import ArtifactStorePort


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ConfessionGap(BaseModel):
    """A claim in the question that could not be grounded in any citation."""

    model_config = ConfigDict(extra="forbid")

    claim: str
    reason: str = "No witness span available for this claim"


class WitnessSpan(BaseModel):
    """One verbatim cited span used to answer the question."""

    model_config = ConfigDict(extra="forbid")

    citation_id: str
    quote: str
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    page: Optional[int] = None


class VoiceQueryResponse(BaseModel):
    """Result of a voice query: cited spans and/or confession gaps."""

    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(default_factory=lambda: f"vq_{uuid4().hex}")
    document_id: str
    confessed: bool
    spans: list[WitnessSpan] = Field(default_factory=list)
    gaps: list[ConfessionGap] = Field(default_factory=list)


class UngroundedAnswerError(Exception):
    """Raised if a span would leave the pipeline without a verbatim quote."""


class VoiceSynthesizer:
    """Answers a question about a document using only verbatim cited spans
    from that document's latest EXTRACTION artifact.

    TODO: swap for real similarity search"""

    def __init__(self, store: ArtifactStorePort):
        self._store = store

    async def query(
        self,
        document_id: str,
        question: str,
        persona_scope: PersonaScope,
    ) -> VoiceQueryResponse:
        if not persona_scope.hard_citation_gate:
            raise ValueError(
                "VoiceSynthesizer requires PersonaScope.hard_citation_gate=True"
            )

        bundle = await self._store.get_latest(
            document_id=document_id,
            artifact_type=ArtifactType.EXTRACTION.value,
        )

        cited_fields = self._extract_cited_fields(bundle)
        matched_field = self._match_field(question, cited_fields)

        if matched_field is None:
            return VoiceQueryResponse(
                document_id=document_id,
                confessed=True,
                spans=[],
                gaps=[ConfessionGap(claim=question)],
            )

        citation = self._select_citation(matched_field)
        if citation is None:
            # Field matched, but has no supporting citation at all
            return VoiceQueryResponse(
                document_id=document_id,
                confessed=True,
                spans=[],
                gaps=[
                    ConfessionGap(
                        claim=question,
                        reason=f"'{matched_field.field_name}' has no supporting citation",
                    )
                ],
            )

        span = WitnessSpan(
            citation_id=citation.citation_id,
            quote=citation.quote,
            char_start=citation.locator.char_start,
            char_end=citation.locator.char_end,
            page=citation.locator.page_start,
        )
        self._assert_verbatim(span)

        return VoiceQueryResponse(
            document_id=document_id,
            confessed=False,
            spans=[span],
            gaps=[],
        )

    def _extract_cited_fields(self, bundle: Any) -> list[CitedField]:
        """Pull CitedFields out of an ArtifactBundle's payload."""
        if bundle is None:
            return []

        raw_fields = bundle.payload.get("fields", [])
        fields: list[CitedField] = []
        for raw in raw_fields:
            try:
                fields.append(CitedField.model_validate(raw))
            except Exception:
                continue
        return fields

    def _match_field(
        self, question: str, cited_fields: list[CitedField]
    ) -> Optional[CitedField]:
        #Placeholder
        question_terms = set(re.findall(r"[a-z0-9]+", question.lower()))

        best: Optional[CitedField] = None
        best_overlap = 0
        for cf in cited_fields:
            field_terms = set(re.findall(r"[a-z0-9]+", cf.field_name.lower()))
            overlap = len(question_terms & field_terms)
            if overlap > best_overlap:
                best_overlap = overlap
                best = cf

        return best if best_overlap > 0 else None

    def _select_citation(self, cited_field: CitedField) -> Optional[Citation]:
        if not cited_field.citations:
            return None
        return max(cited_field.citations, key=lambda c: c.confidence)

    def _assert_verbatim(self, span: WitnessSpan) -> None:
        if not span.quote:
            raise UngroundedAnswerError(
                f"Span for citation {span.citation_id} has no verbatim quote"
            )