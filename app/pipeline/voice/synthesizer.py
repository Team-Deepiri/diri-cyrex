"""Voice of the Document — citation-gated witness synthesis (v1).

Returns verbatim cited spans when the witness set supports the question,
otherwise confesses with structured gaps (hard citation gate).
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional, Sequence

from pydantic import BaseModel, Field

from app.pipeline.contracts.models import (
    ArtifactBundle,
    Citation,
    CitationLocator,
    PersonaScope,
)
from app.pipeline.contracts.ports import ArtifactStorePort

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "what",
        "when",
        "where",
        "who",
        "how",
        "does",
        "do",
        "did",
        "can",
        "could",
        "would",
        "should",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "be",
        "this",
        "that",
        "it",
    }
)

_MATCH_THRESHOLD = 0.25


class ConfessionGap(BaseModel):
    """Gap recorded when the witness set cannot support a claim."""

    claim_attempted: str
    reason: str = "no_citation"


class WitnessSpan(BaseModel):
    citation_id: str
    quote: str
    char_start: int = 0
    char_end: int = 0
    page: Optional[int] = None


class VoiceQueryResult(BaseModel):
    confessed: bool
    spans: List[WitnessSpan] = Field(default_factory=list)
    gaps: Optional[List[ConfessionGap]] = None


def _tokens(text: str) -> set[str]:
    raw = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in raw if t not in _STOP_WORDS and len(t) > 1}


def _score_question_against_quote(question: str, quote: str) -> float:
    q_tokens = _tokens(question)
    if not q_tokens:
        return 0.0
    quote_l = quote.lower()
    hits = sum(1 for t in q_tokens if t in quote_l)
    return hits / len(q_tokens)


def _locator_page(locator: CitationLocator) -> Optional[int]:
    if locator.page_start is not None:
        return locator.page_start
    return None


def _citation_to_witness(citation: Citation) -> WitnessSpan:
    loc = citation.locator
    return WitnessSpan(
        citation_id=citation.citation_id,
        quote=citation.quote,
        char_start=loc.char_start or 0,
        char_end=loc.char_end or len(citation.quote),
        page=_locator_page(loc),
    )


def _iter_payload_citations(payload: dict[str, Any]) -> Iterable[Citation]:
    for field in payload.get("fields") or []:
        if not isinstance(field, dict):
            continue
        for raw in field.get("citations") or []:
            if isinstance(raw, Citation):
                yield raw
            elif isinstance(raw, dict):
                yield Citation.model_validate(raw)

    synth = payload.get("synthesis_result")
    if isinstance(synth, dict):
        for raw in synth.get("all_citations") or []:
            if isinstance(raw, dict):
                yield Citation.model_validate(raw)


def collect_witness_citations(bundles: Sequence[ArtifactBundle]) -> List[Citation]:
    """Merge bundle-level and payload-embedded citations (deduped by id)."""
    seen: set[str] = set()
    out: List[Citation] = []
    for bundle in bundles:
        for citation in bundle.citations:
            if citation.citation_id not in seen:
                seen.add(citation.citation_id)
                out.append(citation)
        if bundle.payload:
            for citation in _iter_payload_citations(bundle.payload):
                if citation.citation_id not in seen:
                    seen.add(citation.citation_id)
                    out.append(citation)
    return out


class VoiceSynthesizer:
    """Citation-gated voice query over stored artifact witness sets."""

    def __init__(self, store: ArtifactStorePort) -> None:
        self._store = store

    async def query(
        self,
        document_id: str,
        question: str,
        persona_scope: Optional[PersonaScope] = None,
    ) -> VoiceQueryResult:
        scope = persona_scope or PersonaScope()
        if scope.corpus_filter and document_id not in scope.corpus_filter:
            return VoiceQueryResult(
                confessed=True,
                spans=[],
                gaps=[
                    ConfessionGap(
                        claim_attempted=question,
                        reason="document_not_in_corpus_filter",
                    )
                ],
            )

        bundles = await self._store.list_by_document(document_id)
        citations = collect_witness_citations(bundles)
        if not citations:
            return VoiceQueryResult(
                confessed=True,
                spans=[],
                gaps=[
                    ConfessionGap(
                        claim_attempted=question,
                        reason="no_witness_set",
                    )
                ],
            )

        scored = [
            (c, _score_question_against_quote(question, c.quote)) for c in citations
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        best_citation, best_score = scored[0]

        if scope.hard_citation_gate and best_score < _MATCH_THRESHOLD:
            return VoiceQueryResult(
                confessed=True,
                spans=[],
                gaps=[
                    ConfessionGap(
                        claim_attempted=question,
                        reason="no_citation",
                    )
                ],
            )

        spans = [_citation_to_witness(best_citation)]
        if not scope.witness_set_only and len(scored) > 1:
            for citation, score in scored[1:4]:
                if score >= _MATCH_THRESHOLD:
                    spans.append(_citation_to_witness(citation))

        return VoiceQueryResult(confessed=False, spans=spans, gaps=None)
