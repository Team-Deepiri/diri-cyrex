"""Voice of the Document — citation-gated witness synthesis (RAG embedding match)."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Protocol, Sequence

from pydantic import BaseModel, Field

from app.pipeline.contracts.models import (
    ArtifactBundle,
    Citation,
    CitationLocator,
    ConfessionGap,
    PersonaScope,
    WitnessSpan,
)
from app.pipeline.contracts.ports import ArtifactStorePort

_DEFAULT_MATCH_THRESHOLD = 0.35


class WitnessScorer(Protocol):
    def score(self, question: str, quote: str) -> float: ...
    def rank(
        self, question: str, quotes: Sequence[str], *, threshold: float = 0.35
    ) -> List[Any]: ...


class VoiceQueryResult(BaseModel):
    confessed: bool
    spans: List[WitnessSpan] = Field(default_factory=list)
    gaps: Optional[List[ConfessionGap]] = None


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

    def __init__(
        self,
        store: ArtifactStorePort,
        scorer: Optional[WitnessScorer] = None,
        match_threshold: float = _DEFAULT_MATCH_THRESHOLD,
    ) -> None:
        self._store = store
        self._scorer = scorer
        self._match_threshold = match_threshold

    def _get_scorer(self) -> WitnessScorer:
        if self._scorer is not None:
            return self._scorer
        from app.pipeline.voice.witness_scorer import get_witness_scorer

        return get_witness_scorer()

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

        scorer = self._get_scorer()
        scored = [(c, scorer.score(question, c.quote)) for c in citations]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        best_citation, best_score = scored[0]

        if scope.hard_citation_gate and best_score < self._match_threshold:
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
                if score >= self._match_threshold:
                    spans.append(_citation_to_witness(citation))

        return VoiceQueryResult(confessed=False, spans=spans, gaps=None)
