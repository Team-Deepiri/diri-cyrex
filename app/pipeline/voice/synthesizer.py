"""Voice of the Document — citation-gated witness synthesis (RAG embedding match).

Audio I/O is delegated to deepiri-speech via ``query_with_speech`` / spoken helpers —
this class owns grounding only.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Protocol, Sequence
from uuid import uuid4

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

logger = logging.getLogger("cyrex.pipeline.voice.synthesizer")

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
    query_id: str = Field(default_factory=lambda: f"vq_{uuid4().hex}")
    document_id: str = ""
    question: str = ""

    def spoken_text(self) -> str:
        """Plain language for TTS (deepiri-speech). Never invents facts."""
        if not self.confessed and self.spans:
            quotes = [s.quote.strip() for s in self.spans if s.quote and s.quote.strip()]
            if len(quotes) == 1:
                return quotes[0]
            if quotes:
                return " ".join(f"{i}. {q}" for i, q in enumerate(quotes, 1))
        if self.gaps:
            reasons = [g.reason for g in self.gaps if g.reason]
            if reasons:
                return reasons[0]
            return (
                f"I could not ground this claim in the document: {self.gaps[0].claim_attempted}"
            )
        return "No answer found in the document."

    def speech_payload(self) -> dict[str, Any]:
        """Structured payload for speech / UI clients."""
        return {
            "query_id": self.query_id,
            "document_id": self.document_id,
            "question": self.question,
            "confessed": self.confessed,
            "spoken_text": self.spoken_text(),
            "span_count": len(self.spans),
            "gap_count": len(self.gaps or []),
        }


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
                document_id=document_id,
                question=question,
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
                document_id=document_id,
                question=question,
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
                document_id=document_id,
                question=question,
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

        return VoiceQueryResult(
            document_id=document_id,
            question=question,
            confessed=False,
            spans=spans,
            gaps=None,
        )

    async def query_with_speech(
        self,
        *,
        document_id: str,
        question: Optional[str] = None,
        persona_scope: Optional[PersonaScope] = None,
        audio: Optional[bytes] = None,
        audio_mime_type: str = "audio/wav",
        synthesize_audio: bool = True,
        speech_client: Any = None,
    ) -> tuple[VoiceQueryResult, dict[str, Any]]:
        """
        Full duplex helper: optional STT → grounded query → optional TTS.

        Returns ``(VoiceQueryResult, speech_meta)`` where speech_meta includes
        ``spoken_text``, optional ``audio`` bytes / mime, and STT/TTS provider info.
        """
        scope = persona_scope or PersonaScope()

        tts_voice: Optional[str] = None
        if speech_client is not None:
            speech_enabled = True
        else:
            from app.settings import settings

            speech_enabled = bool(getattr(settings, "SPEECH_ENABLED", True))
            tts_voice = getattr(settings, "SPEECH_TTS_VOICE", None)

        meta: dict[str, Any] = {
            "engine": "deepiri-speech",
            "enabled": speech_enabled,
            "stt": None,
            "tts": None,
            "spoken_text": None,
            "audio": None,
            "audio_mime_type": None,
        }

        def _client() -> Any:
            if speech_client is not None:
                return speech_client
            from app.integrations.speech_client import get_speech_client

            return get_speech_client()

        q = (question or "").strip()
        if audio:
            if not speech_enabled:
                raise RuntimeError("audio STT requires SPEECH_ENABLED and deepiri-speech")
            stt = await _client().transcribe(
                audio,
                mime_type=audio_mime_type,
                session_id=document_id,
            )
            q = (stt.get("text") or "").strip()
            meta["stt"] = {
                "provider": stt.get("provider"),
                "model": stt.get("model"),
                "text": q,
            }
            if not q:
                resp = VoiceQueryResult(
                    document_id=document_id,
                    question="",
                    confessed=True,
                    gaps=[
                        ConfessionGap(
                            claim_attempted="",
                            reason="STT returned empty transcript",
                        )
                    ],
                )
                meta["spoken_text"] = resp.spoken_text()
                return resp, meta

        if not q:
            resp = VoiceQueryResult(
                document_id=document_id,
                question="",
                confessed=True,
                gaps=[
                    ConfessionGap(
                        claim_attempted="",
                        reason="question or audio required",
                    )
                ],
            )
            meta["spoken_text"] = resp.spoken_text()
            return resp, meta

        response = await self.query(document_id, q, scope)
        spoken = response.spoken_text()
        meta["spoken_text"] = spoken

        if synthesize_audio and spoken and speech_enabled:
            try:
                audio_out, mime = await _client().synthesize(
                    spoken,
                    voice=tts_voice,
                    session_id=document_id,
                )
                meta["audio"] = audio_out
                meta["audio_mime_type"] = mime
                meta["tts"] = {
                    "voice": tts_voice,
                    "bytes": len(audio_out),
                    "mime": mime,
                }
            except Exception as exc:
                logger.warning("TTS via deepiri-speech failed: %s", exc)
                meta["tts"] = {"error": str(exc)}

        return response, meta
