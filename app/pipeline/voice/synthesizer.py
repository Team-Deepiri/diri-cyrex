"""Voice of the Document — grounded answers from extraction artifacts.

Never fabricates. Matches questions to CitedFields / Citation.quote text and
returns verbatim witness spans or confession gaps.

Audio I/O is delegated to deepiri-speech via ``query_with_speech`` / spoken helpers —
this class owns grounding only.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.pipeline.contracts.models import ArtifactType, Citation, CitedField, PersonaScope
from app.pipeline.contracts.ports import ArtifactStorePort

logger = logging.getLogger("cyrex.pipeline.voice.synthesizer")

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "how",
        "when",
        "where",
        "why",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "me",
        "my",
        "our",
        "your",
        "their",
        "please",
        "tell",
        "about",
        "document",
        "lease",
    }
)

# Common lease / contract question aliases → field name tokens
_FIELD_ALIASES: dict[str, set[str]] = {
    "rent": {"rent", "base_rent", "monthly_rent", "base", "payment", "amount"},
    "term": {"term", "lease_term", "duration", "period", "expiration", "expiry", "end_date"},
    "tenant": {"tenant", "lessee", "occupant", "renter"},
    "landlord": {"landlord", "lessor", "owner", "property_owner"},
    "address": {"address", "premises", "property", "location", "site"},
    "deposit": {"deposit", "security_deposit", "security"},
    "square": {"square", "sqft", "sf", "area", "footage"},
    "commencement": {"commencement", "start", "start_date", "begin", "effective"},
}


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
    field_name: Optional[str] = None
    confidence: Optional[float] = None
    match_score: Optional[float] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    page: Optional[int] = None


class VoiceQueryResponse(BaseModel):
    """Result of a voice query: cited spans and/or confession gaps."""

    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(default_factory=lambda: f"vq_{uuid4().hex}")
    document_id: str
    question: str = ""
    confessed: bool
    spans: list[WitnessSpan] = Field(default_factory=list)
    gaps: list[ConfessionGap] = Field(default_factory=list)
    fields_considered: int = 0
    match_threshold: float = 0.0

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
                f"I could not ground this claim in the document: {self.gaps[0].claim}"
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
            "gap_count": len(self.gaps),
        }


class UngroundedAnswerError(Exception):
    """Raised if a span would leave the pipeline without a verbatim quote."""


@dataclass
class _ScoredField:
    field: CitedField
    score: float


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------


class VoiceSynthesizer:
    """Document-grounded Q&A over EXTRACTION CitedFields.

    Parameters
    ----------
    store:
        Artifact store (latest EXTRACTION for the document).
    min_score:
        Minimum match score to accept a field (0–1 scale after normalization).
    max_spans:
        Max witness spans to return when several fields match.
    """

    def __init__(
        self,
        store: ArtifactStorePort,
        *,
        min_score: float = 0.12,
        max_spans: int = 3,
    ):
        self._store = store
        self.min_score = min_score
        self.max_spans = max_spans

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

        question = (question or "").strip()
        if not question:
            return VoiceQueryResponse(
                document_id=document_id,
                question=question,
                confessed=True,
                gaps=[ConfessionGap(claim="", reason="Empty question")],
            )

        if persona_scope.corpus_filter and document_id not in persona_scope.corpus_filter:
            return VoiceQueryResponse(
                document_id=document_id,
                question=question,
                confessed=True,
                gaps=[
                    ConfessionGap(
                        claim=question,
                        reason=f"Document '{document_id}' is outside the active corpus filter",
                    )
                ],
            )

        bundle = await self._store.get_latest(
            document_id=document_id,
            artifact_type=ArtifactType.EXTRACTION.value,
        )
        cited_fields = self._extract_cited_fields(bundle)

        if not cited_fields:
            return VoiceQueryResponse(
                document_id=document_id,
                question=question,
                confessed=True,
                fields_considered=0,
                match_threshold=self.min_score,
                gaps=[
                    ConfessionGap(
                        claim=question,
                        reason="No extraction fields available for this document",
                    )
                ],
            )

        scored = self._rank_fields(question, cited_fields)
        accepted = [s for s in scored if s.score >= self.min_score][: self.max_spans]

        if not accepted:
            return VoiceQueryResponse(
                document_id=document_id,
                question=question,
                confessed=True,
                fields_considered=len(cited_fields),
                match_threshold=self.min_score,
                gaps=[
                    ConfessionGap(
                        claim=question,
                        reason="No witness span available for this claim",
                    )
                ],
            )

        spans: list[WitnessSpan] = []
        gaps: list[ConfessionGap] = []
        for item in accepted:
            citation = self._select_citation(item.field)
            if citation is None:
                gaps.append(
                    ConfessionGap(
                        claim=question,
                        reason=(
                            f"Field '{item.field.field_name}' matched but has no "
                            "supporting citation"
                        ),
                    )
                )
                continue
            span = WitnessSpan(
                citation_id=citation.citation_id,
                quote=citation.quote,
                field_name=item.field.field_name,
                confidence=citation.confidence,
                match_score=round(item.score, 4),
                char_start=citation.locator.char_start,
                char_end=citation.locator.char_end,
                page=citation.locator.page_start,
            )
            self._assert_verbatim(span)
            spans.append(span)

        if not spans:
            return VoiceQueryResponse(
                document_id=document_id,
                question=question,
                confessed=True,
                fields_considered=len(cited_fields),
                match_threshold=self.min_score,
                gaps=gaps
                or [
                    ConfessionGap(
                        claim=question,
                        reason="Matched fields lacked verbatim citations",
                    )
                ],
            )

        # Partial confession if some matches lacked citations
        confessed = bool(gaps) and len(spans) == 0
        return VoiceQueryResponse(
            document_id=document_id,
            question=question,
            confessed=confessed,
            spans=spans,
            gaps=gaps,
            fields_considered=len(cited_fields),
            match_threshold=self.min_score,
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
    ) -> tuple[VoiceQueryResponse, dict[str, Any]]:
        """
        Full duplex helper: optional STT → grounded query → optional TTS.

        Returns ``(VoiceQueryResponse, speech_meta)`` where speech_meta includes
        ``spoken_text``, optional ``audio`` bytes / mime, and STT/TTS provider info.
        """
        scope = persona_scope or PersonaScope()

        # Injected clients skip settings/speech imports (unit tests).
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
                resp = VoiceQueryResponse(
                    document_id=document_id,
                    question="",
                    confessed=True,
                    gaps=[ConfessionGap(claim="", reason="STT returned empty transcript")],
                )
                meta["spoken_text"] = resp.spoken_text()
                return resp, meta

        if not q:
            resp = VoiceQueryResponse(
                document_id=document_id,
                question="",
                confessed=True,
                gaps=[ConfessionGap(claim="", reason="question or audio required")],
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

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _rank_fields(
        self, question: str, cited_fields: list[CitedField]
    ) -> list[_ScoredField]:
        q_tokens = self._tokenize(question)
        if not q_tokens:
            return []

        scored: list[_ScoredField] = []
        for cf in cited_fields:
            score = self._score_field(q_tokens, question.lower(), cf)
            if score > 0:
                scored.append(_ScoredField(field=cf, score=score))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored

    def _score_field(
        self, q_tokens: set[str], question_lower: str, cf: CitedField
    ) -> float:
        name_tokens = self._tokenize(cf.field_name)
        value_tokens = self._tokenize(str(cf.value) if cf.value is not None else "")
        quote_tokens: set[str] = set()
        for cit in cf.citations:
            quote_tokens |= self._tokenize(cit.quote)

        # Jaccard-ish overlaps
        name_overlap = self._overlap_ratio(q_tokens, name_tokens)
        value_overlap = self._overlap_ratio(q_tokens, value_tokens)
        quote_overlap = self._overlap_ratio(q_tokens, quote_tokens)

        # Alias boost (rent → base_rent, etc.)
        alias_boost = 0.0
        field_blob = name_tokens | self._expand_aliases(name_tokens)
        for alias_key, alias_set in _FIELD_ALIASES.items():
            if alias_key in q_tokens or (q_tokens & alias_set):
                if field_blob & alias_set or alias_key in field_blob:
                    alias_boost = max(alias_boost, 0.35)

        # Substring hits on field name / value
        substr = 0.0
        fname = cf.field_name.lower().replace("_", " ")
        if fname and fname in question_lower:
            substr = 0.4
        val = str(cf.value).lower() if cf.value is not None else ""
        if val and len(val) > 2 and val in question_lower:
            substr = max(substr, 0.25)

        # Confidence prior from field
        prior = 0.05 * float(cf.confidence or 0.0)

        raw = (
            0.45 * name_overlap
            + 0.25 * quote_overlap
            + 0.15 * value_overlap
            + alias_boost
            + substr
            + prior
        )
        # Soft-cap to ~1.0
        return 1.0 - math.exp(-raw)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = set(_TOKEN_RE.findall(text.lower().replace("_", " ")))
        return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}

    @staticmethod
    def _overlap_ratio(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        if inter == 0:
            return 0.0
        return inter / float(len(a))

    @staticmethod
    def _expand_aliases(tokens: set[str]) -> set[str]:
        expanded: set[str] = set()
        for t in tokens:
            for key, aliases in _FIELD_ALIASES.items():
                if t == key or t in aliases:
                    expanded |= aliases
                    expanded.add(key)
        return expanded

    def _extract_cited_fields(self, bundle: Any) -> list[CitedField]:
        if bundle is None:
            return []

        payload = getattr(bundle, "payload", None) or {}
        raw_fields = payload.get("fields", [])
        if not isinstance(raw_fields, list):
            return []

        fields: list[CitedField] = []
        for raw in raw_fields:
            try:
                if isinstance(raw, CitedField):
                    fields.append(raw)
                else:
                    fields.append(CitedField.model_validate(raw))
            except Exception as e:
                logger.warning(
                    "Skipping malformed CitedField in artifact %s: %s",
                    getattr(bundle, "artifact_id", "<unknown>"),
                    e,
                )
        return fields

    def _select_citation(self, cited_field: CitedField) -> Optional[Citation]:
        if not cited_field.citations:
            return None
        return max(cited_field.citations, key=lambda c: c.confidence)

    def _assert_verbatim(self, span: WitnessSpan) -> None:
        if not span.quote or not span.quote.strip():
            raise UngroundedAnswerError(
                f"Span for citation {span.citation_id} has no verbatim quote"
            )
