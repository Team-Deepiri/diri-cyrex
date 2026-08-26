"""Tests for VoiceSynthesizer citation gate."""

from __future__ import annotations

import pytest

from app.pipeline.contracts.models import (
    ArtifactBundle,
    ArtifactType,
    Citation,
    CitationLocator,
    PersonaScope,
    Provenance,
)
from app.pipeline.voice.synthesizer import VoiceSynthesizer, collect_witness_citations


class _MemoryStore:
    def __init__(self, bundles: list[ArtifactBundle]) -> None:
        self._bundles = bundles

    async def list_by_document(self, document_id: str) -> list[ArtifactBundle]:
        return [b for b in self._bundles if b.document_id == document_id]

    async def get(self, artifact_id: str):
        return None


def _citation(quote: str, cid: str = "cit_1") -> Citation:
    return Citation(
        citation_id=cid,
        document_id="doc_1",
        source_doc_hash="hash",
        locator=CitationLocator(
            locator_type="char_range",
            char_start=10,
            char_end=10 + len(quote),
        ),
        quote=quote,
        confidence=0.9,
    )


def _bundle(citations: list[Citation]) -> ArtifactBundle:
    return ArtifactBundle(
        artifact_id="art_1",
        document_id="doc_1",
        artifact_type=ArtifactType.EXTRACTION,
        source_doc_hash="hash",
        confidence=0.9,
        payload={},
        provenance=Provenance(source_doc_hash="hash", document_id="doc_1"),
        citations=citations,
    )


class TestCollectWitnessCitations:
    def test_dedupes_by_citation_id(self):
        c = _citation("rent is 4500")
        bundles = [_bundle([c]), _bundle([c])]
        assert len(collect_witness_citations(bundles)) == 1


class _FakeScorer:
    """Deterministic scorer — token overlap, no embedding model."""

    def score(self, question: str, quote: str) -> float:
        q = set(question.lower().split())
        t = set(quote.lower().split())
        if not q or not t:
            return 0.0
        return len(q & t) / float(len(q))

    def rank(self, question: str, quotes, *, threshold: float = 0.35):
        scored = [(self.score(question, q), q) for q in quotes]
        scored.sort(reverse=True)
        return [{"quote": q, "score": s} for s, q in scored if s >= threshold]


class TestVoiceSynthesizer:
    @pytest.mark.asyncio
    async def test_returns_span_when_quote_matches_question(self):
        store = _MemoryStore(
            [_bundle([_citation("The base rent shall be $4,500 per month.")])]
        )
        synth = VoiceSynthesizer(store, scorer=_FakeScorer())
        result = await synth.query(
            "doc_1",
            "What is the base rent amount?",
            PersonaScope(hard_citation_gate=True),
        )
        assert result.confessed is False
        assert len(result.spans) == 1
        assert "4,500" in result.spans[0].quote

    @pytest.mark.asyncio
    async def test_confesses_when_no_match(self):
        store = _MemoryStore([_bundle([_citation("Termination clause is 90 days.")])])
        synth = VoiceSynthesizer(store, scorer=_FakeScorer())
        result = await synth.query(
            "doc_1",
            "What is the base rent amount?",
            PersonaScope(hard_citation_gate=True),
        )
        assert result.confessed is True
        assert result.gaps is not None
        assert result.gaps[0].reason == "no_citation"

    @pytest.mark.asyncio
    async def test_confesses_when_no_witness_set(self):
        store = _MemoryStore([_bundle([])])
        synth = VoiceSynthesizer(store, scorer=_FakeScorer())
        result = await synth.query("doc_1", "Anything?", PersonaScope())
        assert result.confessed is True
        assert result.gaps[0].reason == "no_witness_set"

    @pytest.mark.asyncio
    async def test_spoken_text_uses_verbatim_quote(self):
        store = _MemoryStore(
            [_bundle([_citation("The base rent shall be $4,500 per month.")])]
        )
        synth = VoiceSynthesizer(store, scorer=_FakeScorer())
        result = await synth.query(
            "doc_1",
            "What is the base rent amount?",
            PersonaScope(hard_citation_gate=False, witness_set_only=True),
        )
        assert "4,500" in result.spoken_text()
        assert result.speech_payload()["span_count"] == 1

    @pytest.mark.asyncio
    async def test_query_with_speech_tts_via_injected_client(self):
        class _Speech:
            async def synthesize(self, text, voice=None, session_id=None):
                return b"WAVDATA", "audio/wav"

            async def transcribe(self, audio, mime_type="audio/wav", session_id=None):
                return {"text": "What is the base rent amount?", "provider": "fake"}

        store = _MemoryStore(
            [_bundle([_citation("The base rent shall be $4,500 per month.")])]
        )
        synth = VoiceSynthesizer(store, scorer=_FakeScorer())
        result, meta = await synth.query_with_speech(
            document_id="doc_1",
            question="What is the base rent amount?",
            persona_scope=PersonaScope(hard_citation_gate=False),
            synthesize_audio=True,
            speech_client=_Speech(),
        )
        assert result.confessed is False
        assert meta["spoken_text"]
        assert meta["audio"] == b"WAVDATA"
        assert meta["audio_mime_type"] == "audio/wav"
