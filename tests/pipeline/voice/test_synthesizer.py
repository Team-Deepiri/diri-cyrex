"""Unit tests for VoiceSynthesizer grounding + spoken_text (no live speech)."""
from __future__ import annotations

import pytest

from app.pipeline.contracts.models import (
    ArtifactBundle,
    ArtifactType,
    Citation,
    CitationLocator,
    CitedField,
    PersonaScope,
    Provenance,
)
from app.pipeline.voice.synthesizer import VoiceSynthesizer
from tests.fakes.artifact_store import InMemoryArtifactStore


def _lease_bundle(document_id: str = "lease_001") -> ArtifactBundle:
    cit = Citation(
        citation_id="cit_rent",
        document_id=document_id,
        source_doc_hash="hash1",
        locator=CitationLocator(
            locator_type="char_range", char_start=100, char_end=140, page_start=1
        ),
        quote="The base rent is $4500 per month",
        confidence=0.95,
    )
    field = CitedField(
        field_name="base_rent",
        value=4500,
        value_type="number",
        citations=[cit],
        confidence=0.9,
    )
    return ArtifactBundle(
        artifact_id="art_extract_1",
        document_id=document_id,
        artifact_type=ArtifactType.EXTRACTION,
        source_doc_hash="hash1",
        confidence=0.9,
        payload={"fields": [field.model_dump(mode="json")]},
        provenance=Provenance(source_doc_hash="hash1", document_id=document_id),
    )


@pytest.fixture
async def store_with_lease():
    store = InMemoryArtifactStore()
    await store.create(_lease_bundle())
    return store


@pytest.mark.asyncio
async def test_rent_question_returns_verbatim_span(store_with_lease):
    synth = VoiceSynthesizer(store_with_lease)
    resp = await synth.query(
        "lease_001",
        "What is the base rent?",
        PersonaScope(),
    )
    assert resp.confessed is False
    assert len(resp.spans) == 1
    assert "4500" in resp.spans[0].quote
    assert resp.spans[0].field_name == "base_rent"
    assert "4500" in resp.spoken_text()
    assert resp.question == "What is the base rent?"


@pytest.mark.asyncio
async def test_ungrounded_question_confesses(store_with_lease):
    synth = VoiceSynthesizer(store_with_lease)
    resp = await synth.query(
        "lease_001",
        "What is the parking allotment for electric scooters?",
        PersonaScope(),
    )
    assert resp.confessed is True
    assert resp.spans == []
    assert resp.gaps
    spoken = resp.spoken_text()
    assert "witness" in spoken.lower() or "ground" in spoken.lower() or "claim" in spoken.lower()


@pytest.mark.asyncio
async def test_hard_citation_gate_required(store_with_lease):
    synth = VoiceSynthesizer(store_with_lease)
    with pytest.raises(ValueError, match="hard_citation_gate"):
        await synth.query(
            "lease_001",
            "What is the rent?",
            PersonaScope(hard_citation_gate=False),
        )


@pytest.mark.asyncio
async def test_corpus_filter_blocks_doc(store_with_lease):
    synth = VoiceSynthesizer(store_with_lease)
    resp = await synth.query(
        "lease_001",
        "What is the rent?",
        PersonaScope(corpus_filter=["other_doc"]),
    )
    assert resp.confessed is True
    assert "corpus" in resp.gaps[0].reason.lower()


@pytest.mark.asyncio
async def test_query_with_speech_tts_mock():
    store = InMemoryArtifactStore()
    await store.create(_lease_bundle())

    class FakeSpeech:
        async def transcribe(self, audio, **kwargs):
            return {"text": "What is the monthly rent?", "provider": "mock", "model": "mock"}

        async def synthesize(self, text, **kwargs):
            return f"MOCK:{text}".encode("utf-8"), "audio/mock"

    synth = VoiceSynthesizer(store)
    resp, meta = await synth.query_with_speech(
        document_id="lease_001",
        audio=b"fake-wav",
        synthesize_audio=True,
        speech_client=FakeSpeech(),
    )
    assert resp.confessed is False
    assert "4500" in resp.spoken_text()
    assert meta["stt"]["provider"] == "mock"
    assert meta["audio"].startswith(b"MOCK:")
    assert meta["spoken_text"]


@pytest.mark.asyncio
async def test_speech_payload_shape(store_with_lease):
    synth = VoiceSynthesizer(store_with_lease)
    resp = await synth.query("lease_001", "rent amount?", PersonaScope())
    payload = resp.speech_payload()
    assert payload["document_id"] == "lease_001"
    assert "spoken_text" in payload
    assert payload["confessed"] is False


def test_tokenize_drops_stopwords_and_underscores():
    tokens = VoiceSynthesizer._tokenize("What is the base_rent on the lease?")
    assert "base" in tokens
    assert "rent" in tokens
    assert "what" not in tokens
    assert "the" not in tokens
    assert "is" not in tokens


def test_score_field_alias_boosts_rent():
    from app.pipeline.contracts.models import Citation, CitationLocator, CitedField

    cit = Citation(
        citation_id="c1",
        document_id="d1",
        source_doc_hash="h",
        locator=CitationLocator(locator_type="char_range", char_start=0, char_end=10),
        quote="The base rent is $4500 per month",
        confidence=0.9,
    )
    field = CitedField(
        field_name="base_rent",
        value=4500,
        value_type="number",
        citations=[cit],
        confidence=0.9,
    )
    synth = VoiceSynthesizer.__new__(VoiceSynthesizer)
    q = "how much is monthly rent"
    q_tokens = VoiceSynthesizer._tokenize(q)
    score = synth._score_field(q_tokens, q.lower(), field)
    assert score >= 0.12


def test_score_field_unrelated_is_low():
    from app.pipeline.contracts.models import Citation, CitationLocator, CitedField

    cit = Citation(
        citation_id="c1",
        document_id="d1",
        source_doc_hash="h",
        locator=CitationLocator(locator_type="char_range", char_start=0, char_end=10),
        quote="The base rent is $4500 per month",
        confidence=0.9,
    )
    field = CitedField(
        field_name="base_rent",
        value=4500,
        value_type="number",
        citations=[cit],
        confidence=0.9,
    )
    synth = VoiceSynthesizer.__new__(VoiceSynthesizer)
    q = "what color is the lobby carpet"
    q_tokens = VoiceSynthesizer._tokenize(q)
    score = synth._score_field(q_tokens, q.lower(), field)
    assert score < 0.12
