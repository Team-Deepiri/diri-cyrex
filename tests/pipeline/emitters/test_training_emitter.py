"""Unit tests for AGI training_emitter (no Redis/Postgres required)."""

from __future__ import annotations

import pytest

from app.pipeline.emitters.training_emitter import MIN_QUALITY, TrainingEmitter


def test_training_emitter_import_does_not_initialize_rag_bridge():
    assert TrainingEmitter.__name__ == "TrainingEmitter"


class _FakeBus:
    def __init__(self):
        self.calls = []

    async def publish(self, stream, event_type, payload, *, maxlen=50000):
        self.calls.append((stream, event_type, payload))
        return "1-0"


class _FakePostgres:
    def __init__(self):
        self.statements = []

    async def execute(self, sql, params=None):
        self.statements.append((sql, params))


@pytest.mark.asyncio
async def test_emit_structured_dual_writes(monkeypatch):
    bus = _FakeBus()
    pg = _FakePostgres()
    emitter = TrainingEmitter(postgres=pg, producer="training_emitter")
    emitter._bus = bus
    emitter._schema_ready = True

    rid = await emitter.emit_structured(
        instruction="Extract rent",
        output="$1200",
        input_text="Lease clause...",
        document_id="doc-1",
        artifact_id="art-1",
        quality_score=0.9,
    )
    assert rid
    assert len(bus.calls) == 1
    stream, event_type, payload = bus.calls[0]
    assert stream == "pipeline.helox-training.structured"
    assert payload["producer"] == "training_emitter"
    assert payload["instruction"] == "Extract rent"
    assert any("helox_training_samples" in s[0] for s in pg.statements)
    assert any("helox_sample_lineage" in s[0] for s in pg.statements)


@pytest.mark.asyncio
async def test_emit_correction_uses_correction_writer_producer():
    bus = _FakeBus()
    emitter = TrainingEmitter(postgres=None)
    emitter._bus = bus

    await emitter.emit_correction(
        instruction="Fix party name",
        corrected_output="Acme LLC",
        artifact_id="art-2",
    )
    assert bus.calls[0][2]["producer"] == "correction_writer"


@pytest.mark.asyncio
async def test_quality_gate_drops_low_scores():
    bus = _FakeBus()
    emitter = TrainingEmitter(postgres=None)
    emitter._bus = bus
    rid = await emitter.emit_raw(text="nope", quality_score=MIN_QUALITY - 0.1)
    assert rid is None
    assert bus.calls == []
