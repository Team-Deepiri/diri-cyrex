"""Integration test: PostgresArtifactStore → projector → PressureSignalSink.

Exercises the full Track A emission chain (Appendix A of the design plan):
a bundle is persisted via ``PostgresArtifactStore.create()`` and the optional
``PressureSignalSink`` receives the projected events. Uses the production
``PressureBusSink`` (bus swapped for a fake, local handlers capture events).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
import pytest_asyncio

from app.pipeline.contracts.models import (
    ArtifactBundle,
    ArtifactType,
    Provenance,
)
from app.pipeline.contracts.pressure_events import (
    LowConfidenceField,
    PressureEvent,
)
from app.pipeline.projectors.pressure_bus_sink import PressureBusSink
from app.pipeline.registry.postgres_store import PostgresArtifactStore


class _FakeBus:
    """Stands in for the stream publisher; no network in tests."""

    def __init__(self) -> None:
        self.calls: List[Any] = []

    async def publish(self, stream, event_type, payload, *, maxlen=50000) -> str:
        self.calls.append((stream, event_type, payload))
        return "2-0"


class _Collector:
    """Local handler for PressureBusSink — receives every emitted event."""

    def __init__(self) -> None:
        self.events: List[PressureEvent] = []

    def __call__(self, event: PressureEvent) -> None:
        self.events.append(event)


@pytest_asyncio.fixture()
async def pressure_store(pg_manager):
    """PostgresArtifactStore wired to a production PressureBusSink."""
    collector = _Collector()
    bus = _FakeBus()
    sink = PressureBusSink(local_handlers=[collector])
    sink._bus = bus

    store = PostgresArtifactStore(postgres=pg_manager, pressure_sink=sink)
    await store.ensure_schema()
    db = await store._db()
    await db.execute(
        "TRUNCATE cyrex.citations, cyrex.artifact_refs, "
        "cyrex.artifacts, cyrex.learning_artifacts CASCADE"
    )
    return store, collector, bus


def _bundle(
    artifact_id: str,
    document_id: str,
    payload: Dict[str, Any],
) -> ArtifactBundle:
    return ArtifactBundle(
        artifact_id=artifact_id,
        document_id=document_id,
        version=1,
        artifact_type=ArtifactType.EXTRACTION,
        source_doc_hash="hash_press_1",
        confidence=0.6,
        payload=payload,
        provenance=Provenance(source_doc_hash="hash_press_1", document_id=document_id),
    )


@pytest.mark.asyncio()
async def test_create_emits_low_confidence_event(pressure_store):
    """A low-confidence field in the payload produces a LowConfidenceField."""
    store, collector, bus = pressure_store
    bundle = _bundle(
        "art_press_1",
        "doc_press_1",
        {"fields": [{"field_name": "rent", "confidence": 0.3}]},
    )
    await store.create(bundle)

    assert len(collector.events) == 1
    event = collector.events[0]
    assert isinstance(event, LowConfidenceField)
    assert event.field_name == "rent"
    assert event.confidence == 0.3
    assert event.document_id == "doc_press_1"
    assert event.artifact_id == "art_press_1"

    # The bus stream received the serialized event as well.
    assert len(bus.calls) == 1
    stream, event_type, payload = bus.calls[0]
    assert stream == "pipeline.pressure.events"
    assert event_type == "pressure.event"
    assert payload["pressure_event_type"] == "low_confidence_field"


@pytest.mark.asyncio()
async def test_create_emits_multiple_event_types(pressure_store):
    """Discrepancies and low-confidence fields project into separate events."""
    store, collector, _bus = pressure_store
    bundle = _bundle(
        "art_press_2",
        "doc_press_2",
        {
            "page": 2,
            "discrepancies": [
                {
                    "field_name": "rent",
                    "pass_a_value": "4500",
                    "pass_b_value": "4800",
                    "confidence_delta": 0.15,
                }
            ],
            "fields": [{"field_name": "sqft", "confidence": 0.25}],
        },
    )
    await store.create(bundle)

    types = {type(e).__name__ for e in collector.events}
    assert types == {"PassDiscrepancy", "LowConfidenceField"}
    assert all(e.document_id == "doc_press_2" for e in collector.events)
    assert all(e.page == 2 for e in collector.events if e.page is not None)


@pytest.mark.asyncio()
async def test_create_no_signals_emits_nothing(pressure_store):
    """A clean payload does not emit any pressure events."""
    store, collector, bus = pressure_store
    bundle = _bundle(
        "art_press_3",
        "doc_press_3",
        {"fields": [{"field_name": "rent", "confidence": 0.95}]},
    )
    await store.create(bundle)

    assert collector.events == []
    assert bus.calls == []


@pytest.mark.asyncio()
async def test_get_round_trip_preserves_citations(pressure_store):
    """Bundles persisted with citations come back with citations loaded."""
    store, _collector, _bus = pressure_store
    from app.pipeline.contracts.models import Citation, CitationLocator

    bundle = _bundle("art_press_4", "doc_press_4", {})
    bundle.citations = [
        Citation(
            citation_id="cit_press_1",
            document_id="doc_press_4",
            source_doc_hash="hash_press_1",
            locator=CitationLocator(
                locator_type="char_range",
                char_start=10,
                char_end=20,
            ),
            quote="hello world",
            confidence=0.9,
        )
    ]
    await store.create(bundle)

    fetched = await store.get("art_press_4")
    assert fetched is not None
    assert len(fetched.citations) == 1
    assert fetched.citations[0].quote == "hello world"
