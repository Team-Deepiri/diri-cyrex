"""Verify fake implementations satisfy Protocol interfaces."""

from __future__ import annotations

import inspect

import pytest

from app.pipeline.contracts.ports import (
    AnticipatePort,
    ArtifactStorePort,
    CorrectionWriterPort,
    DuelRunnerPort,
    ExtractPort,
    InvalidationPort,
    PipelineRunnerPort,
    PressureReadModelPort,
    PressureSignalSink,
    ReckoningReadPort,
)
from tests.fakes.anticipate import FakeAnticipate
from tests.fakes.artifact_store import InMemoryArtifactStore
from tests.fakes.correction import FakeCorrectionWriter
from tests.fakes.invalidation import FakeInvalidationPort
from tests.fakes.pipeline_runner import FakePipelineRunner
from tests.fakes.pressure import FakePressureReadModel, FakePressureSignalSink
from tests.fakes.reckoning import FakeReckoningRead


def _has_methods(obj, protocol) -> list[str]:
    """Return list of protocol methods missing from obj."""
    missing = []
    for name in dir(protocol):
        if name.startswith("_"):
            continue
        attr = getattr(protocol, name, None)
        if not callable(attr):
            continue
        if not hasattr(obj, name):
            missing.append(name)
    return missing


class TestFakeCompliance:
    @pytest.mark.parametrize(
        "fake_cls, protocol",
        [
            (InMemoryArtifactStore, ArtifactStorePort),
            (FakePressureSignalSink, PressureSignalSink),
            (FakePressureReadModel, PressureReadModelPort),
            (FakeReckoningRead, ReckoningReadPort),
            (FakeInvalidationPort, InvalidationPort),
            (FakeCorrectionWriter, CorrectionWriterPort),
            (FakePipelineRunner, PipelineRunnerPort),
            (FakeAnticipate, AnticipatePort),
        ],
    )
    def test_fake_implements_protocol_methods(self, fake_cls, protocol):
        obj = fake_cls()
        missing = _has_methods(obj, protocol)
        assert not missing, f"{fake_cls.__name__} missing: {missing}"

    @pytest.mark.asyncio
    async def test_in_memory_store_crud(self):
        from app.pipeline.contracts.models import ArtifactBundle, ArtifactType, Provenance

        store = InMemoryArtifactStore()
        bundle = ArtifactBundle(
            document_id="doc1",
            artifact_type=ArtifactType.EXTRACTION,
            source_doc_hash="hash1",
            confidence=0.9,
            provenance=Provenance(
                source_doc_hash="hash1",
                document_id="doc1",
            ),
        )
        created = await store.create(bundle)
        fetched = await store.get(created.artifact_id)
        assert fetched is not None
        assert fetched.document_id == "doc1"

    @pytest.mark.asyncio
    async def test_pressure_sink_captures_events(self):
        from app.pipeline.contracts.pressure_events import LowConfidenceField

        sink = FakePressureSignalSink()
        event = LowConfidenceField(
            document_id="d1",
            section_id="s1",
            field_name="rent",
            confidence=0.3,
        )
        await sink.emit(event)
        assert len(sink.events) == 1
        assert sink.events[0].field_name == "rent"
