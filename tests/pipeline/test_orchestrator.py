"""Tests for ArtifactEngineOrchestrator.

These tests verify that the orchestrator correctly:
- Implements ``PipelineRunnerPort``
- Runs the parse stage and stores results
- Emits pressure events when a sink is configured
- Handles optional stages (None = skip gracefully)
- Returns a valid ArtifactBundle
"""

from __future__ import annotations

from typing import Any, List, Optional

import pytest

from app.pipeline.contracts.models import (
    ArtifactBundle,
    ArtifactType,
    PredictionRecord,
    Provenance,
    SynthesisResult,
)
from app.pipeline.contracts.ports import (
    AnticipatePort,
    ExtractPort,
)
from app.pipeline.orchestrator import ArtifactEngineOrchestrator
from app.pipeline.stages.parse import ParseError, ParseResult
from tests.fakes.artifact_store import InMemoryArtifactStore

# ---------------------------------------------------------------------------
# Fakes for optional stages
# ---------------------------------------------------------------------------


class FakeParseStage:
    """A fake parse stage that returns a known ParseResult."""

    def __init__(self, parse_result: Optional[ParseResult] = None) -> None:
        self._result = parse_result or ParseResult(
            raw_text="test content",
            document_type="txt",
            metadata={"filename": "test.txt"},
        )
        self.call_count: int = 0

    async def parse(self, file_content: bytes, filename: str) -> ParseResult:
        self.call_count += 1
        return self._result


class FailingParseStage:
    """A fake parse stage that always raises ParseError."""

    async def parse(self, file_content: bytes, filename: str) -> ParseResult:
        raise ParseError("Simulated parse failure")


class FakeAnticipate(AnticipatePort):
    """Returns a simple prediction record."""

    def __init__(self) -> None:
        self.call_count: int = 0

    async def run(
        self,
        parsed_doc: Any,
        document_class: str,
    ) -> List[PredictionRecord]:
        self.call_count += 1
        return [
            PredictionRecord(
                field_name="rent",
                predicted_mean=4500.0,
            )
        ]


class FakeExtract(ExtractPort):
    """Returns a minimal SynthesisResult."""

    def __init__(self) -> None:
        self.call_count: int = 0

    async def run(
        self,
        parsed_doc: Any,
        document_id: str,
        source_doc_hash: str,
    ) -> SynthesisResult:
        self.call_count += 1
        return SynthesisResult(
            document_id=document_id,
            source_doc_hash=source_doc_hash,
            final_fields=[],
            all_citations=[],
            confidence=0.85,
            provenance=Provenance(
                source_doc_hash=source_doc_hash,
                document_id=document_id,
            ),
        )


# ---------------------------------------------------------------------------
# Orchestrator fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def orchestrator():
    """Standard orchestrator with in-memory store and fake parse stage."""
    store = InMemoryArtifactStore()
    parse_stage = FakeParseStage()
    return ArtifactEngineOrchestrator(
        store=store,
        parse_stage=parse_stage,
    )


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    """Verify structural typing against PipelineRunnerPort."""

    def test_implements_pipeline_runner_port(self, orchestrator):
        """ArtifactEngineOrchestrator structurally satisfies PipelineRunnerPort.

        We check structural conformance rather than ``isinstance()``
        because ``PipelineRunnerPort`` is a plain ``typing.Protocol``
        without ``@runtime_checkable``.
        """
        # Has the required method with the right signature shape
        assert hasattr(orchestrator, "run_document")
        assert callable(orchestrator.run_document)
        # Protocol-level verification is handled by
        # ``test_ports_compliance.py`` in the contracts test suite.


# ---------------------------------------------------------------------------
# Basic pipeline flow
# ---------------------------------------------------------------------------


class TestPipelineFlow:
    """Verify the core document → artifact flow."""

    @pytest.mark.asyncio()
    async def test_run_document_returns_bundle(self, orchestrator):
        """run_document returns an ArtifactBundle with correct type."""
        bundle = await orchestrator.run_document(b"test", "test.txt")
        assert bundle is not None
        assert isinstance(bundle, ArtifactBundle)
        assert bundle.artifact_type == ArtifactType.EXTRACTION
        assert bundle.document_id is not None
        assert bundle.source_doc_hash is not None
        assert bundle.confidence > 0

    @pytest.mark.asyncio()
    async def test_run_document_stores_artifact(self, orchestrator):
        """The created artifact is retrievable from the store after run."""
        bundle = await orchestrator.run_document(b"test", "test.txt")
        retrieved = await orchestrator._store.get(bundle.artifact_id)
        assert retrieved is not None
        assert retrieved.artifact_id == bundle.artifact_id

    @pytest.mark.asyncio()
    async def test_run_document_parse_call_count(self, orchestrator):
        """The parse stage is called exactly once per run_document."""
        # Access the fake parse stage to check call_count
        parse_stage = orchestrator._parse_stage
        assert isinstance(parse_stage, FakeParseStage)
        assert parse_stage.call_count == 0

        await orchestrator.run_document(b"test", "test.txt")
        assert parse_stage.call_count == 1

        await orchestrator.run_document(b"test2", "test2.txt")
        assert parse_stage.call_count == 2


# ---------------------------------------------------------------------------
# Optional stages
# ---------------------------------------------------------------------------


class TestOptionalStages:
    """Verify orchestrator works when stages are None."""

    @pytest.mark.asyncio()
    async def test_all_optional_stages_none(self):
        """Pipeline completes when anticipate/extract/duel are None."""
        store = InMemoryArtifactStore()
        parse_stage = FakeParseStage()
        orch = ArtifactEngineOrchestrator(
            store=store,
            parse_stage=parse_stage,
            anticipate=None,
            extract=None,
            duel=None,
        )
        bundle = await orch.run_document(b"test", "test.txt")
        assert bundle is not None
        assert bundle.artifact_type == ArtifactType.EXTRACTION

    @pytest.mark.asyncio()
    async def test_with_all_stages(self):
        """Pipeline completes when anticipate and extract are provided."""
        store = InMemoryArtifactStore()
        parse_stage = FakeParseStage()
        orch = ArtifactEngineOrchestrator(
            store=store,
            parse_stage=parse_stage,
            anticipate=FakeAnticipate(),
            extract=FakeExtract(),
        )
        bundle = await orch.run_document(b"test", "test.txt")
        assert bundle is not None
        # Verify anticipate was called
        assert isinstance(orch._anticipate, FakeAnticipate)
        assert orch._anticipate.call_count == 1
        # Verify extract was called
        assert isinstance(orch._extract, FakeExtract)
        assert orch._extract.call_count == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Verify orchestrator handles parse failures gracefully."""

    @pytest.mark.asyncio()
    async def test_parse_failure_raises(self):
        """A failing parse stage propagates ParseError."""
        store = InMemoryArtifactStore()
        orch = ArtifactEngineOrchestrator(
            store=store,
            parse_stage=FailingParseStage(),
        )
        with pytest.raises(ParseError, match="Simulated parse failure"):
            await orch.run_document(b"test", "test.txt")


# ---------------------------------------------------------------------------
# Payload shape
# ---------------------------------------------------------------------------


class TestPayloadShape:
    """Verify the artifact payload contains expected keys."""

    @pytest.mark.asyncio()
    async def test_payload_contains_parse_result(self, orchestrator):
        """The artifact payload includes parse_result metadata."""
        bundle = await orchestrator.run_document(b"test", "test.txt")
        assert "parse_result" in bundle.payload
        assert bundle.payload["parse_result"]["document_type"] == "txt"

    @pytest.mark.asyncio()
    async def test_payload_contains_fields(self, orchestrator):
        """The artifact payload includes a fields list."""
        bundle = await orchestrator.run_document(b"test", "test.txt")
        assert "fields" in bundle.payload
        assert isinstance(bundle.payload["fields"], list)


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    """Verify the optional overall pipeline timeout."""

    @pytest.mark.asyncio()
    async def test_run_document_accepts_timeout(self, orchestrator):
        """A generous timeout does not alter a successful run."""
        bundle = await orchestrator.run_document(b"test", "test.txt", timeout=30)
        assert bundle is not None
        assert bundle.artifact_type == ArtifactType.EXTRACTION

    @pytest.mark.asyncio()
    async def test_run_document_times_out(self):
        """A too-short timeout raises asyncio.TimeoutError."""
        import asyncio

        class SlowParseStage:
            async def parse(self, file_content: bytes, filename: str) -> ParseResult:
                await asyncio.sleep(10)
                return ParseResult(raw_text="slow", document_type="txt")

        store = InMemoryArtifactStore()
        orch = ArtifactEngineOrchestrator(
            store=store,
            parse_stage=SlowParseStage(),
        )
        with pytest.raises(asyncio.TimeoutError):
            await orch.run_document(b"test", "slow.txt", timeout=0.05)

