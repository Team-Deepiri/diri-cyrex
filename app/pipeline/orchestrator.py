"""Artifact Engine Orchestrator — implements PipelineRunnerPort.

The orchestrator wires document parsing, optional adversarial stages
(anticipate, extract, duel), reflection, storage, and pressure event
emission into a single ``PipelineRunnerPort.run_document()`` method.

Ports are injected via the constructor.  Any stage can be ``None`` —
the pipeline adapts gracefully (optional stages are simply skipped).

Usage::

    from app.pipeline.orchestrator import ArtifactEngineOrchestrator
    from app.pipeline.registry.sqlite_store import SqliteArtifactStore
    from app.pipeline.stages.parse import ParseStage
    from tests.fakes.anticipate import NoOpAnticipate
    from tests.fakes.pressure import FakePressureSignalSink

    store = SqliteArtifactStore(":memory:")
    orch = ArtifactEngineOrchestrator(
        store=store,
        parse_stage=ParseStage(),
        anticipate=NoOpAnticipate(),
        pressure_sink=FakePressureSignalSink(),
    )
    bundle = await orch.run_document(b"hello", "test.txt")
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.pipeline.contracts.models import (
    ArtifactBundle,
    ArtifactType,
    Provenance,
)
from app.pipeline.contracts.ports import (
    AnticipatePort,
    ArtifactStorePort,
    DuelRunnerPort,
    ExtractPort,
    PipelineRunnerPort,
    PressureSignalSink,
)
from app.pipeline.stages.parse import ParseError, ParseResult, ParseStage
from app.pipeline.tools.reflect import ReflectTool

logger = logging.getLogger(__name__)


class ArtifactEngineOrchestrator(PipelineRunnerPort):
    """Orchestrator that runs the full artifact pipeline on a document.

    Constructor parameters:
        store: ``ArtifactStorePort`` implementation for persisting artifacts.
        parse_stage: ``ParseStage`` for converting raw bytes into text.
        anticipate: Optional ``AnticipatePort`` for pre-extraction predictions.
        extract: Optional ``ExtractPort`` for multi-pass extraction.
        duel: Optional ``DuelRunnerPort`` for adversarial two-agent extraction.
        pressure_sink: Optional ``PressureSignalSink`` for pressure events.
        reflect_tool: Optional ``ReflectTool`` for post-extraction validation.
    """

    def __init__(
        self,
        store: ArtifactStorePort,
        parse_stage: ParseStage,
        anticipate: Optional[AnticipatePort] = None,
        extract: Optional[ExtractPort] = None,
        duel: Optional[DuelRunnerPort] = None,
        pressure_sink: Optional[PressureSignalSink] = None,
        reflect_tool: Optional[ReflectTool] = None,
    ) -> None:
        self._store = store
        self._parse_stage = parse_stage
        self._anticipate = anticipate
        self._extract = extract
        self._duel = duel
        self._pressure_sink = pressure_sink
        self._reflect_tool = reflect_tool or ReflectTool()

    # ------------------------------------------------------------------
    # PipelineRunnerPort implementation
    # ------------------------------------------------------------------

    async def run_document(
        self,
        file_content: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactBundle:
        """Run the full extraction pipeline on a document.

        Args:
            file_content: Raw bytes of the uploaded document.
            filename: Original filename for type detection.
            metadata: Optional document-level metadata.

        Returns:
            The primary ``ArtifactBundle`` produced by the pipeline.

        Raises:
            ParseError: If the document cannot be parsed.
        """
        meta = metadata or {}
        document_id = meta.get("document_id", f"doc_{uuid4().hex}")
        source_doc_hash = hashlib.sha256(file_content).hexdigest()

        # 1. Parse
        logger.info("Parsing document %s (%d bytes)", filename, len(file_content))
        parse_result: ParseResult = await self._parse_stage.parse(
            file_content=file_content,
            filename=filename,
        )

        # 2. Anticipate (optional)
        prediction_records: List[Any] = []
        if self._anticipate is not None:
            logger.info("Running anticipate stage for %s", document_id)
            prediction_records = await self._anticipate.run(
                parsed_doc=parse_result,
                document_class=meta.get("document_class", "unknown"),
            )

        # 3. Extract (optional)
        synthesis_result: Any = None
        if self._extract is not None:
            logger.info("Running extract stage for %s", document_id)
            synthesis_result = await self._extract.run(
                parsed_doc=parse_result,
                document_id=document_id,
                source_doc_hash=source_doc_hash,
            )

        # 4. Reflect on extracted fields
        reflection_result: Any = None
        final_fields: List[Any] = []
        all_citations: List[Any] = []
        all_discrepancies: List[Any] = []

        if synthesis_result is not None:
            final_fields = getattr(synthesis_result, "final_fields", [])
            all_citations = getattr(synthesis_result, "all_citations", [])
            all_discrepancies = getattr(synthesis_result, "discrepancies", [])

            logger.info(
                "Running reflection on %d fields for %s",
                len(final_fields),
                document_id,
            )
            reflection_result = self._reflect_tool.reflect_fields(
                fields=final_fields,
                source_text=parse_result.raw_text,
            )

        # 5. Duel (optional)
        duel_state: Any = None
        if self._duel is not None:
            logger.info("Running duel stage for %s", document_id)
            duel_state = await self._duel.run(
                parsed_doc=parse_result,
                document_id=document_id,
                source_doc_hash=source_doc_hash,
            )

        # 6. Build & store the primary artifact bundle
        payload: Dict[str, Any] = {
            "fields": [f.model_dump() if hasattr(f, "model_dump") else f for f in final_fields],
            "parse_result": {
                "document_type": parse_result.document_type,
                "page_count": parse_result.page_count,
            },
        }

        if synthesis_result is not None:
            payload["synthesis_result"] = (
                synthesis_result.model_dump() if hasattr(synthesis_result, "model_dump")
                else synthesis_result
            )
        if reflection_result is not None:
            payload["reflection_result"] = (
                reflection_result.model_dump() if hasattr(reflection_result, "model_dump")
                else reflection_result
            )
        if prediction_records:
            payload["prediction_records"] = [
                r.model_dump() if hasattr(r, "model_dump") else r
                for r in prediction_records
            ]
        if all_discrepancies:
            payload["discrepancies"] = [
                d.model_dump() if hasattr(d, "model_dump") else d
                for d in all_discrepancies
            ]

        # Build provenance
        provenance = Provenance(
            source_doc_hash=source_doc_hash,
            document_id=document_id,
            version=1,
            model_id=meta.get("model_id"),
        )

        bundle = ArtifactBundle(
            artifact_id=f"art_{uuid4().hex}",
            document_id=document_id,
            version=1,
            artifact_type=ArtifactType.EXTRACTION,
            source_doc_hash=source_doc_hash,
            confidence=self._compute_confidence(
                synthesis_result, reflection_result, parse_result
            ),
            payload=payload,
            provenance=provenance,
            citations=all_citations,
            created_at=datetime.now(timezone.utc),
        )

        await self._store.create(bundle)

        # 7. Build and store duel artifact if duel ran
        if duel_state is not None:
            duel_payload = (
                duel_state.model_dump() if hasattr(duel_state, "model_dump")
                else duel_state
            )
            duel_bundle = ArtifactBundle(
                artifact_id=f"art_{uuid4().hex}",
                document_id=document_id,
                version=1,
                artifact_type=ArtifactType.REASONING,
                source_doc_hash=source_doc_hash,
                confidence=0.5,
                payload={"duel_state": duel_payload},
                provenance=Provenance(
                    source_doc_hash=source_doc_hash,
                    document_id=document_id,
                    depends_on=[bundle.artifact_id],
                ),
                created_at=datetime.now(timezone.utc),
            )
            await self._store.create(duel_bundle)

        logger.info(
            "Pipeline complete for %s — artifact %s",
            document_id,
            bundle.artifact_id,
        )
        return bundle

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(
        synthesis_result: Any,
        reflection_result: Any,
        parse_result: ParseResult,
    ) -> float:
        """Derive an overall confidence score from pipeline stage outputs."""
        confidence = 0.5  # baseline

        if synthesis_result is not None:
            sc = getattr(synthesis_result, "confidence", None)
            if sc is not None:
                confidence = float(sc)

        if reflection_result is not None:
            issues = getattr(reflection_result, "issues", [])
            if issues:
                # Knock off 5% per error-severity issue (min 0.10)
                error_count = sum(1 for i in issues if getattr(i, "severity", None) == "error")
                confidence -= error_count * 0.05
                confidence = max(confidence, 0.10)

        return round(confidence, 4)
