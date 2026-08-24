"""Artifact Engine Orchestrator — implements PipelineRunnerPort.

Wires parse → optional anticipate/extract/duel → reflect → Postgres store
(optional pressure emit happens inside the store via its ``PressureSignalSink``).

Usage::

    from app.pipeline.orchestrator import ArtifactEngineOrchestrator
    from app.pipeline.registry.postgres_store import PostgresArtifactStore
    from app.pipeline.stages.parse import ParseStage
    from app.database.postgres import get_postgres_manager

    store = PostgresArtifactStore(await get_postgres_manager())
    await store.ensure_schema()
    orch = ArtifactEngineOrchestrator(store=store, parse_stage=ParseStage())
    bundle = await orch.run_document(b"hello", "test.txt")
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.pipeline.contracts.models import (
    ArtifactBundle,
    ArtifactType,
    Provenance,
    ReflectionResult,
    SynthesisResult,
)
from app.pipeline.contracts.ports import (
    AnticipatePort,
    ArtifactStorePort,
    DuelRunnerPort,
    ExtractPort,
    PipelineRunnerPort,
)
from app.pipeline.stages.parse import ParseResult, ParseStage
from app.pipeline.tools.confidence import ConfidenceCalculator
from app.pipeline.tools.reflect import ReflectTool

try:
    from app.pipeline.stages.reckoning import ReckoningStage
except ImportError:  # pragma: no cover
    ReckoningStage = None  # type: ignore

logger = logging.getLogger(__name__)


def _serialize_if_pydantic(obj: Any) -> Any:
    """Dump Pydantic models; pass through plain values."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


class ArtifactEngineOrchestrator(PipelineRunnerPort):
    """Orchestrator that runs the full artifact pipeline on a document."""

    def __init__(
        self,
        store: ArtifactStorePort,
        parse_stage: ParseStage,
        anticipate: Optional[AnticipatePort] = None,
        extract: Optional[ExtractPort] = None,
        duel: Optional[DuelRunnerPort] = None,
        reckoning: Any = None,
        reflect_tool: Optional[ReflectTool] = None,
        confidence_calculator: Optional[ConfidenceCalculator] = None,
    ) -> None:
        self._store = store
        self._parse_stage = parse_stage
        self._anticipate = anticipate
        self._extract = extract
        self._duel = duel
        self._reckoning = reckoning or (ReckoningStage() if ReckoningStage else None)
        self._reflect_tool = reflect_tool or ReflectTool()
        self._confidence_calculator = confidence_calculator or ConfidenceCalculator()

    async def run_document(
        self,
        file_content: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
    ) -> ArtifactBundle:
        """Run the full extraction pipeline on a document.

        Args:
            file_content: Raw document bytes.
            filename: Source filename (extension drives parse behavior).
            metadata: Optional run context (``document_id``, ``document_class``,
                ``model_id``).
            timeout: Optional overall pipeline timeout in seconds. When exceeded,
                ``asyncio.TimeoutError`` propagates and the run is cancelled.
        """
        if timeout is not None:
            return await asyncio.wait_for(
                self._run_pipeline(file_content, filename, metadata or {}),
                timeout=timeout,
            )
        return await self._run_pipeline(file_content, filename, metadata or {})

    async def _run_pipeline(
        self,
        file_content: bytes,
        filename: str,
        meta: Dict[str, Any],
    ) -> ArtifactBundle:
        document_id = meta.get("document_id", f"doc_{uuid4().hex}")
        source_doc_hash = hashlib.sha256(file_content).hexdigest()
        now = datetime.now(timezone.utc)

        logger.info("Parsing document %s (%d bytes)", filename, len(file_content))
        parse_result: ParseResult = await self._parse_stage.parse(
            file_content=file_content,
            filename=filename,
        )

        prediction_records: List[Any] = []
        if self._anticipate is not None:
            logger.info("Running anticipate stage for %s", document_id)
            prediction_records = await self._anticipate.run(
                parsed_doc=parse_result,
                document_class=meta.get("document_class", "unknown"),
            )

        synthesis_result: Optional[SynthesisResult] = None
        if self._extract is not None:
            logger.info("Running extract stage for %s", document_id)
            synthesis_result = await self._extract.run(
                parsed_doc=parse_result,
                document_id=document_id,
                source_doc_hash=source_doc_hash,
            )

        reflection_result: Optional[ReflectionResult] = None
        final_fields: List[Any] = []
        all_citations: List[Any] = []
        all_discrepancies: List[Any] = []

        if synthesis_result is not None:
            final_fields = list(synthesis_result.final_fields or [])
            all_citations = list(synthesis_result.all_citations or [])
            all_discrepancies = list(synthesis_result.discrepancies or [])

            if self._reckoning is not None and prediction_records:
                logger.info("Running reckoning stage for %s", document_id)
                prediction_records = await self._reckoning.run(
                    prediction_records, final_fields
                )

            logger.info(
                "Running reflection on %d fields for %s",
                len(final_fields),
                document_id,
            )
            reflection_result = self._reflect_tool.reflect_fields(
                fields=final_fields,
                source_text=parse_result.raw_text,
            )

        duel_state: Any = None
        if self._duel is not None:
            logger.info("Running duel stage for %s", document_id)
            duel_state = await self._duel.run(
                parsed_doc=parse_result,
                document_id=document_id,
                source_doc_hash=source_doc_hash,
            )

        payload: Dict[str, Any] = {
            "fields": [_serialize_if_pydantic(f) for f in final_fields],
            "parse_result": {
                "document_type": parse_result.document_type,
                "page_count": parse_result.page_count,
            },
        }

        if synthesis_result is not None:
            payload["synthesis_result"] = _serialize_if_pydantic(synthesis_result)
        if reflection_result is not None:
            payload["reflection_result"] = _serialize_if_pydantic(reflection_result)
        if prediction_records:
            payload["prediction_records"] = [
                _serialize_if_pydantic(r) for r in prediction_records
            ]
        if all_discrepancies:
            payload["discrepancies"] = [
                _serialize_if_pydantic(d) for d in all_discrepancies
            ]

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
            confidence=self._confidence_calculator.compute(
                synthesis_result, reflection_result, parse_result
            ),
            payload=payload,
            provenance=provenance,
            citations=all_citations,
            created_at=now,
        )

        await self._store.create(bundle)

        if duel_state is not None:
            duel_bundle = ArtifactBundle(
                artifact_id=f"art_{uuid4().hex}",
                document_id=document_id,
                version=1,
                artifact_type=ArtifactType.REASONING,
                source_doc_hash=source_doc_hash,
                confidence=0.5,
                payload={"duel_state": _serialize_if_pydantic(duel_state)},
                provenance=Provenance(
                    source_doc_hash=source_doc_hash,
                    document_id=document_id,
                    depends_on=[bundle.artifact_id],
                ),
                created_at=now,
            )
            await self._store.create(duel_bundle)

        logger.info(
            "Pipeline complete for %s — artifact %s",
            document_id,
            bundle.artifact_id,
        )
        return bundle
