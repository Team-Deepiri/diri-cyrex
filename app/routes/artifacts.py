"""
Artifact Engine API Routes — Postgres-backed pipeline runner and store.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from pydantic import BaseModel, Field
from typing import Optional, Any, List
from app.pipeline.contracts.ports import CorrectionWriterPort, ArtifactStorePort
from app.pipeline.contracts.models import LearningArtifact
from app.pipeline.emitters.training_emitter import TrainingEmitter
from app.pipeline.stages.reckoning import emit_learning_artifacts
from app.database.postgres import get_postgres_manager

from ..pipeline.contracts.models import (
    ArtifactBundle,
    Citation,
    PersonaScope,
    Provenance,
)

from ..pipeline.contracts.ports import PipelineRunnerPort
from ..logging_config import get_logger

logger = get_logger("cyrex.api.artifacts")

router = APIRouter(prefix="/api/v1/artifacts", tags=["artifacts"])


def get_pipeline_runner() -> PipelineRunnerPort:
    """Resolved via ``app.dependency_overrides`` in ``main.py`` (Postgres orchestrator)."""
    raise RuntimeError(
        "Pipeline runner not wired — ensure app.dependency_overrides[get_pipeline_runner] is set"
    )


def get_artifact_store() -> ArtifactStorePort:
    """Resolved via ``app.dependency_overrides`` in ``main.py``."""
    raise RuntimeError(
        "Artifact store not wired — ensure app.dependency_overrides[get_artifact_store] is set"
    )


def get_correction_writer() -> CorrectionWriterPort:
    """Resolved via ``app.dependency_overrides`` in ``main.py`` (PostgresCorrectionStore)."""
    raise RuntimeError(
        "Correction writer not wired — ensure app.dependency_overrides[get_correction_writer] is set"
    )

# ----------------------------------------------------------------------------
# Request / Response Models
# ----------------------------------------------------------------------------

class VoiceQueryRequest(BaseModel):
    document_id: str
    question: str  # TODO: confirm field name once VoiceQueryRequest defined
    persona_scope: PersonaScope = Field(default_factory=PersonaScope)


class WitnessSpan(BaseModel):
    citation_id: str
    quote: str
    char_start: int
    char_end: int
    page: Optional[int] = None


class ConfessionGap(BaseModel):
    claim_attempted: str
    reason: str = "no_citation"


class VoiceQueryResponse(BaseModel):
    confessed: bool
    spans: List[WitnessSpan]
    gaps: Optional[List[ConfessionGap]] = None


class VoiceQueryApiResponse(BaseModel):
    success: bool
    response: VoiceQueryResponse


class CorrectionRequest(BaseModel):
    field_name: str
    corrected_value: Any
    corrected_citation: Citation
    actor_id: str

class ArtifactResponse(BaseModel):
    success: bool
    artifact: ArtifactBundle
    uploaded_at: Optional[str] = None

class ProvenanceResponse(BaseModel):
    success: bool
    artifact_id: str
    provenance: Provenance
    citations: List[Citation]


class ArtifactGraphResponse(BaseModel):
    success: bool
    artifact_id: str
    nodes: List[ArtifactBundle]
    edges: List[dict[str, Any]]

class CorrectionResponse(BaseModel):
    success: bool
    artifact_id: str
    field_name: str
    corrected_value: Any
    submitted_at: str


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------

@router.post("/upload", response_model=ArtifactResponse)
async def upload_artifact(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    runner: PipelineRunnerPort = Depends(get_pipeline_runner)
):
    """Upload a document and run the full pipeline. Returns ArtifactBundle."""
    logger.info("Artifact upload requested", filename=file.filename)
    try:
        file_content = await file.read()
        bundle = await runner.run_document(
            file_content=file_content,
            filename=file.filename or "upload",
            metadata={"document_id": document_id} if document_id else None,
        )
        return ArtifactResponse(
            success=True,
            artifact=bundle,
            uploaded_at=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        logger.error(f"Artifact upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artifact_id}", response_model=ArtifactResponse)
async def get_artifact(
    artifact_id: str,
    store: ArtifactStorePort = Depends(get_artifact_store),
):
    """Get an artifact bundle by ID."""
    logger.info("Artifact fetch requested", artifact_id=artifact_id)
    try:
        bundle = await store.get(artifact_id)
        if bundle is None:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return ArtifactResponse(
            success=True,
            artifact=bundle,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Artifact fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artifact_id}/provenance", response_model=ProvenanceResponse)
async def get_provenance(
    artifact_id: str,
    store: ArtifactStorePort = Depends(get_artifact_store),
):
    """Walk the artifact graph backward to source PDF spans."""
    logger.info("Provenance walk requested", artifact_id=artifact_id)
    try:
        bundle = await store.get(artifact_id)
        if bundle is None:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return ProvenanceResponse(
            success=True,
            artifact_id=artifact_id,
            provenance=bundle.provenance,
            citations=bundle.citations,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Provenance walk failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artifact_id}/graph", response_model=ArtifactGraphResponse)
async def get_artifact_graph(
    artifact_id: str,
    hops: int = Query(2, ge=1, le=5),
    store: ArtifactStorePort = Depends(get_artifact_store),
):
    """Artifact dependency neighborhood for Canvas provenance river."""
    get_graph = getattr(store, "get_graph_neighborhood", None)
    if get_graph is None:
        raise HTTPException(status_code=501, detail="Graph walk not supported")
    try:
        bundle = await store.get(artifact_id)
        if bundle is None:
            raise HTTPException(status_code=404, detail="Artifact not found")
        graph = await get_graph(artifact_id, hops=hops)
        return ArtifactGraphResponse(
            success=True,
            artifact_id=artifact_id,
            nodes=graph.get("nodes") or [],
            edges=graph.get("edges") or [],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Artifact graph failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{artifact_id}/corrections", response_model=CorrectionResponse)
async def submit_correction(
    artifact_id: str,
    request: CorrectionRequest,
    correction_writer: CorrectionWriterPort = Depends(get_correction_writer),
):
    """Submit a human correction. Returns a LearningArtifact bundle."""
    logger.info("Correction submitted", artifact_id=artifact_id, field=request.field_name)
    try:
        bundle = await correction_writer.submit_correction(
            artifact_id=artifact_id,
            field_name=request.field_name,
            corrected_value=request.corrected_value,
            corrected_citation=request.corrected_citation,
            actor_id=request.actor_id,
        )
        try:
            learning_raw = (bundle.payload or {}).get("learning_artifact")
            if learning_raw:
                learning = LearningArtifact.model_validate(learning_raw)
                pg = await get_postgres_manager()
                emitter = TrainingEmitter(postgres=pg, producer="cyrex.correction_writer")
                await emit_learning_artifacts([learning], emitter)
        except Exception as exc:
            logger.warning("correction training emit skipped: %s", exc)
        return CorrectionResponse(
            success=True,
            artifact_id=artifact_id,
            field_name=request.field_name,
            corrected_value=request.corrected_value,
            submitted_at=bundle.created_at.isoformat(),
        )
    except Exception as e:
        logger.error(f"Correction submission failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voice/query", response_model=VoiceQueryApiResponse)
async def voice_query(
    request: VoiceQueryRequest,
    store: ArtifactStorePort = Depends(get_artifact_store),
):
    """Voice Q&A — returns only verbatim cited spans or a confession."""
    from app.pipeline.voice.synthesizer import VoiceSynthesizer

    logger.info("Voice query received", document_id=request.document_id)
    try:
        result = await VoiceSynthesizer(store).query(
            document_id=request.document_id,
            question=request.question,
            persona_scope=request.persona_scope,
        )
        spans = [
            WitnessSpan(
                citation_id=s.citation_id,
                quote=s.quote,
                char_start=s.char_start,
                char_end=s.char_end,
                page=s.page,
            )
            for s in result.spans
        ]
        gaps = (
            [ConfessionGap(**g.model_dump()) for g in result.gaps]
            if result.gaps
            else None
        )
        return VoiceQueryApiResponse(
            success=True,
            response=VoiceQueryResponse(
                confessed=result.confessed,
                spans=spans,
                gaps=gaps,
            ),
        )
    except Exception as e:
        logger.error(f"Voice query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
