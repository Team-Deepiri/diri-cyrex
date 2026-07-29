"""
Artifact Engine API Routes (Prajwala). Stubs backed by FakePipelineRunner for Week 1.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Optional, Any, List
from datetime import datetime
from app.pipeline.voice.corrections import CorrectionStage
from app.pipeline.voice.synthesizer import (
    VoiceQueryResponse as SynthesizerVoiceQueryResponse,
    VoiceSynthesizer,
)
from app.pipeline.contracts.ports import ArtifactStorePort, CorrectionWriterPort

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

# ----------------------------------------------------------------------------
# TODO: swap for real in bootstrap.py and PostgresArtifactStore
# ----------------------------------------------------------------------------

def get_pipeline_runner() -> PipelineRunnerPort:
    from tests.fakes.pipeline_runner import FakePipelineRunner
    return FakePipelineRunner()

def get_artifact_store() -> ArtifactStorePort:
    # TODO: swap for PostgresArtifactStore
    raise RuntimeError("Artifact store dependency is not configured")

def get_correction_writer() -> CorrectionWriterPort:
    # TODO: replace with real PostgresArtifactStore-backed writer. Current implementation is in-memory
    return CorrectionStage()

# ----------------------------------------------------------------------------
# Request / Response Models
# ----------------------------------------------------------------------------

class VoiceQueryRequest(BaseModel):
    document_id: str
    question: str
    persona_scope: PersonaScope = Field(default_factory=PersonaScope)


class VoiceQueryApiResponse(BaseModel):
    success: bool
    response: SynthesizerVoiceQueryResponse

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
    runner: PipelineRunnerPort = Depends(get_pipeline_runner)
):
    """Get an artifact bundle by ID."""
    logger.info("Artifact fetch requested", artifact_id=artifact_id)
    try:
        # TODO: swap for ArtifactStorePort later
        fake_bundle = runner._golden
        if fake_bundle.artifact_id != artifact_id:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return ArtifactResponse(
            success=True,
            artifact=fake_bundle,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Artifact fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artifact_id}/provenance", response_model=ProvenanceResponse)
async def get_provenance(
    artifact_id: str,
    runner: PipelineRunnerPort = Depends(get_pipeline_runner)
):
    """Walk the artifact graph backward to source PDF spans."""
    logger.info("Provenance walk requested", artifact_id=artifact_id)
    try:
        # TODO: swap for graph walk via ArtifactStorePort.get_graph_neighborhood later
        fake_bundle = runner._golden
        return ProvenanceResponse(
            success=True,
            artifact_id=artifact_id,
            provenance=fake_bundle.provenance,
            citations=fake_bundle.citations,
        )
    except Exception as e:
        logger.error(f"Provenance walk failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{artifact_id}/corrections", response_model=CorrectionResponse)
async def submit_correction(
    artifact_id: str,
    request: CorrectionRequest,
    runner: PipelineRunnerPort = Depends(get_pipeline_runner),
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
    logger.info("Voice query received", document_id=request.document_id)
    try:
        synthesizer = VoiceSynthesizer(store=store)
        response = await synthesizer.query(
            document_id=request.document_id,
            question=request.question,
            persona_scope=request.persona_scope,
        )
        return VoiceQueryApiResponse(success=True, response=response)
    except ValueError as e:
        # ex. hard_citation_gate=False
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Voice query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
