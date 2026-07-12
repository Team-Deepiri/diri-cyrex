"""
Artifact Engine API Routes (Prajwala). Stubs backed by FakePipelineRunner for Week 1.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Optional, Any, List
from datetime import datetime

from ..pipeline.contracts.models import (
    Citation,
    PersonaScope,
)

from ..pipeline.contracts.ports import PipelineRunnerPort
from ..logging_config import get_logger
from tests.fakes.pipeline_runner import FakePipelineRunner

logger = get_logger("cyrex.api.artifacts")

router = APIRouter(prefix="/api/v1/artifacts", tags=["artifacts"])

# ----------------------------------------------------------------------------
# TODO: swap for real in bootstrap.py (Week 5)
# ----------------------------------------------------------------------------

def get_pipeline_runner() -> PipelineRunnerPort:
    return FakePipelineRunner()

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
    pass  # TODO: define when building voice/synthesizer.py


class VoiceQueryResponse(BaseModel):
    confessed: bool
    spans: List[WitnessSpan]
    gaps: Optional[List[ConfessionGap]] = None


class CorrectionRequest(BaseModel):
    field_name: str
    corrected_value: Any
    corrected_citation: Citation
    actor_id: str


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------

@router.post("/upload")
async def upload_artifact(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
):
    """Upload a document and run the full pipeline. Returns ArtifactBundle."""
    logger.info("Artifact upload requested", filename=file.filename)
    try:
        runner = get_pipeline_runner()
        file_content = await file.read()
        bundle = await runner.run_document(
            file_content=file_content,
            filename=file.filename or "upload",
            metadata={"document_id": document_id} if document_id else None,
        )
        return {
            "success": True,
            "artifact": bundle.model_dump(),
            "uploaded_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Artifact upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artifact_id}")
async def get_artifact(artifact_id: str):
    """Get an artifact bundle by ID."""
    logger.info("Artifact fetch requested", artifact_id=artifact_id)
    try:
        # TODO: swap for ArtifactStorePort later
        fake_bundle = FakePipelineRunner()._golden
        if fake_bundle.artifact_id != artifact_id:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return {
            "success": True,
            "artifact": fake_bundle.model_dump(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Artifact fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artifact_id}/provenance")
async def get_provenance(artifact_id: str):
    """Walk the artifact graph backward to source PDF spans."""
    logger.info("Provenance walk requested", artifact_id=artifact_id)
    try:
        # TODO: swap for graph walk via ArtifactStorePort.get_graph_neighborhood later
        fake_bundle = FakePipelineRunner()._golden
        return {
            "success": True,
            "artifact_id": artifact_id,
            "provenance": fake_bundle.provenance.model_dump(),
            "citations": [c.model_dump() for c in fake_bundle.citations],
        }
    except Exception as e:
        logger.error(f"Provenance walk failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{artifact_id}/corrections")
async def submit_correction(artifact_id: str, request: CorrectionRequest):
    """Submit a human correction. Returns a LearningArtifact bundle."""
    logger.info("Correction submitted", artifact_id=artifact_id, field=request.field_name)
    try:
        # TODO: swap for real CorrectionWriterPort later
        return {
            "success": True,
            "artifact_id": artifact_id,
            "field_name": request.field_name,
            "corrected_value": request.corrected_value,
            "submitted_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Correction submission failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voice/query")
async def voice_query(request: VoiceQueryRequest):
    """Voice Q&A — returns only verbatim cited spans or a confession."""
    logger.info("Voice query received", document_id=request.document_id)
    try:
        # TODO: swap for real voice/synthesizer.py later
        fake_span = WitnessSpan(
            citation_id="cit_fake_001",
            quote="The base rent shall be $4,500 per month.",
            char_start=1042,
            char_end=1080,
            page=1,
        )
        return VoiceQueryResponse(
            confessed=False,
            spans=[fake_span],
            gaps=None,
        )
    except Exception as e:
        logger.error(f"Voice query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))