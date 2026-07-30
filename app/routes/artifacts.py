"""
Artifact Engine API Routes (Track C) — voice query wired to deepiri-speech.

VoiceSynthesizer stays document-grounded (verbatim citations / confession).
Audio I/O goes through platform deepiri-speech (STT/TTS / LiveKit / Pipecat).
"""
from __future__ import annotations

import base64
from datetime import datetime
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.integrations.speech_client import get_speech_client
from app.pipeline.contracts.models import Citation, PersonaScope
from app.pipeline.contracts.ports import (
    ArtifactStorePort,
    CorrectionWriterPort,
    PipelineRunnerPort,
)
from app.pipeline.voice.corrections import CorrectionStage
from app.pipeline.voice.synthesizer import (
    VoiceQueryResponse as SynthesizerVoiceQueryResponse,
    VoiceSynthesizer,
)
from app.settings import settings

from ..logging_config import get_logger

logger = get_logger("cyrex.api.artifacts")

router = APIRouter(prefix="/api/v1/artifacts", tags=["artifacts"])


def get_pipeline_runner() -> PipelineRunnerPort:
    from tests.fakes.pipeline_runner import FakePipelineRunner

    return FakePipelineRunner()


def get_artifact_store() -> ArtifactStorePort:
    raise RuntimeError("Artifact store dependency is not configured")


def get_correction_writer() -> CorrectionWriterPort:
    return CorrectionStage()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ArtifactResponse(BaseModel):
    success: bool
    artifact: Optional[Any] = None
    uploaded_at: Optional[str] = None


class ProvenanceResponse(BaseModel):
    success: bool
    artifact_id: str
    provenance: Any = None
    citations: List[Any] = Field(default_factory=list)


class CorrectionRequest(BaseModel):
    field_name: str
    corrected_value: Any
    corrected_citation: Optional[Citation] = None
    actor_id: str = "anonymous"


class CorrectionResponse(BaseModel):
    success: bool
    artifact_id: str
    field_name: str
    corrected_value: Any
    submitted_at: str


class VoiceQueryRequest(BaseModel):
    document_id: str
    question: Optional[str] = None
    # Optional mic capture — base64 audio; STT via deepiri-speech when set
    audio_b64: Optional[str] = None
    audio_mime_type: str = "audio/webm"
    persona_scope: PersonaScope = Field(default_factory=PersonaScope)
    synthesize_audio: bool = True


class VoiceQueryApiResponse(BaseModel):
    success: bool
    response: SynthesizerVoiceQueryResponse
    spoken_text: Optional[str] = None
    audio_b64: Optional[str] = None
    audio_mime_type: Optional[str] = None
    speech: Optional[dict[str, Any]] = None
    question_used: Optional[str] = None


def _spoken_text_from_response(response: SynthesizerVoiceQueryResponse) -> str:
    if not response.confessed and response.spans:
        return " ".join(s.quote for s in response.spans if s.quote)
    if response.gaps:
        return response.gaps[0].reason or "I could not ground that claim in the document."
    return "No answer found."


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/voice/speech-health")
async def voice_speech_health():
    """Probe platform deepiri-speech (Pipecat + LiveKit status nested in /health)."""
    if not settings.SPEECH_ENABLED:
        return {"ok": False, "enabled": False, "error": "SPEECH_ENABLED=0"}
    try:
        health = await get_speech_client().health()
        return {
            "ok": health.get("status") == "healthy",
            "enabled": True,
            "speech_url": settings.SPEECH_PUBLIC_URL,
            "livekit_url": settings.LIVEKIT_PUBLIC_URL,
            "speech": health,
        }
    except Exception as exc:
        logger.warning("speech health failed: %s", exc)
        return {"ok": False, "enabled": True, "error": str(exc)}


@router.post("/voice/session")
async def voice_live_session(user_id: Optional[str] = None, room_name: Optional[str] = None):
    """Mint a LiveKit + WS session via deepiri-speech for realtime duplex."""
    if not settings.SPEECH_ENABLED:
        raise HTTPException(status_code=503, detail="Speech engine disabled")
    try:
        return await get_speech_client().create_live_session(
            user_id=user_id, room_name=room_name
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"speech session failed: {exc}") from exc


@router.post("/upload", response_model=ArtifactResponse)
async def upload_artifact(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    runner: PipelineRunnerPort = Depends(get_pipeline_runner),
):
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


@router.post("/voice/query", response_model=VoiceQueryApiResponse)
async def voice_query(
    request: VoiceQueryRequest,
    store: ArtifactStorePort = Depends(get_artifact_store),
):
    """
    Document-grounded Q&A + optional deepiri-speech STT/TTS.

    1. If audio_b64 → STT via speech service
    2. VoiceSynthesizer → verbatim spans or confession (never fabricates)
    3. If synthesize_audio → TTS spoken answer via speech service
    """
    logger.info("Voice query received", document_id=request.document_id)
    speech_meta: dict[str, Any] = {
        "engine": "deepiri-speech",
        "enabled": settings.SPEECH_ENABLED,
        "stt": None,
        "tts": None,
    }
    question = (request.question or "").strip()

    try:
        if request.audio_b64:
            if not settings.SPEECH_ENABLED:
                raise HTTPException(
                    status_code=503,
                    detail="audio_b64 requires SPEECH_ENABLED and deepiri-speech",
                )
            raw = base64.b64decode(request.audio_b64)
            stt = await get_speech_client().transcribe(
                raw,
                mime_type=request.audio_mime_type,
                session_id=request.document_id,
            )
            question = (stt.get("text") or "").strip()
            speech_meta["stt"] = {
                "provider": stt.get("provider"),
                "model": stt.get("model"),
                "chars": len(question),
            }
            if not question:
                raise HTTPException(status_code=400, detail="STT returned empty transcript")

        if not question:
            raise HTTPException(
                status_code=400, detail="question or audio_b64 required"
            )

        synthesizer = VoiceSynthesizer(store=store)
        response = await synthesizer.query(
            document_id=request.document_id,
            question=question,
            persona_scope=request.persona_scope,
        )

        spoken = _spoken_text_from_response(response)
        audio_b64: Optional[str] = None
        audio_mime: Optional[str] = None

        if request.synthesize_audio and spoken and settings.SPEECH_ENABLED:
            try:
                audio, mime = await get_speech_client().synthesize(
                    spoken,
                    voice=settings.SPEECH_TTS_VOICE,
                    session_id=request.document_id,
                )
                audio_b64 = base64.b64encode(audio).decode("ascii")
                audio_mime = mime
                speech_meta["tts"] = {
                    "voice": settings.SPEECH_TTS_VOICE,
                    "bytes": len(audio),
                    "mime": mime,
                }
            except Exception as tts_exc:
                logger.warning("TTS skipped: %s", tts_exc)
                speech_meta["tts"] = {"error": str(tts_exc)}

        return VoiceQueryApiResponse(
            success=True,
            response=response,
            spoken_text=spoken,
            audio_b64=audio_b64,
            audio_mime_type=audio_mime,
            speech=speech_meta,
            question_used=question,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Voice query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artifact_id}", response_model=ArtifactResponse)
async def get_artifact(
    artifact_id: str,
    runner: PipelineRunnerPort = Depends(get_pipeline_runner),
):
    logger.info("Artifact fetch requested", artifact_id=artifact_id)
    try:
        fake_bundle = runner._golden
        if fake_bundle.artifact_id != artifact_id:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return ArtifactResponse(success=True, artifact=fake_bundle)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Artifact fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artifact_id}/provenance", response_model=ProvenanceResponse)
async def get_provenance(
    artifact_id: str,
    runner: PipelineRunnerPort = Depends(get_pipeline_runner),
):
    logger.info("Provenance walk requested", artifact_id=artifact_id)
    try:
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
    correction_writer: CorrectionWriterPort = Depends(get_correction_writer),
):
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
