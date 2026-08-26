"""
Artifact Engine API Routes — Postgres-backed pipeline + deepiri-speech voice.

Voice grounding stays citation-gated (GuardedVoiceSynthesizer).
Audio I/O goes through platform deepiri-speech (STT/TTS / LiveKit / Pipecat).
"""
from __future__ import annotations

import base64
from datetime import datetime
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from app.database.postgres import get_postgres_manager
from app.integrations.speech_client import get_speech_client
from app.pipeline.contracts.models import LearningArtifact
from app.pipeline.contracts.ports import ArtifactStorePort, CorrectionWriterPort
from app.pipeline.emitters.training_emitter import TrainingEmitter
from app.pipeline.stages.reckoning import emit_learning_artifacts
from app.settings import settings

from ..logging_config import get_logger
from ..pipeline.contracts.models import (
    ArtifactBundle,
    Citation,
    ConfessionGap,
    PersonaScope,
    Provenance,
    WitnessSpan,
)
from ..pipeline.contracts.ports import PipelineRunnerPort

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
    question: Optional[str] = None
    # Optional mic capture — base64 audio; STT via deepiri-speech when set
    audio_b64: Optional[str] = None
    audio_mime_type: str = "audio/webm"
    persona_scope: PersonaScope = Field(default_factory=PersonaScope)
    synthesize_audio: bool = True


class VoiceQueryResponse(BaseModel):
    confessed: bool
    spans: List[WitnessSpan]
    gaps: Optional[List[ConfessionGap]] = None


class VoiceQueryApiResponse(BaseModel):
    success: bool
    response: VoiceQueryResponse
    spoken_text: Optional[str] = None
    audio_b64: Optional[str] = None
    audio_mime_type: Optional[str] = None
    speech: Optional[dict[str, Any]] = None
    question_used: Optional[str] = None


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
# Routes — static /voice/* before /{artifact_id}
# ----------------------------------------------------------------------------


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
    except Exception:
        logger.exception("speech health failed")
        return {"ok": False, "enabled": True, "error": "Speech health check failed"}


@router.post("/voice/session")
async def voice_live_session(
    user_id: Optional[str] = None, room_name: Optional[str] = None
):
    """Mint a LiveKit + WS session via deepiri-speech; persist Cyrex session + emit realtime."""
    if not settings.SPEECH_ENABLED:
        raise HTTPException(status_code=503, detail="Speech engine disabled")
    try:
        speech_session = await get_speech_client().create_live_session(
            user_id=user_id, room_name=room_name
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"speech session failed: {exc}") from exc

    cyrex_session_id: Optional[str] = None
    try:
        from app.core.session_manager import get_session_manager

        sm = await get_session_manager()
        if not getattr(sm, "_cleanup_task", None):
            await sm.initialize()
        session = await sm.create_session(
            user_id=user_id,
            metadata={
                "kind": "voice",
                "speech": speech_session,
                "room_name": speech_session.get("room_name") or room_name,
            },
            context={
                "livekit_url": speech_session.get("livekit_url")
                or speech_session.get("url"),
                "ws_url": speech_session.get("ws_url"),
            },
        )
        cyrex_session_id = session.session_id
    except Exception as exc:
        logger.warning("cyrex session persist skipped: %s", exc)

    try:
        from app.integrations.realtime_streaming import (
            StreamEventType,
            get_stream_publisher,
        )

        publisher = await get_stream_publisher()
        await publisher.connect()
        await publisher.publish_event(
            event_type=StreamEventType.CONNECTION,
            data={
                "kind": "voice_session",
                "speech": speech_session,
                "cyrex_session_id": cyrex_session_id,
            },
            channel=f"voice:{cyrex_session_id or speech_session.get('session_id') or 'anon'}",
            session_id=cyrex_session_id,
        )
    except Exception as exc:
        logger.warning("voice session realtime publish skipped: %s", exc)

    return {
        **speech_session,
        "cyrex_session_id": cyrex_session_id,
    }


@router.post("/voice/query", response_model=VoiceQueryApiResponse)
async def voice_query(
    request: VoiceQueryRequest,
    store: ArtifactStorePort = Depends(get_artifact_store),
):
    """Document-grounded Q&A + deepiri-speech STT/TTS via GuardedVoiceSynthesizer."""
    from app.pipeline.voice.guarded_synthesizer import GuardedVoiceSynthesizer

    logger.info("Voice query received", document_id=request.document_id)
    if not (request.question or "").strip() and not request.audio_b64:
        raise HTTPException(status_code=400, detail="question or audio_b64 required")

    try:
        audio_bytes = base64.b64decode(request.audio_b64) if request.audio_b64 else None
        result, speech_meta = await GuardedVoiceSynthesizer(store).query_with_speech(
            document_id=request.document_id,
            question=request.question,
            persona_scope=request.persona_scope,
            audio=audio_bytes,
            audio_mime_type=request.audio_mime_type,
            synthesize_audio=request.synthesize_audio,
            speech_client=get_speech_client() if settings.SPEECH_ENABLED else None,
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

        audio_b64: Optional[str] = None
        audio_mime = speech_meta.get("audio_mime_type")
        raw_audio = speech_meta.get("audio")
        if isinstance(raw_audio, (bytes, bytearray)) and raw_audio:
            audio_b64 = base64.b64encode(raw_audio).decode("ascii")

        public_speech = {
            "engine": speech_meta.get("engine"),
            "enabled": speech_meta.get("enabled"),
            "stt": speech_meta.get("stt"),
            "tts": speech_meta.get("tts"),
            "payload": result.speech_payload(),
        }

        return VoiceQueryApiResponse(
            success=True,
            response=VoiceQueryResponse(
                confessed=result.confessed,
                spans=spans,
                gaps=gaps,
            ),
            spoken_text=speech_meta.get("spoken_text") or result.spoken_text(),
            audio_b64=audio_b64,
            audio_mime_type=audio_mime,
            speech=public_speech,
            question_used=result.question or request.question,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Voice query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=ArtifactResponse)
async def upload_artifact(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    runner: PipelineRunnerPort = Depends(get_pipeline_runner),
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
