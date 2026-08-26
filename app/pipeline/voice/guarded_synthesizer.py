"""Guarded voice synthesis — guardrails + synthesizer + dataset export hook."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from diri_agent_guardrails.agi.voice import build_voice_guardrail_engine
from diri_agent_guardrails.core.interfaces import GuardrailEngine

from app.pipeline.contracts.models import PersonaScope
from app.pipeline.contracts.ports import ArtifactStorePort
from app.pipeline.voice.synthesizer import (
    ConfessionGap,
    VoiceQueryResult,
    VoiceSynthesizer,
)

logger = logging.getLogger("cyrex.pipeline.voice.guarded_synthesizer")


class GuardedVoiceSynthesizer:
    """Voice synthesizer with diri-agent-guardrails enforcement."""

    def __init__(
        self,
        store: ArtifactStorePort,
        engine: Optional[GuardrailEngine] = None,
    ) -> None:
        self._store = store
        self._inner = VoiceSynthesizer(store)
        self._engine = engine or build_voice_guardrail_engine()

    def _get_scorer(self):
        from app.pipeline.voice.witness_scorer import get_witness_scorer

        return get_witness_scorer()

    async def query(
        self,
        document_id: str,
        question: str,
        persona_scope: Optional[PersonaScope] = None,
    ) -> VoiceQueryResult:
        scope = persona_scope or PersonaScope()
        pre = self._engine.check(
            question,
            document_id=document_id,
            corpus_filter=scope.corpus_filter,
            witness_set_only=scope.witness_set_only,
        )
        if not pre.passed:
            return VoiceQueryResult(
                document_id=document_id,
                question=question,
                confessed=True,
                spans=[],
                gaps=[
                    ConfessionGap(
                        claim_attempted=question,
                        reason=pre.reason_code.value if pre.reason_code else "guardrail_block",
                    )
                ],
            )

        result = await VoiceSynthesizer(
<<<<<<< HEAD
            self._store, scorer=self._get_scorer()
=======
            self._inner._store, scorer=self._get_scorer()
>>>>>>> origin/dev
        ).query(document_id, question, scope)

        span_dicts = [s.model_dump() for s in result.spans]
        post = self._engine.check(
            " ".join(s.quote for s in result.spans) if result.spans else "",
            question=question,
            spans=span_dicts,
            confessed=result.confessed,
            hard_citation_gate=scope.hard_citation_gate,
        )
        if not post.passed and not result.confessed:
            return VoiceQueryResult(
                document_id=document_id,
                question=question,
                confessed=True,
                spans=[],
                gaps=[
                    ConfessionGap(
                        claim_attempted=question,
                        reason=post.reason_code.value if post.reason_code else "citation_gate",
                    )
                ],
            )

        if result.spans and len(result.spans) > 1:
            ranked = self._get_scorer().rank(
                question, [s.quote for s in result.spans], threshold=0.0
            )
            order = {row["quote"]: i for i, row in enumerate(ranked)}
            result.spans.sort(key=lambda s: order.get(s.quote, 999))

        return result

    async def query_with_speech(
        self,
        *,
        document_id: str,
        question: Optional[str] = None,
        persona_scope: Optional[PersonaScope] = None,
        audio: Optional[bytes] = None,
        audio_mime_type: str = "audio/wav",
        synthesize_audio: bool = True,
        speech_client: Any = None,
    ) -> tuple[VoiceQueryResult, dict[str, Any]]:
        """Guarded grounding with optional deepiri-speech STT/TTS."""
        scope = persona_scope or PersonaScope()

        tts_voice: Optional[str] = None
        if speech_client is not None:
            speech_enabled = True
        else:
            from app.settings import settings

            speech_enabled = bool(getattr(settings, "SPEECH_ENABLED", True))
            tts_voice = getattr(settings, "SPEECH_TTS_VOICE", None)

        meta: dict[str, Any] = {
            "engine": "deepiri-speech",
            "enabled": speech_enabled,
            "stt": None,
            "tts": None,
            "spoken_text": None,
            "audio": None,
            "audio_mime_type": None,
        }

        def _client() -> Any:
            if speech_client is not None:
                return speech_client
            from app.integrations.speech_client import get_speech_client

            return get_speech_client()

        q = (question or "").strip()
        if audio:
            if not speech_enabled:
                raise RuntimeError("audio STT requires SPEECH_ENABLED and deepiri-speech")
            stt = await _client().transcribe(
                audio,
                mime_type=audio_mime_type,
                session_id=document_id,
            )
            q = (stt.get("text") or "").strip()
            meta["stt"] = {
                "provider": stt.get("provider"),
                "model": stt.get("model"),
                "text": q,
            }
            if not q:
                resp = VoiceQueryResult(
                    document_id=document_id,
                    question="",
                    confessed=True,
                    gaps=[
                        ConfessionGap(
                            claim_attempted="",
                            reason="STT returned empty transcript",
                        )
                    ],
                )
                meta["spoken_text"] = resp.spoken_text()
                return resp, meta

        if not q:
            resp = VoiceQueryResult(
                document_id=document_id,
                question="",
                confessed=True,
                gaps=[
                    ConfessionGap(
                        claim_attempted="",
                        reason="question or audio required",
                    )
                ],
            )
            meta["spoken_text"] = resp.spoken_text()
            return resp, meta

        response = await self.query(document_id, q, scope)
        spoken = response.spoken_text()
        meta["spoken_text"] = spoken

        if synthesize_audio and spoken and speech_enabled:
            try:
                audio_out, mime = await _client().synthesize(
                    spoken,
                    voice=tts_voice,
                    session_id=document_id,
                )
                meta["audio"] = audio_out
                meta["audio_mime_type"] = mime
                meta["tts"] = {
                    "voice": tts_voice,
                    "bytes": len(audio_out),
                    "mime": mime,
                }
            except Exception as exc:
                logger.warning("TTS via deepiri-speech failed: %s", exc)
                meta["tts"] = {"error": str(exc)}

        return response, meta

    async def to_training_records(
        self, result: VoiceQueryResult, *, document_id: str
    ) -> List[dict[str, Any]]:
        """Export witness spans as structured training rows."""
        rows: List[dict[str, Any]] = []
        for span in result.spans:
            rows.append(
                {
                    "instruction": f"Witness for document {document_id}",
                    "input": "",
                    "output": span.quote,
                    "text": span.quote,
                    "category": "voice_witness",
                    "quality_score": 0.95,
                    "producer": "cyrex.voice",
                    "metadata": {
                        "document_id": document_id,
                        "citation_id": span.citation_id,
                        "char_start": span.char_start,
                        "char_end": span.char_end,
                    },
                }
            )
        return rows
