"""HTTP client for platform deepiri-speech (STT/TTS/LiveKit).

Not the Track C VoiceSynthesizer — that stays document-grounded.
This is the real audio engine: faster-whisper / Kokoro / Pipecat / LiveKit.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from app.settings import settings

logger = logging.getLogger("cyrex.integrations.speech")


class SpeechClient:
    def __init__(self, base_url: Optional[str] = None, timeout: float = 120.0):
        self.base_url = (base_url or settings.SPEECH_URL).rstrip("/")
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(settings.SPEECH_ENABLED)

    async def health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{self.base_url}/health")
            r.raise_for_status()
            return r.json()

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "audio.wav",
        mime_type: str = "audio/wav",
        language: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        data: dict[str, str] = {}
        if language:
            data["language"] = language
        if session_id:
            data["session_id"] = session_id
        files = {"file": (filename, audio, mime_type)}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base_url}/v1/stt", data=data, files=files)
            r.raise_for_status()
            return r.json()

    async def synthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> tuple[bytes, str]:
        body: dict[str, Any] = {"text": text}
        if voice:
            body["voice"] = voice
        if session_id:
            body["session_id"] = session_id
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base_url}/v1/tts", json=body)
            r.raise_for_status()
            return r.content, r.headers.get("content-type", "application/octet-stream")

    async def create_live_session(
        self, *, user_id: Optional[str] = None, room_name: Optional[str] = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if user_id:
            payload["user_id"] = user_id
        if room_name:
            payload["room_name"] = room_name
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{self.base_url}/v1/sessions", json=payload)
            r.raise_for_status()
            return r.json()


_client: Optional[SpeechClient] = None


def get_speech_client() -> SpeechClient:
    global _client
    if _client is None:
        _client = SpeechClient()
    return _client
