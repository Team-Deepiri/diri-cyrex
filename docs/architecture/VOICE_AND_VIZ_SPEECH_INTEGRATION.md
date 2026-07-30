# Voice + Viz × deepiri-speech

Working branch: `joe_black/feature/voice-and-viz-impl`  
Target: `prajwala-immareddy/feature/voice-and-viz` (PR #139)  
Speech stack: platform **deepiri-speech** (Pipecat in-process + LiveKit SFU)

## Division of labor

| Piece | Owner |
|-------|--------|
| Document grounding (verbatim quotes / confession) | `VoiceSynthesizer` (Track C — keep) |
| STT / TTS / LiveKit / Pipecat | `deepiri-speech` via `app/integrations/speech_client.py` |
| UI | Witness Stitch — Ask / Mic / Speak |

## VoiceSynthesizer

`app/pipeline/voice/synthesizer.py`:

- **Matching** — token overlap on field name / quote / value, lease-field aliases (`rent` → `base_rent`, etc.), multi-span (up to `max_spans`)
- **`spoken_text()`** — TTS-ready string from verbatim quotes or confession reason (never invents facts)
- **`speech_payload()`** — structured meta for UI / speech clients
- **`query_with_speech(...)`** — optional STT → `query` → optional TTS through deepiri-speech

Route `POST /api/v1/artifacts/voice/query` calls `query_with_speech` (not duplicated STT/TTS in the route).

## Wired paths

1. `POST /api/v1/artifacts/voice/query`
   - optional `audio_b64` → speech `/v1/stt`
   - grounded answer via `VoiceSynthesizer`
   - `synthesize_audio` → speech `/v1/tts` → `audio_b64` in response
2. `GET /api/v1/artifacts/voice/speech-health` → speech `/health`
3. `POST /api/v1/artifacts/voice/session` → speech `/v1/sessions` (LiveKit token)

## Env

```bash
SPEECH_ENABLED=1
SPEECH_URL=http://speech:5020          # in-compose
SPEECH_PUBLIC_URL=http://localhost:5020
LIVEKIT_PUBLIC_URL=ws://localhost:7880
```

Platform PR: https://github.com/Team-Deepiri/deepiri-platform/pull/302
