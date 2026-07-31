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

## Persistence (postgres-cyrex, not SQLite)

Runtime DI wires:

- `PostgresArtifactStore` → `cyrex.artifacts` / `artifact_refs` / `citations`
- `PostgresCorrectionStore` → `cyrex.learning_artifacts`

**Relation to Track A [#128](https://github.com/Team-Deepiri/diri-cyrex/pull/128):** Tyler’s Weeks 1–2 keep `SqliteArtifactStore` for orchestrator unit tests. Both implement `ArtifactStorePort`. Production (`app.main`) uses Postgres against `postgres-cyrex`. Postgres store accepts the same optional `pressure_sink` hook so #128’s projector plugs in when merged.

## VoiceSynthesizer

- Scored matching + aliases + multi-span
- `spoken_text()` / `speech_payload()` / `query_with_speech()` (STT → ground → TTS)

## Wired paths

1. `POST /api/v1/artifacts/voice/query` — STT/ground/TTS
2. `GET /api/v1/artifacts/voice/speech-health`
3. `POST /api/v1/artifacts/voice/session` — deepiri-speech LiveKit + Cyrex `SessionManager` + realtime `CONNECTION` event

## Env

```bash
SPEECH_ENABLED=1
SPEECH_URL=http://speech:5020
SPEECH_PUBLIC_URL=http://localhost:5020
LIVEKIT_PUBLIC_URL=ws://localhost:7880
POSTGRES_HOST=postgres-cyrex
POSTGRES_PORT=5432
POSTGRES_DB=cyrex_db
```

Platform speech PR: https://github.com/Team-Deepiri/deepiri-platform/pull/302
