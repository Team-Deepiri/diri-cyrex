# Voice + Viz × deepiri-speech

Working branch: `joe_black/feature/voice-and-viz-impl`  
Target: `prajwala-immareddy/feature/voice-and-viz` (PR #139)  
Speech stack: platform `deepiri-speech` + LiveKit + Pipecat (in-process)

## From PR #139 (present on this branch)

- `app/pipeline/voice/synthesizer.py` — document-grounded VoiceSynthesizer
- Artifact Engine UI: Duel Arena, Reckoning Compass, Witness Stitch, Terrain Survey, Fault Drill-Down
- Routes: artifacts, reckoning, voice query path via main

## Platform speech (from joe_black/feature/speech)

- `setup.sh --run` brings up messaging + RTG + **livekit** + **speech**
- cyrex-interface messaging / realtime delivery clients

## Implementation next

1. Wire Voice Query UI → `deepiri-speech` `/v1/stt` + `/v1/tts` (and/or LiveKit room) instead of mock-only confession
2. Keep synthesizer grounding rules; use speech engines for audio I/O
3. Optional: Pipecat WS `/v1/session/ws?protocol=json` for duplex agent turns that call Cyrex + synthesizer
